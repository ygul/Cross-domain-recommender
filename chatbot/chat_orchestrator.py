from __future__ import annotations

from typing import Literal, Optional, Tuple, Set

from query_embedder import QueryEmbedder
from vector_store import VectorStore
import llm_adapter
from results_formatter import format_results

# VERANDERING: 'Turn' toegevoegd aan de imports
from clarify_gate import ClarifyGate, ElicitationResult, Turn
from elicitation_logger import ElicitationLogger


ItemType = Literal["Book", "TV series", "movie"]

class ChatOrchestrator:
    def __init__(
        self,
        llm_provider: Literal["openai", "huggingface"] = "openai",
        max_items: int = 3,
        enable_clarify_gate: bool = True,
        clarify_max_questions: int = 2,
        enable_logging: bool = True,
        logs_dir: str = "logs",
    ) -> None:
        self.llm_adapter = llm_adapter.create_llm_adapter(provider=llm_provider)
        self.vector_store = VectorStore()
        self.vector_store_alternative = VectorStore(use_alternative_collection=True)

        def _build_embedder(store: VectorStore) -> QueryEmbedder:
            meta = store._collection.metadata or {}
            model_name = meta.get("embedding_model") or "paraphrase-multilingual-MiniLM-L12-v2"
            return QueryEmbedder(model_name=model_name, llm_adapter=self.llm_adapter)

        self.embedder_primary = _build_embedder(self.vector_store)
        self.embedder_alternative = _build_embedder(self.vector_store_alternative)
        self.max_items = max_items

        self.clarify_gate: Optional[ClarifyGate] = (
            ClarifyGate(llm=self.llm_adapter, max_questions=clarify_max_questions)
            if enable_clarify_gate
            else None
        )

        self.logger: Optional[ElicitationLogger] = (
            ElicitationLogger(base_dir=logs_dir) if enable_logging else None
        )
        
        self.last_search_results = []

    def run_once(
        self,
        user_input: str,
        item_types: Set[ItemType] | None = None,
        use_alternative_collection: bool | None = None,
    ) -> str:
        """
        Retrieve + format results for one query.
        """
        effective_alt = bool(use_alternative_collection) if use_alternative_collection is not None else False
        store = self.vector_store_alternative if effective_alt else self.vector_store
        embedder = self.embedder_alternative if effective_alt else self.embedder_primary

        embedding = embedder.embed(user_input)

        if item_types is None:
            results = store.similarity_search(embedding, k=self.max_items)
        else:
            where_filter = (
                {"$or": [{"item_type": item_type} for item_type in item_types]}
                if len(item_types) > 1
                else {"item_type": list(item_types)[0]}
            )
            results = store.filtered_similarity_search(
                embedding,
                k=self.max_items,
                where=where_filter,
            )

        self.last_search_results = results
        return format_results(user_input, results, self.llm_adapter)

    def chat(
        self,
        seed: str,
        item_types: Set[ItemType] | None = None,
        input_fn=input,
        print_fn=print,
        use_alternative_collection: bool | None = None,
    ) -> Tuple[str, Optional[ElicitationResult]]:
        """
        Full chat flow WITH Iterative Search & Query Rewriting.
        """
        seed = (seed or "").strip()
        if seed == "":
            return "Empty input.", None

        elicited_turns = []
        final_query = seed
        
        # 1. Eerste keer zoeken (Initial Context)
        current_context_formatted = self.run_once(seed, item_types, use_alternative_collection)
        
        history = [seed]

        if self.clarify_gate is not None:
            # Loop voor max 2 vragen
            for i in range(2): 
                
                # A. Genereer vraag
                question = self.clarify_gate.generate_single_turn(seed, current_context_formatted, history)
                
                if "NO_QUESTION" in question:
                    print_fn("[DEBUG] Gate decided to stop asking questions.")
                    break
                
                print_fn(f"\n[Bot]: {question}")
                # Let op: we voegen hem nog niet toe aan elicited_turns, dat doen we pas als we het antwoord hebben

                # B. Vang antwoord
                user_response = input_fn(question)
                print_fn(f"[DEBUG] User answer: {user_response}")

                # VERANDERING: Maak een Turn object aan!
                current_turn = Turn(question=question, answer=user_response)
                elicited_turns.append(current_turn)
                
                # --- C. QUERY REWRITING ---
                system_prompt = (
                    "Act as a Search Optimizer for a recommender system.\n"
                    "Combine the 'Core Subject' from the Original Request with the 'Style/Mood' from the User Feedback.\n"
                    "RULES:\n"
                    "1. PRESERVE the main noun/subject from the Original Request (e.g. if user asked for 'lawyer', you MUST include 'lawyer').\n"
                    "2. Add the style keywords from the feedback.\n"
                    "3. Remove conversational filler.\n"
                    "4. Output ONLY the new query string."
                )

                user_prompt = (
                    f"Original Request (Core Subject): {seed}\n"
                    f"User Feedback (Style/Mood): {user_response}\n\n"
                    f"Optimized Query:"
                )

                #system_prompt = (
                #    "Act as a Search Optimizer.\n"
                #    "Rewrite the search query based on the Original Request and the new User Feedback.\n"
                #    "RULES:\n"
                #    "1. Focus on atmosphere, genre, and specific keywords.\n"
                #    "2. Remove conversational fillers.\n"
                #    "3. Output ONLY the new query string."
                #)

                #user_prompt = (
                #    f"Original Request: {seed}\n"
                #    f"User Feedback: {user_response}\n\n"
                #    f"Optimized Query:"
                #)
                
                if hasattr(self.llm_adapter, 'generate'):
                     refined_query = self.llm_adapter.generate(system_prompt, user_prompt)
                else:
                     refined_query = self.llm_adapter.chat_completion(system_prompt, user_prompt)

                refined_query = refined_query.replace('"', '').replace("Optimized Query:", "").strip()
                
                print_fn(f"[DEBUG] Iteration {i+1} - Rewritten Query: '{refined_query}'")
                
                final_query = refined_query
                history.append(f"User Answer: {user_response}")
                
                # D. ITERATIVE SEARCH
                current_context_formatted = self.run_once(final_query, item_types, use_alternative_collection)

        if self.logger is not None:
            self.logger.log(
                user_seed=seed,
                turns=elicited_turns,
                final_query=final_query,
                item_types=item_types,
            )

        result_obj = ElicitationResult(
            final_query=final_query,
            turns=elicited_turns
        )

        return current_context_formatted, result_obj
    
    def get_last_search_results(self):
        return self.last_search_results.copy()
    
    def get_primary_query_embedder(self) -> QueryEmbedder:
        return self.embedder_primary
    
    def get_alternative_query_embedder(self) -> QueryEmbedder:
        return self.embedder_alternative