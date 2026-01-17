from __future__ import annotations

from typing import Literal, Optional, Tuple

from query_embedder import QueryEmbedder
from vector_store import VectorStore
import llm_adapter
from results_formatter import format_results

from clarify_gate import ClarifyGate, ElicitationResult


ItemType = Literal["Book", "TV series", "movie"]


# Usage note:
# - Per call, pass use_alternative_collection=True to chat()/run_once() to query the
#   alternative Chroma collection (e.g., model2); omit or False for the primary.
# - You can flip this flag each call to compare collections without code changes.

class ChatOrchestrator:
    def __init__(
        self,
        llm_provider: Literal["openai", "huggingface"] = "openai",
        max_items: int = 3,
        enable_clarify_gate: bool = True,
        clarify_max_questions: int = 2,
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
        
        # Store last search results for external access (e.g., by judge module)
        self.last_search_results = []

    def run_once(
        self,
        user_input: str,
        item_types: set[ItemType] | None = None,
        use_alternative_collection: bool | None = None,
    ) -> str:
        """
        Retrieve + format results for one query (no elicitation here).
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

        # Store results for external access
        self.last_search_results = results

        return format_results(user_input, results, self.llm_adapter)

    def chat(
        self,
        seed: str,
        item_types: set[ItemType] | None = None,
        input_fn=input,
        print_fn=print,
        use_alternative_collection: bool | None = None,
    ) -> Tuple[str, Optional[ElicitationResult]]:
        """
        Full chat flow (design-aligned):
        seed -> optional ClarifyGate (0/1/2 questions) -> log -> retrieval+format

        Args:
            seed: User input text
            item_types: Optional item-type filter
            input_fn, print_fn: I/O hooks for CLI
            use_alternative_collection: Per-call override; True queries the alternative collection/model,
                False the primary; None uses the primary collection/model by default.

        Returns:
          (formatted_output, elicitation_result_or_None)
        """
        seed = (seed or "").strip()
        if seed == "":
            return "Empty input. Please provide a description of what you're looking for.", None

        elicited: Optional[ElicitationResult] = None
        final_query = seed

        if self.clarify_gate is not None:
            elicited = self.clarify_gate.run(seed, input_fn=input_fn, print_fn=print_fn)
            final_query = elicited.final_query

        formatted = self.run_once(final_query, item_types=item_types, use_alternative_collection=use_alternative_collection)
        # Note: last_search_results is already set by run_once()
        return formatted, elicited
    
    def get_last_search_results(self):
        """
        Get the vector search results from the last run_once() or chat() call.
        
        Returns:
            list: List of search results with metadata, distances, and documents.
                  Empty list if no searches have been performed yet.
                  
        Example:
            >>> orchestrator = ChatOrchestrator()
            >>> response = orchestrator.run_once("science fiction books")
            >>> results = orchestrator.get_last_search_results()
            >>> print(f"Found {len(results)} results")
            >>> for result in results:
            ...     print(f"Distance: {result['distance']}, Title: {result['metadata']['title']}")
        """
        return self.last_search_results.copy()  # Return copy to prevent external modification
    
    def get_primary_query_embedder(self) -> QueryEmbedder:
        """
        Get the primary embedding model instance.
        
        Returns the QueryEmbedder configured with the primary collection's embedding model
        (typically paraphrase-multilingual-MiniLM-L12-v2). This embedder is used for all
        queries when use_alternative_collection=False or is not specified.
        
        Returns:
            QueryEmbedder: The primary embedder instance with the model used by the main collection.
            
        Example:
            >>> orchestrator = ChatOrchestrator()
            >>> embedder = orchestrator.get_primary_embedder()
            >>> vector = embedder.embed("science fiction books")
            >>> print(f"Embedding dimension: {len(vector)}")
        """
        return self.embedder_primary
    
    def get_alternative_query_embedder(self) -> QueryEmbedder:
        """
        Get the alternative embedding model instance.
        
        Returns the QueryEmbedder configured with the alternative collection's embedding model
        (typically sentence-transformers/all-MiniLM-L6-v2). This embedder is used for all
        queries when use_alternative_collection=True is specified.
        
        Returns:
            QueryEmbedder: The alternative embedder instance with the model used by the alternative collection.
            
        Example:
            >>> orchestrator = ChatOrchestrator()
            >>> embedder = orchestrator.get_alternative_embedder()
            >>> vector = embedder.embed("science fiction books")
            >>> print(f"Embedding dimension: {len(vector)}")
        """
        return self.embedder_alternative
