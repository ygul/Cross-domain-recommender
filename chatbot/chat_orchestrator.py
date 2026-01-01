from __future__ import annotations

from typing import Literal, Optional, Tuple

from query_embedder import QueryEmbedder
from vector_store import VectorStore
import llm_adapter
from results_formatter import format_results

from clarify_gate import ClarifyGate, ElicitationResult
from elicitation_logger import ElicitationLogger


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

        if self.logger is not None:
            self.logger.log(
                user_seed=seed,
                turns=[] if elicited is None else elicited.turns,
                final_query=final_query,
                item_types=item_types,
            )

        formatted = self.run_once(final_query, item_types=item_types, use_alternative_collection=use_alternative_collection)
        return formatted, elicited
