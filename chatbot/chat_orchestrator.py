
from query_embedder import QueryEmbedder
from vector_store import VectorStore
import llm_adapter
from typing import Literal
from results_formatter import format_results

class ChatOrchestrator:
    def __init__(self, llm_provider: Literal['openai', 'huggingface'] = "openai", max_items: int = 3) -> None:
        self.llm_adapter = llm_adapter.create_llm_adapter(provider=llm_provider)
        self.embedder = QueryEmbedder(llm_adapter=self.llm_adapter)
        self.vector_store = VectorStore()
        self.max_items = max_items

    def run_once(
        self, 
        user_input: str,
        item_types: set[Literal['Book', 'TV series', 'movie']] | None = None,
    ) -> str:
        """
        Process a single user input and return relevant items from the vector store.
        
        Args:
            user_input: The user's query
            item_types: Optional set of item types to filter by ('book', 'series', 'movie').
                       If None, no filtering is applied.
                       If multiple types are provided, they are combined with OR logic.
        
        Returns:
            Formatted results as a string
        """
        # Embed the user query
        embedding = self.embedder.embed(user_input)
        
        # Determine whether to use filtered or unfiltered search
        if item_types is None:
            # No filtering
            results = self.vector_store.similarity_search(embedding, k=self.max_items)
        else:
            # Build OR filter for multiple item types
            where_filter = {
                "$or": [{"item_type": item_type} for item_type in item_types]
            } if len(item_types) > 1 else {"item_type": list(item_types)[0]}
            
            results = self.vector_store.filtered_similarity_search(
                embedding, 
                k=self.max_items,
                where=where_filter
            )
        
        # Format the results for user readability
        formatted_output = format_results(user_input, results, self.llm_adapter)
        return formatted_output
        