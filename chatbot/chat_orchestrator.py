
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

    def run_once(self, user_input: str) -> str:
        """Process a single user input and return relevant items from the vector store."""
        # Embed the user query
        embedding = self.embedder.embed(user_input)
        results = self.vector_store.similarity_search(embedding, k=self.max_items)
        # Format the results for user readability
        formatted_output = format_results(user_input, results, self.llm_adapter)
        return formatted_output
        