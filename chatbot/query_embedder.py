from chromadb.utils import embedding_functions
import llm_adapter
from llm_adapter import LLMAdapter

SYSTEM_PROMPT = """ You rewrite user search queries to be optimal for semantic vector search. 
                    Preserve the original intent.
                    Do not add new concepts, opinions, or recommendations.
                    Return only the rewritten query text in English; no introduction text etc."""

SYSTEM_PROMPT = \
"""You rewrite user search queries to be optimal for semantic vector search over plot synopses.
Preserve the user's intent and emotional tone exactly and do not introduce new entities, settings, themes, or genres.
If the user expresses tone constraints (e.g. dark, serious, not light), reflect them explicitly.
Convert the query into a compact "synopsis-like" description using the user's own concepts only.
Use English, third person, present tense.
Return only the rewritten text."""

class QueryEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", llm_adapter: LLMAdapter | None = None) -> None:
        """
        Initialize the QueryEmbedder.

        Args:
            model_name (str, optional): Name of the sentence transformer model to use for embeddings.
                Defaults to "paraphrase-multilingual-MiniLM-L12-v2".
            llm_adapter (llm_adapter.LLMAdapter | None, optional): An optional LLM adapter instance.
                If not provided, defaults to None.

        Returns:
            None
        """
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self._llm_adapter = llm_adapter

    def embed(self, query: str) -> list[float]:
        """Embed a single query string into a vector. Use the LLM adapter if provided during initialization to improve the query first."""
        if self._llm_adapter:
            query = self.improve_query_with_llm(query)
        
        return self._ef([query])[0].tolist()

    def improve_query_with_llm(self, query):
        system_prompt = SYSTEM_PROMPT
        try:
            if not self._llm_adapter:
                raise ValueError("LLM adapter is not provided.")
            improved_query = self._llm_adapter.generate(system_prompt, query)
            query = improved_query
        except Exception as e:
            print(f"LLM adapter failed to improve query: {e}")
        return query

# Smoke test
if __name__ == "__main__":
    test_query = "I want to see something that contains pirates and treasure. and romance too."

    # Improve the query with LLM and print it
    llm_adapter_instance = llm_adapter.create_llm_adapter(provider="openai")
    embedder = QueryEmbedder(llm_adapter=llm_adapter_instance)
    improved_query = embedder.improve_query_with_llm(test_query)
    print(f"Test query: {test_query}")
    print(f"Improved query: {improved_query}")

    # Get the embedding for the improved query with LLM
    embedding = embedder.embed(test_query)
    print(f"Embedding: {embedding[:5]}... (length: {len(embedding)})")