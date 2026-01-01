from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import configparser
from typing import Any
import chromadb



# -------------------------------------------------
# Config & paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"

config = configparser.ConfigParser()
read_ok = config.read(CONFIG_PATH)
if not read_ok:
    raise FileNotFoundError(f"Could not read config.ini at: {CONFIG_PATH}")

DB_ROOT = (BASE_DIR / config["BESTANDEN"]["database_mapnaam"]).resolve()
COLLECTION_NAME = config["BESTANDEN"]["collectie_naam"]
COLLECTION_NAME_MODEL2 = config["BESTANDEN"]["collectie_naam_model2"]

# -------------------------------------------------
# Vector store
# -------------------------------------------------
@dataclass(frozen=True)
class SearchResult:
    id: str
    score: float
    document: str | None
    metadata: dict[str, Any] | None


class VectorStore:
    def __init__(
        self,
        db_root: Path | None = None,
        collection_name: str | None = None,
        use_alternative_collection: bool = False,
    ) -> None:
        if use_alternative_collection and collection_name:
            raise AssertionError("If use_alternative_collection is True, collection_name must be None or empty")

        db_root = db_root or DB_ROOT
        resolved_collection = (
            COLLECTION_NAME_MODEL2 if use_alternative_collection else (collection_name or COLLECTION_NAME)
        )

        self._client = chromadb.PersistentClient(path=str(db_root))
        self._collection = self._client.get_collection(name=resolved_collection)

    def similarity_search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]

        ids = ids[0] if ids else []
        docs = docs[0] if docs else []
        metas = metas[0] if metas else []
        dists = dists[0] if dists else []

        results: list[SearchResult] = []
        for i in range(len(ids)):
            dist = float(dists[i]) if i < len(dists) else 0.0
            metadata = dict(metas[i]) if i < len(metas) and metas[i] is not None else None
            results.append(
                SearchResult(
                    id=str(ids[i]),
                    score=dist,            # Note: Chroma returns distance (lower = closer) depending on metric
                    document=docs[i] if i < len(docs) else None,
                    metadata=metadata,
                )
            )
        return results

    def filtered_similarity_search(
        self, 
        query_embedding: list[float], 
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search with optional metadata filters.
        
        Args:
            query_embedding: The embedding vector to search with
            k: Number of results to return
            where: Chroma where filter dict. Examples:
                   {"item_type": "book"}
                   {"item_type": {"$eq": "book"}}
                   {"year": {"$gte": 2020}}
                   {"$and": [{"item_type": "book"}, {"year": {"$gte": 2020}}]}
        
        Returns:
            List of SearchResult objects
        """
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]

        ids = ids[0] if ids else []
        docs = docs[0] if docs else []
        metas = metas[0] if metas else []
        dists = dists[0] if dists else []

        results: list[SearchResult] = []
        for i in range(len(ids)):
            dist = float(dists[i]) if i < len(dists) else 0.0
            metadata = dict(metas[i]) if i < len(metas) and metas[i] is not None else None
            results.append(
                SearchResult(
                    id=str(ids[i]),
                    score=dist,
                    document=docs[i] if i < len(docs) else None,
                    metadata=metadata,
                )
            )
        return results


if __name__ == "__main__":
    from query_embedder import QueryEmbedder

    # Smoke test: primary collection
    print(f"Chroma DB root: {DB_ROOT}")
    print(f"Collection (primary): {COLLECTION_NAME}")
    vs = VectorStore(DB_ROOT, COLLECTION_NAME)
    count = vs._collection.count()
    print(f"Loaded primary collection with {count} items.")

    # Smoke test: alternative collection
    print(f"Collection (alternative): {COLLECTION_NAME_MODEL2}")
    vs_alt = VectorStore(DB_ROOT, None, use_alternative_collection=True)
    count_alt = vs_alt._collection.count()
    print(f"Loaded alternative collection with {count_alt} items.")

    # Create an embedding, run similarity searches for both collections
    embedder = QueryEmbedder()
    query = "I want to see stuff about a space bounty hunter going on adventures. I would really like that."
    query = "I want something emotional about family, loss, and difficult choices, preferably not too light."
    print(f"Original query: {query}")

    import llm_adapter
    llm_embedder = QueryEmbedder(llm_adapter=llm_adapter.create_llm_adapter(provider="openai"))
    improved_query = llm_embedder.improve_query_with_llm(query)
    print(f"Improved query: {improved_query}")

    def run_search(label: str, store: VectorStore) -> None:
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Searching in collection: {label}")

        # Original query
        query_embedding = embedder.embed(query)
        results = store.similarity_search(query_embedding, k=3)
        print("-- Results for original query --")
        for r in results:
            name = None
            if r.metadata:
                name = r.metadata.get("name") or r.metadata.get("title") or r.metadata.get("Title")
            name = name or "(no name)"
            print(f"- ID: {r.id}, Name: {name}, Score: {r.score}")
        print()

        # Improved query
        improved_query_embedding = llm_embedder.embed(improved_query)
        improved_results = store.similarity_search(improved_query_embedding, k=3)
        print("-- Results for improved query --")
        for r in improved_results:
            name = None
            if r.metadata:
                name = r.metadata.get("name") or r.metadata.get("title") or r.metadata.get("Title")
            name = name or "(no name)"
            print(f"- ID: {r.id}, Name: {name}, Score: {r.score}")
        print()

    run_search(f"primary ({COLLECTION_NAME})", vs)
    run_search(f"alternative ({COLLECTION_NAME_MODEL2})", vs_alt)