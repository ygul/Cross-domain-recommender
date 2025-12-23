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
    def __init__(self, db_root: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(db_root))
        self._collection = self._client.get_collection(name=collection_name)

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


if __name__ == "__main__":
    from query_embedder import QueryEmbedder

    # Smoke test: does the collection load?
    print(f"Chroma DB root: {DB_ROOT}")
    print(f"Collection: {COLLECTION_NAME}")
    vs = VectorStore(DB_ROOT, COLLECTION_NAME)
    count = vs._collection.count()
    print(f"Loaded collection with {count} items.")

    # Create an embedding, run a similarity search and print results
    embedder = QueryEmbedder()
    query = "I want to see stuff about a space bounty hunter going on adventures. I would really like that."
    query = "I want something emotional about family, loss, and difficult choices, preferably not too light."
    print(f"Original query: {query}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")

    query_embedding = embedder.embed(query)
    results = vs.similarity_search(query_embedding, k=3)
    for r in results:
        print(f"- ID: {r.id}, Score: {r.score}, Metadata: {r.metadata}")
        print()

    # Now do the same but with LLM-improved query
    import llm_adapter
    llm_embedder = QueryEmbedder(llm_adapter=llm_adapter.create_llm_adapter(provider="openai"))
    improved_query = llm_embedder.improve_query_with_llm(query)
    improved_query_embedding = llm_embedder.embed(query)
    improved_results = vs.similarity_search(improved_query_embedding, k=3)
    print(f"Results for improved query: {improved_query}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for r in improved_results:
        print(f"- ID: {r.id}, Score: {r.score}, Metadata: {r.metadata}")
        print()