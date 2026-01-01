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

    import llm_adapter

    # Create embedders per collection based on collection metadata
    def build_embedders(store: VectorStore):
        meta = store._collection.metadata or {}
        model_name = meta.get("embedding_model") or "paraphrase-multilingual-MiniLM-L12-v2"
        base_embedder = QueryEmbedder(model_name=model_name)
        llm_emb = QueryEmbedder(
            model_name=model_name,
            llm_adapter=llm_adapter.create_llm_adapter(provider="openai"),
        )
        return model_name, base_embedder, llm_emb

    primary_model_name, embedder, llm_embedder = build_embedders(vs)
    alt_model_name, embedder_alt, llm_embedder_alt = build_embedders(vs_alt)

    query = "I want to see stuff about a space bounty hunter going on adventures. I would really like that."
    query = "I want something emotional about family, loss, and difficult choices, preferably not too light."
    
    # Short description of The Mandalorian
    query = "After the fall of the Galactic Empire, lawlessness has spread throughout the galaxy. A lone gunfighter makes his way through the outer reaches, earning his keep as a bounty hunter."

    # Long description of The Mandalorian (used for the actual embedding, so distance should be zero for both models/collections)
    query = """In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue. Each mission presents not only a test of skill but also a moral quandary, as the line between right and wrong blurs in a world where survival often comes at a steep price. As he traverses desolate planets and bustling spaceports, the bounty hunter encounters a diverse array of characters—some allies, others adversaries—each with their own agendas and secrets. The weight of his past looms large, shaping his choices and challenging his resolve. In this unforgiving environment, the hunter must grapple with themes of identity, loyalty, and the quest for redemption, all while evading the shadows of his own making. The atmosphere is thick with tension, as the hunter's journey unfolds against a backdrop of stunning vistas and gritty underbellies, where every corner turned could lead to fortune or peril. With each encounter, the stakes rise, and the hunter's resolve is tested, forcing him to confront not only the dangers of the galaxy but also the deeper conflicts within himself. In this relentless pursuit of purpose, the hunter discovers that sometimes, the greatest battles are fought not against others, but within one's own soul."""    

    print(f"Original query: {query}")

    improved_query = llm_embedder.improve_query_with_llm(query)
    print(f"Improved query (primary model {primary_model_name}): {improved_query}")

    def run_search(label: str, store: VectorStore, base_emb: QueryEmbedder, llm_emb: QueryEmbedder, model_name: str) -> None:
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(f"Searching in collection: {label} (embedding model: {model_name})")

        # Original query
        query_embedding = base_emb.embed(query)
        results = store.similarity_search(query_embedding, k=3)
        print("-- Results for original query --")
        for r in results:
            name = None
            if r.metadata:
                name = r.metadata.get("name") or r.metadata.get("title") or r.metadata.get("Title")
            name = name or "(no name)"
            print(f"- ID: {r.id}, Name: {name}, Score: {r.score:.4f}")
        print()

        # Improved query
        improved_query_embedding = llm_emb.embed(improved_query)
        improved_results = store.similarity_search(improved_query_embedding, k=3)
        print("-- Results for improved query --")
        for r in improved_results:
            name = None
            if r.metadata:
                name = r.metadata.get("name") or r.metadata.get("title") or r.metadata.get("Title")
            name = name or "(no name)"
            print(f"- ID: {r.id}, Name: {name}, Score: {r.score:.4f}")
        print()

    run_search(f"primary ({COLLECTION_NAME})", vs, embedder, llm_embedder, primary_model_name)
    run_search(f"alternative ({COLLECTION_NAME_MODEL2})", vs_alt, embedder_alt, llm_embedder_alt, alt_model_name)