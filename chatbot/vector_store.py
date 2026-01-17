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


def _find_config_ini(start_dir: Path) -> Path:
    """
    Find config.ini by walking up parent directories from start_dir.
    This allows running modules from different working directories (repo root vs chatbot/).
    """
    for candidate_dir in (start_dir, *start_dir.parents):
        candidate = candidate_dir / "config.ini"
        if candidate.exists():
            return candidate
    return start_dir / "config.ini"  # fallback for error message


CONFIG_PATH = _find_config_ini(BASE_DIR)

config = configparser.ConfigParser()
read_ok = config.read(CONFIG_PATH)
if not read_ok:
    raise FileNotFoundError(f"Could not read config.ini at: {CONFIG_PATH}")

CONFIG_DIR = CONFIG_PATH.parent

DB_ROOT = (CONFIG_DIR / config["BESTANDEN"]["database_mapnaam"]).resolve()
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
    embedding: list[float] | None = None  # The actual embedding vector of the found item


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
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        embeds = res.get("embeddings") or [[]]

        ids = ids[0] if ids else []
        docs = docs[0] if docs else []
        metas = metas[0] if metas else []
        dists = dists[0] if dists else []
        embeds = embeds[0] if embeds else []

        results: list[SearchResult] = []
        for i in range(len(ids)):
            dist = float(dists[i]) if i < len(dists) else 0.0
            metadata = dict(metas[i]) if i < len(metas) and metas[i] is not None else None
            embedding = list(embeds[i]) if i < len(embeds) and embeds[i] is not None else None
            results.append(
                SearchResult(
                    id=str(ids[i]),
                    score=dist,            # Note: Chroma returns distance (lower = closer) depending on metric
                    document=docs[i] if i < len(docs) else None,
                    metadata=metadata,
                    embedding=embedding,
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
        """
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        embeds = res.get("embeddings") or [[]]

        ids = ids[0] if ids else []
        docs = docs[0] if docs else []
        metas = metas[0] if metas else []
        dists = dists[0] if dists else []
        embeds = embeds[0] if embeds else []

        results: list[SearchResult] = []
        for i in range(len(ids)):
            dist = float(dists[i]) if i < len(dists) else 0.0
            metadata = dict(metas[i]) if i < len(metas) and metas[i] is not None else None
            embedding = list(embeds[i]) if i < len(embeds) and embeds[i] is not None else None
            results.append(
                SearchResult(
                    id=str(ids[i]),
                    score=dist,
                    document=docs[i] if i < len(docs) else None,
                    metadata=metadata,
                    embedding=embedding,
                )
            )
        return results