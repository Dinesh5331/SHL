import os
import logging
import hashlib
import numpy as np
import lancedb
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from app.schemas import CatalogItem, ExtractedSlots

logger = logging.getLogger(__name__)

_embedding_model: TextEmbedding | None = None
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading BAAI/bge-small-en-v1.5 …")
        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        logger.info("Embedding model ready.")
    return _embedding_model


class HybridRetriever:
    def __init__(self, catalog: list[CatalogItem]):
        self.catalog = catalog
        self.searchable_texts = [item.searchable_text for item in catalog]

        tokenized = [text.lower().split() for text in self.searchable_texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built (%d docs).", len(catalog))

        self.embed_model = _get_embedding_model()
        embeddings = self._load_or_build_embeddings(catalog)

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "lancedb",
        )
        self.db = lancedb.connect(db_path)

        rows: list[dict] = []
        for i, item in enumerate(catalog):
            rows.append({
                "id": i,
                "name": item.name,
                "link": item.link,
                "test_type": item.test_type,
                "description": item.description,
                "keys": ", ".join(item.keys),
                "job_levels": ", ".join(item.job_levels),
                "languages": ", ".join(item.languages[:8]),
                "duration": item.duration or "",
                "text": self.searchable_texts[i],
                "vector": embeddings[i].tolist(),
            })

        self.table = self.db.create_table("catalog", rows, mode="overwrite")
        logger.info("LanceDB table created (%d rows).", len(rows))

    def _load_or_build_embeddings(self, catalog: list[CatalogItem]) -> np.ndarray:
        """Cold-start fast path: load pre-computed embeddings baked into the
        Docker image (see scripts/prebuild_cache.py). Falls back to computing
        them live only if the cache is missing or stale (e.g. local dev
        without having run the prebuild script)."""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        emb_path = os.path.join(data_dir, "embeddings_cache.npy")
        hash_path = os.path.join(data_dir, "embeddings_cache.hash")
        catalog_path = os.path.join(data_dir, "catalog.json")

        current_hash = None
        if os.path.exists(catalog_path):
            with open(catalog_path, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()

        if os.path.exists(emb_path) and os.path.exists(hash_path) and current_hash:
            with open(hash_path) as f:
                cached_hash = f.read().strip()
            if cached_hash == current_hash:
                arr = np.load(emb_path)
                if len(arr) == len(catalog):
                    logger.info("Loaded cached embeddings — cold start fast path (%d items).", len(arr))
                    return arr
                logger.warning("Cached embedding count (%d) != catalog size (%d) — rebuilding.", len(arr), len(catalog))
            else:
                logger.warning("Catalog changed since cache was built — rebuilding embeddings.")
        else:
            logger.warning("No embedding cache found — computing live (run scripts/prebuild_cache.py to fix this).")

        return np.array(list(self.embed_model.embed(self.searchable_texts)))

    

    @staticmethod
    def build_refined_query(slots: ExtractedSlots, latest_message: str) -> str:
        parts: list[str] = []

        if slots.role:
            parts.append(f"assessment for {slots.role}")
        if slots.seniority:
            parts.append(f"{slots.seniority} level")
        if slots.skills:
            parts.append(slots.skills)
        if slots.domain:
            parts.append(f"{slots.domain}")
        if slots.assessment_types:
            parts.append(slots.assessment_types)
        if slots.purpose:
            parts.append(f"for {slots.purpose}")
        if slots.language_pref:
            parts.append(f"in {slots.language_pref}")
        if slots.constraints:
            parts.append(slots.constraints)

        if latest_message:
            parts.append(latest_message)

        return " ".join(parts) if parts else latest_message

    @staticmethod
    def _broaden_query(slots: ExtractedSlots) -> str:
        parts: list[str] = []
        if slots.role:
            parts.append(slots.role)
        if slots.domain:
            parts.append(slots.domain)
        if slots.assessment_types:
            parts.append(slots.assessment_types)
        if slots.seniority:
            parts.append(slots.seniority)
        return " ".join(parts) if parts else "general assessment"

    def search(
        self,
        query: str,
        slots: ExtractedSlots | None = None,
        top_k: int = 15,
        *,
        _is_fallback: bool = False,
    ) -> list[dict]:
        if slots and not _is_fallback:
            refined = self.build_refined_query(slots, query)
        else:
            refined = query

        logger.info("Retrieval query: %s", refined[:200])

        tok_query = refined.lower().split()
        bm25_scores = self.bm25.get_scores(tok_query)
        bm25_order = np.argsort(bm25_scores)[::-1]

        q_emb = list(self.embed_model.embed([BGE_QUERY_PREFIX + refined]))[0]
        sem_results = self.table.search(q_emb.tolist()).limit(top_k * 3).to_list()
        sem_ids = [r["id"] for r in sem_results]

        K = 60
        rrf: dict[int, float] = {}

        for rank, idx in enumerate(bm25_order[: top_k * 3]):
            idx = int(idx)
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)

        for rank, idx in enumerate(sem_ids):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (K + rank + 1)

        sorted_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)

        top_score = rrf.get(sorted_ids[0], 0.0) if sorted_ids else 0.0
        if top_score < 0.015 and slots and not _is_fallback:
            broad = self._broaden_query(slots)
            logger.info("Weak results (%.4f) — broadening → %s", top_score, broad)
            return self.search(broad, slots=None, top_k=top_k, _is_fallback=True)

        results: list[dict] = []
        for idx in sorted_ids[:top_k]:
            item = self.catalog[idx]
            results.append({
                "catalog_item": item,
                "score": rrf[idx],
                "name": item.name,
                "link": item.link,
                "test_type": item.test_type,
                "description": item.description,
                "keys": item.keys,
                "job_levels": item.job_levels,
                "languages": item.languages,
                "duration": item.duration,
            })

        return results

    def search_by_names(self, names: list[str]) -> list[dict]:
        results: list[dict] = []
        seen: set[str] = set()
        for name in names:
            name_lower = name.lower().strip()
            for item in self.catalog:
                if item.name.lower() in seen:
                    continue
                if (
                    name_lower in item.name.lower()
                    or item.name.lower() in name_lower
                ):
                    results.append({
                        "catalog_item": item,
                        "name": item.name,
                        "link": item.link,
                        "test_type": item.test_type,
                        "description": item.description,
                        "keys": item.keys,
                        "job_levels": item.job_levels,
                        "languages": item.languages,
                        "duration": item.duration,
                    })
                    seen.add(item.name.lower())
                    break
        return results