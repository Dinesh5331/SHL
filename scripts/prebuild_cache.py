"""Run once at Docker build time. Downloads the BGE embedding model and
pre-computes + caches catalog embeddings, so runtime cold start only
has to load a small .npy file instead of re-embedding 377 items."""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.catalog import load_catalog
from app.retrieval import _get_embedding_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")
EMB_PATH = os.path.join(DATA_DIR, "embeddings_cache.npy")
HASH_PATH = os.path.join(DATA_DIR, "embeddings_cache.hash")


def catalog_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    print("Loading catalog...")
    catalog = load_catalog(CATALOG_PATH)
    texts = [item.searchable_text for item in catalog]

    print("Loading/downloading BGE embedding model...")
    model = _get_embedding_model()

    print(f"Embedding {len(texts)} catalog items (build-time, not runtime)...")
    embeddings = np.array(list(model.embed(texts)))

    np.save(EMB_PATH, embeddings)
    with open(HASH_PATH, "w") as f:
        f.write(catalog_hash(CATALOG_PATH))

    print(f"Cached embeddings {embeddings.shape} -> {EMB_PATH}")
    print("Cold start will now load this file instead of recomputing.")


if __name__ == "__main__":
    main()