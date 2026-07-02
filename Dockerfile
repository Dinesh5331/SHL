FROM python:3.11-slim

WORKDIR /app

# ── 1. System deps (minimal) ─────────────────────────────────────────────────
# libgomp: required by LanceDB's native components
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python deps ───────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Pre-download BGE embedding model (build-time, not runtime) ───────────
# Baked into the image so Render's cold start doesn't pay model-download cost.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# ── 4. Copy application code (.dockerignore excludes secrets + cache) ────────
COPY . .

# ── 5. Pre-compute catalog embeddings (build-time fast path) ─────────────────
# Produces data/embeddings_cache.npy + data/embeddings_cache.hash.
# Runtime cold start loads the .npy (~0.5s) instead of recomputing (~40s).
RUN python scripts/prebuild_cache.py

# ── 6. Expose & run ──────────────────────────────────────────────────────────
EXPOSE 8000

# Render injects PORT at runtime; fall back to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
