"""
SHL Assessment Recommender Agent
=================================
Conversational agent for recommending SHL Individual Test Solutions.

Architecture
------------
- POST /chat   : stateless multi-turn conversation endpoint
- GET  /health : readiness probe for Render / Docker

Tech stack
----------
- FastAPI + Uvicorn (async)
- Groq (llama-3.3-70b-versatile) via groq-python SDK
- LanceDB + BAAI/bge-small-en-v1.5 (fastembed) — semantic search
- rank-bm25 — keyword search
- Hybrid RRF merge for retrieval

Modules
-------
catalog   - load, normalise and index the SHL catalog JSON
retrieval - HybridRetriever (BM25 + LanceDB + RRF) with embedding cache
router    - slot extraction + action classification (single LLM call)
generator - response generation + catalog guardrail (zero hallucination)
prompts   - all LLM prompt templates
llm_client - Groq API wrapper (JSON mode, retries, timeout)
schemas   - Pydantic request/response models
"""

__version__ = "1.0.0"
__author__ = "SHL Labs Assignment"
