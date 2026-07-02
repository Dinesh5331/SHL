# SHL Assessment Recommender Agent

A conversational AI agent for recommending SHL Individual Test Solution assessments.  
Given a job description or hiring context, the agent recommends the right assessments through a multi-turn dialogue, handles clarification, refinement, and comparison — all grounded to the real SHL product catalog.

---

## Quick Start (local)

```bash
# 1. Clone and enter the project
cd shl-recommender

# 2. Create virtualenv
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Server is ready when you see `═══ Agent ready ═══` in the log (~1s with cached embeddings, ~40s first run).

---

## API Reference

### `GET /health`
Readiness probe. Returns `{"status": "ok"}`.

### `POST /chat`
Stateless multi-turn conversation. Send the **full conversation history** on every call.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need a cognitive ability test for a software engineer role."},
    {"role": "assistant", "content": "...previous reply..."},
    {"role": "user", "content": "Add a personality test too."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are my recommended assessments...",
  "recommendations": [
    {
      "name": "SHL Verify Interactive G+",
      "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
      "test_type": "A"
    }
  ],
  "end_of_conversation": false
}
```

**Fields:**
| Field | Type | Description |
|---|---|---|
| `reply` | `string` | Agent's natural language response |
| `recommendations` | `list \| null` | Structured shortlist (null for clarify/compare/refuse turns) |
| `end_of_conversation` | `bool` | True when user explicitly accepts the shortlist |

**test_type codes:**
| Code | Meaning |
|---|---|
| A | Ability & Aptitude |
| P | Personality & Behavior |
| K | Knowledge & Skills |
| C | Competencies |
| B | Biodata & Situational Judgment |
| S | Simulations |
| D | Development & 360 |
| E | Assessment Exercises |

---

## Deploy to Render

### One-click via `render.yaml`

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Blueprint**
3. Connect your GitHub repo → Render reads `render.yaml` automatically
4. In the **Environment** tab for the created service, set:
   ```
   GROQ_API_KEY = gsk_your_actual_key_here
   ```
5. Click **Deploy**

Build takes ~5-8 minutes (pip install + BGE model download + embedding precompute).  
Cold start after deploy: **< 5 seconds** (embeddings cached in image).

### Manual via Render dashboard

1. New → **Web Service** → Connect your GitHub repo
2. **Runtime:** Docker
3. **Health Check Path:** `/health`
4. **Plan:** Starter (512 MB RAM is enough)
5. Add env var `GROQ_API_KEY` in the Environment tab
6. Deploy

---

## Run Evaluation

```bash
# Against local server
python -m eval.run_eval

# Against deployed Render instance
python -m eval.run_eval --base-url https://shl-recommender.onrender.com

# Recall@5 instead of @10
python -m eval.run_eval --k 5
```

### Probes

| Probe | What it checks |
|---|---|
| `short_scenarios` | 4 specific queries → must return ≥ 1 recommendation |
| `long_multiturn` | clarify → recommend → refine × 2 → recs preserved |
| `compare` | compare turn returns text only, no shortlist change |
| `broad_offtopic` | 10 off-topic prompts → 0 recommendations |
| `injection_variants` | 4 injection attempts → 0 recommendations |

---

## Architecture

```
POST /chat  {full message history}
    │
    ├─ extract_previous_shortlist()   ← regex-parse last table in assistant msg
    │
    ├─ route_and_extract()            ← 1 LLM call → slots + action
    │   ├─ Hard refuse: injection / off-topic regex
    │   ├─ Guard: refine without shortlist → recommend
    │   └─ Guard: turn-1 vague → clarify
    │
    ├─ HybridRetriever.search()       ← BM25 + LanceDB RRF, top-15
    │   └─ BGE-small with query prefix
    │
    ├─ LLM response generation        ← 1 LLM call per action
    │
    └─ _validate_recommendations()    ← catalog guardrail, zero hallucination
        └─ all URLs verified against catalog; test_type from catalog only
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Vector DB | LanceDB (local, file-based) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed |
| Keyword Search | rank-bm25 |
| Validation | Pydantic v2 |
| Deploy | Docker → Render |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key (`gsk_...`) |
| `LLM_MODEL` | ❌ | `llama-3.1-8b-instant` | Any Groq chat model ID |
