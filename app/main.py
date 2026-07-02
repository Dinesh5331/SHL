import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChatMessage, ChatRequest, ChatResponse
from app.catalog import load_catalog, get_catalog
from app.retrieval import HybridRetriever
from app.llm_client import LLMClient
from app.router import route_and_extract, extract_previous_shortlist
from app.generator import generate_response

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

retriever: HybridRetriever | None = None
llm_client: LLMClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client
    logger.info("═══ Starting SHL Assessment Recommender Agent ═══")
    raw_key = os.getenv("GROQ_API_KEY", "")
    masked = (raw_key[:12] + "..." + raw_key[-4:]) if len(raw_key) > 16 else "(not set)"
    logger.info("Active GROQ_API_KEY: %s  (model: %s)", masked, os.getenv("LLM_MODEL", "default"))

    catalog = load_catalog()
    logger.info("Catalog: %d items loaded.", len(catalog))

    retriever = HybridRetriever(catalog)  # fast now — loads cached embeddings
    llm_client = LLMClient()
    logger.info("═══ Agent ready ═══")
    yield
    logger.info("═══ Shutting down ═══")


app = FastAPI(
    title="SHL Assessment Recommender Agent",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _handle_chat(messages: list[ChatMessage]) -> ChatResponse:
    previous_shortlist = extract_previous_shortlist(messages)
    logger.info(
        "Previous shortlist: %d items  %s",
        len(previous_shortlist),
        [r["name"] for r in previous_shortlist] if previous_shortlist else "[]",
    )

    action, slots, reasoning, accepts_shortlist = await asyncio.to_thread(
        route_and_extract,
        messages=messages,
        llm_client=llm_client,
        previous_shortlist=previous_shortlist,
    )

    response = await asyncio.to_thread(
        generate_response,
        messages=messages,
        action=action,
        slots=slots,
        retriever=retriever,
        llm_client=llm_client,
        previous_shortlist=previous_shortlist,
        accepts_shortlist=accepts_shortlist,
    )

    rec_count = len(response.recommendations) if response.recommendations else 0
    logger.info(
        "Response: action=%s  recs=%d  eoc=%s",
        action.value, rec_count, response.end_of_conversation,
    )
    return response


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = request.messages
    if not messages:
        return ChatResponse(
            reply=(
                "Hello! I'm the SHL assessment recommendation agent. "
                "Tell me about the role you're hiring for and I'll help "
                "you find the right assessments."
            ),
            recommendations=None,
            end_of_conversation=False,
        )

    try:
        # 25s budget — leaves ~5s under the evaluator's 30s/call cap for
        # network + serialization overhead, and guarantees a schema-valid
        # response even in the worst case instead of a hard connection drop.
        return await asyncio.wait_for(_handle_chat(messages), timeout=25.0)
    except asyncio.TimeoutError:
        logger.error("Chat handling exceeded 25s budget — returning fallback")
        return ChatResponse(
            reply="That's taking longer than expected — could you rephrase or simplify your request?",
            recommendations=None,
            end_of_conversation=False,
        )