import json
import logging
from app.schemas import (
    ChatMessage, ChatResponse, Recommendation,
    ExtractedSlots, RouterAction,
)
from app.llm_client import LLMClient
from app.retrieval import HybridRetriever
from app.catalog import find_item_by_name, get_catalog_by_url
from app.prompts import (
    SYSTEM_PROMPT, CLARIFY_PROMPT, RECOMMEND_PROMPT,
    REFINE_PROMPT, COMPARE_PROMPT, REFUSE_PROMPT,
)
from app.router import format_conversation

logger = logging.getLogger(__name__)

MAX_TOTAL_MESSAGES = 8


def _fmt_retrieved(items: list[dict], max_items: int = 15) -> str:
    lines: list[str] = []
    for i, item in enumerate(items[:max_items], 1):
        langs = item.get("languages", [])
        lang_str = ", ".join(langs[:4])
        if len(langs) > 4:
            lang_str += f" (+{len(langs) - 4} more)"

        lines.append(
            f"[{i}] Name: {item['name']}\n"
            f"    URL: {item.get('link', item.get('url', ''))}\n"
            f"    Keys (use these for test_type): {', '.join(item.get('keys', []))}\n"
            f"    Description: {(item.get('description') or 'N/A')[:200]}\n"
            f"    Job Levels: {', '.join(item.get('job_levels', [])) or 'N/A'}\n"
            f"    Duration: {item.get('duration') or 'N/A'}\n"
            f"    Languages: {lang_str or 'N/A'}"
        )
    return "\n\n".join(lines) if lines else "No matching items found."


def _fmt_shortlist(shortlist: list[dict] | None) -> str:
    if not shortlist:
        return "None"
    return "\n".join(
        f"{i}. {r['name']}  |  type: {r['test_type']}  |  {r['url']}"
        for i, r in enumerate(shortlist, 1)
    )


def _validate_recommendations(raw_recs: list[dict]) -> list[Recommendation]:
    catalog_by_url = get_catalog_by_url()
    validated: list[Recommendation] = []
    seen_urls: set[str] = set()

    for rec in raw_recs:
        name = rec.get("name", "").strip()
        url = rec.get("url", "").strip()

        if url in seen_urls:
            continue

        if url in catalog_by_url:
            item = catalog_by_url[url]
            validated.append(Recommendation(
                name=item.name,
                url=item.link,
                test_type=item.test_type,
            ))
            seen_urls.add(url)
            continue

        item = find_item_by_name(name)
        if item and item.link not in seen_urls:
            validated.append(Recommendation(
                name=item.name,
                url=item.link,
                test_type=item.test_type,
            ))
            seen_urls.add(item.link)
            continue

        logger.warning(
            "GUARDRAIL DROP — hallucinated item: name=%r  url=%r", name, url
        )

    return validated


def _is_turn_cap_reached(messages: list[ChatMessage]) -> bool:
    return len(messages) >= MAX_TOTAL_MESSAGES - 1


def _handle_refuse(conv: str, llm: LLMClient) -> ChatResponse:
    result = llm.call_safe_json(SYSTEM_PROMPT, REFUSE_PROMPT.format(conversation=conv))
    return ChatResponse(
        reply=result.get(
            "reply",
            "I can only help with SHL assessment selection. "
            "What role or function are you hiring for?",
        ),
        recommendations=None,
        end_of_conversation=False,
    )


def _handle_clarify(
    conv: str,
    slots: ExtractedSlots,
    llm: LLMClient,
    messages: list[ChatMessage],
) -> ChatResponse:
    turn_count = sum(1 for m in messages if m.role == "user")
    prompt = CLARIFY_PROMPT.format(
        conversation=conv,
        slots=json.dumps(slots.model_dump(exclude_none=True), indent=2),
        turn_count=turn_count,
    )
    result = llm.call_safe_json(SYSTEM_PROMPT, prompt)
    return ChatResponse(
        reply=result.get(
            "reply",
            "Could you tell me more about the role and what you'd like to assess?",
        ),
        recommendations=None,
        end_of_conversation=False,
    )


def _handle_recommend(
    conv: str,
    slots: ExtractedSlots,
    retriever: HybridRetriever,
    llm: LLMClient,
    latest: str,
    accepts_shortlist: bool,
) -> ChatResponse:
    retrieved = retriever.search(latest, slots, top_k=15)
    prompt = RECOMMEND_PROMPT.format(
        conversation=conv,
        role=slots.role or "Not specified",
        seniority=slots.seniority or "Not specified",
        domain=slots.domain or "Not specified",
        skills=slots.skills or "Not specified",
        language_pref=slots.language_pref or "Not specified",
        assessment_types=slots.assessment_types or "Not specified",
        constraints=slots.constraints or "None",
        purpose=slots.purpose or "Not specified",
        retrieved_items=_fmt_retrieved(retrieved),
    )
    result = llm.call_safe_json(SYSTEM_PROMPT, prompt, max_tokens=3000)
    recs = _validate_recommendations(result.get("recommendations", []))

    return ChatResponse(
        reply=result.get("reply", "Here are my recommended assessments:"),
        recommendations=recs,
        end_of_conversation=accepts_shortlist,
    )


def _handle_refine(
    conv: str,
    slots: ExtractedSlots,
    retriever: HybridRetriever,
    llm: LLMClient,
    latest: str,
    prev: list[dict],
    accepts_shortlist: bool,
) -> ChatResponse:
    retrieved = retriever.search(latest, slots, top_k=15)
    prompt = REFINE_PROMPT.format(
        conversation=conv,
        previous_shortlist=_fmt_shortlist(prev),
        refinement_request=latest,
        retrieved_items=_fmt_retrieved(retrieved),
    )
    result = llm.call_safe_json(SYSTEM_PROMPT, prompt, max_tokens=3000)
    recs = _validate_recommendations(result.get("recommendations", []))

    return ChatResponse(
        reply=result.get("reply", "Here's the updated shortlist:"),
        recommendations=recs,
        end_of_conversation=accepts_shortlist,
    )


def _handle_compare(
    conv: str,
    retriever: HybridRetriever,
    llm: LLMClient,
    latest: str,
    prev: list[dict],
) -> ChatResponse:
    retrieved = retriever.search(latest, top_k=10)
    prompt = COMPARE_PROMPT.format(
        conversation=conv,
        assessment_data=_fmt_retrieved(retrieved),
        previous_shortlist=_fmt_shortlist(prev),
    )
    result = llm.call_safe_json(SYSTEM_PROMPT, prompt)
    return ChatResponse(
        reply=result.get("reply", "Here's a comparison of those assessments:"),
        recommendations=None,
        end_of_conversation=False,
    )


def generate_response(
    messages: list[ChatMessage],
    action: RouterAction,
    slots: ExtractedSlots,
    retriever: HybridRetriever,
    llm_client: LLMClient,
    previous_shortlist: list[dict] | None = None,
    accepts_shortlist: bool = False,
) -> ChatResponse:
    conv = format_conversation(messages)
    latest = messages[-1].content if messages else ""

    if _is_turn_cap_reached(messages) and action not in (RouterAction.RECOMMEND, RouterAction.REFINE):
        logger.info("Turn cap reached — forcing RECOMMEND")
        action = RouterAction.RECOMMEND

    try:
        if action == RouterAction.REFUSE:
            return _handle_refuse(conv, llm_client)

        if action == RouterAction.CLARIFY:
            return _handle_clarify(conv, slots, llm_client, messages)

        if action == RouterAction.RECOMMEND:
            return _handle_recommend(
                conv, slots, retriever, llm_client, latest, accepts_shortlist,
            )

        if action == RouterAction.REFINE:
            return _handle_refine(
                conv, slots, retriever, llm_client, latest,
                previous_shortlist or [], accepts_shortlist,
            )

        if action == RouterAction.COMPARE:
            return _handle_compare(
                conv, retriever, llm_client, latest,
                previous_shortlist or [],
            )

    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)

    return ChatResponse(
        reply="I encountered an issue. Could you rephrase or tell me more about the role?",
        recommendations=None,
        end_of_conversation=False,
    )