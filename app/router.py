import re
import logging
from app.schemas import ChatMessage, ExtractedSlots, RouterAction
from app.llm_client import LLMClient
from app.prompts import SYSTEM_PROMPT, SLOT_EXTRACTION_AND_ROUTING_PROMPT
from app.catalog import get_catalog_by_name

logger = logging.getLogger(__name__)

_TABLE_ROW_RE = re.compile(
    r"\|\s*\d+\s*\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<test_type>[^|]+?)\s*\|"
    r".*?\|.*?\|.*?\|\s*(?P<url>https?://\S+?)\s*\|"
)


def format_conversation(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Agent"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def extract_previous_shortlist(messages: list[ChatMessage]) -> list[dict]:
    catalog_by_name = get_catalog_by_name()
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        rows = _TABLE_ROW_RE.findall(msg.content)
        if not rows:
            continue
        shortlist: list[dict] = []
        for name, test_type, url in rows:
            name = name.strip()
            item = catalog_by_name.get(name.lower().strip())
            if item:
                shortlist.append({"name": item.name, "url": item.link, "test_type": item.test_type})
            else:
                shortlist.append({"name": name, "url": url.strip(), "test_type": test_type.strip()})
        if shortlist:
            return shortlist
    return []


_INJECTION_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+your\s+(prompt|rules)",
        r"you\s+are\s+now\s+a",
        r"forget\s+(everything|your\s+(rules|instructions))",
        r"what\s+is\s+your\s+system\s+prompt",
        r"repeat\s+your\s+(instructions|prompt)",
        r"(pretend|act)\s+(like\s+)?you\s+are\s+(?!an?\s+shl)",
        r"disregard\s+(all|any)\s+(previous|prior)",
    ]
]

_OFFTOPIC_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(are\s+we|am\s+i)\s+(legally|required\s+by\s+law)",
        r"what\s+does\s+the\s+law\s+say",
        r"legal\s+(requirement|obligation|compliance|liability)",
        r"(write|draft)\s+(a\s+)?(job\s+description|offer\s+letter|contract)",
        r"salary\s+(range|recommendation|suggestion|benchmark)",
        r"how\s+to\s+(fire|terminate|lay\s*off)",
        r"(visa|immigration)\s+(requirement|sponsor)",
    ]
]


def _check_hard_refuse(latest: str) -> bool:
    lower = latest.lower().strip()
    for pat in _INJECTION_RE:
        if pat.search(lower):
            logger.info("Hard rule → REFUSE (injection)")
            return True
    for pat in _OFFTOPIC_RE:
        if pat.search(lower):
            if not re.search(r"\bshl\b|\bassessment\b|\btest\b", lower):
                logger.info("Hard rule → REFUSE (off-topic)")
                return True
    return False


def route_and_extract(
    messages: list[ChatMessage],
    llm_client: LLMClient,
    previous_shortlist: list[dict] | None = None,
) -> tuple[RouterAction, ExtractedSlots, str, bool]:
    latest_message = messages[-1].content if messages else ""

    if _check_hard_refuse(latest_message):
        return RouterAction.REFUSE, ExtractedSlots(), "Hard rule: refuse", False

    conversation_text = format_conversation(messages)

    shortlist_text = "None"
    if previous_shortlist:
        lines = [
            f"{i}. {r['name']}  —  {r['url']}  —  type: {r['test_type']}"
            for i, r in enumerate(previous_shortlist, 1)
        ]
        shortlist_text = "\n".join(lines)

    prompt = SLOT_EXTRACTION_AND_ROUTING_PROMPT.format(
        conversation=conversation_text,
        previous_shortlist=shortlist_text,
    )

    try:
        result = llm_client.call_safe_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
        )

        sd = result.get("slots", {})
        slots = ExtractedSlots(
            role=sd.get("role"),
            seniority=sd.get("seniority"),
            domain=sd.get("domain"),
            skills=sd.get("skills"),
            language_pref=sd.get("language_pref"),
            assessment_types=sd.get("assessment_types"),
            constraints=sd.get("constraints"),
            purpose=sd.get("purpose"),
        )

        action_str = result.get("action", "clarify")
        try:
            action = RouterAction(action_str)
        except ValueError:
            action = RouterAction.CLARIFY

        reasoning = result.get("reasoning", "")

        if action == RouterAction.REFINE and not previous_shortlist:
            action = RouterAction.RECOMMEND
            reasoning += " [switched to recommend: no prior shortlist]"

        if (
            len(messages) == 1
            and action == RouterAction.RECOMMEND
            and not slots.role
            and not slots.domain
            and not slots.skills
            and not slots.assessment_types
        ):
            action = RouterAction.CLARIFY
            reasoning += " [hard override: turn 1, no usable signal at all]"

        accepts_shortlist = bool(result.get("accepts_shortlist", False)) and bool(previous_shortlist)

        logger.info("Router → %s | accepts=%s | %s", action.value, accepts_shortlist, reasoning)
        logger.info("Slots  → %s", slots.model_dump_json())
        return action, slots, reasoning, accepts_shortlist

    except Exception as exc:
        logger.error("Router LLM failed: %s — fallback CLARIFY", exc)
        return RouterAction.CLARIFY, ExtractedSlots(), f"Fallback ({exc})", False