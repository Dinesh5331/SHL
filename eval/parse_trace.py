import re
from pathlib import Path

_TURN_RE = re.compile(r"### Turn \d+\s*\n(.*?)(?=### Turn \d+|\Z)", re.DOTALL)
_USER_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\*\*Agent\*\*|\Z)", re.DOTALL)
_AGENT_RE = re.compile(r"\*\*Agent\*\*\s*\n(.*?)(?=\n_`end_of_conversation`|\Z)", re.DOTALL)
_EOC_RE = re.compile(r"_`end_of_conversation`:\s*\*\*(true|false)\*\*_")
_TABLE_ROW_RE = re.compile(
    r"\|\s*\d+\s*\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<test_type>[^|]+?)\s*\|"
    r".*?\|.*?\|.*?\|\s*<?(?P<url>https?://\S+?)>?\s*\|"
)


def parse_trace_file(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    turns = _TURN_RE.findall(text)

    history: list[dict] = []
    last_shortlist: list[str] = []

    for turn_text in turns:
        user_match = _USER_RE.search(turn_text)
        agent_match = _AGENT_RE.search(turn_text)

        if user_match:
            history.append({
                "role": "user",
                "content": user_match.group(1).strip().replace("\n> ", "\n"),
            })

        if agent_match:
            agent_text = agent_match.group(1).strip()
            history.append({"role": "assistant", "content": agent_text})

            rows = _TABLE_ROW_RE.findall(agent_text)
            if rows:
                last_shortlist = [name.strip() for name, _tt, _url in rows]

    return {
        "id": Path(path).stem,
        "history": history[:-1] if history and history[-1]["role"] == "assistant" else history,
        "expected": last_shortlist,
    }


def load_all_traces(traces_dir: str) -> list[dict]:
    traces = []
    for path in sorted(Path(traces_dir).glob("*.md")):
        trace = parse_trace_file(str(path))
        if trace["history"] and trace["expected"]:
            traces.append(trace)
    return traces