"""Load, normalize, and index the SHL product catalog."""

import json
import os
import logging
from pathlib import Path
from app.schemas import CatalogItem

logger = logging.getLogger(__name__)

# ── Key → single-letter code mapping ────────────────────────────────────────

KEY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def _build_test_type(keys: list[str]) -> str:
    """Derive comma-separated test-type codes from catalog keys."""
    codes: list[str] = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else ""  


def _build_searchable_text(raw: dict) -> str:
    """Concatenate fields into a rich string for BM25 + embedding indexing."""
    parts = [
        raw.get("name", ""),
        raw.get("description", ""),
        f"Category: {', '.join(raw.get('keys', []))}",
        f"Job levels: {', '.join(raw.get('job_levels', []))}",
    ]
    langs = raw.get("languages", [])
    if langs:
        parts.append(f"Languages: {', '.join(langs[:8])}")
    dur = raw.get("duration", "")
    if dur:
        parts.append(f"Duration: {dur}")
    if raw.get("adaptive") == "yes":
        parts.append("Adaptive test")
    if raw.get("remote") == "yes":
        parts.append("Remote/online")
    return ". ".join(filter(None, parts))


# ── Module-level catalog stores (populated by load_catalog) ─────────────────

_catalog: list[CatalogItem] = []
_catalog_by_name: dict[str, CatalogItem] = {}
_catalog_by_url: dict[str, CatalogItem] = {}


def load_catalog(catalog_path: str | None = None) -> list[CatalogItem]:
    """Read the JSON file, validate each entry, populate lookup dicts."""
    global _catalog, _catalog_by_name, _catalog_by_url

    if catalog_path is None:
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "catalog.json",
        )

    logger.info(f"Loading catalog from {catalog_path}")
    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_items: list[dict] = json.load(f)

    items: list[CatalogItem] = []
    for raw in raw_items:
        # Only keep entries that scraped successfully
        if raw.get("status") != "ok":
            continue

        item = CatalogItem(
            entity_id=raw.get("entity_id", ""),
            name=raw.get("name", ""),
            link=raw.get("link", ""),
            description=raw.get("description", ""),
            keys=raw.get("keys", []),
            test_type=_build_test_type(raw.get("keys", [])),
            job_levels=raw.get("job_levels", []),
            languages=raw.get("languages", []),
            duration=raw.get("duration", ""),
            remote=raw.get("remote", ""),
            adaptive=raw.get("adaptive", ""),
            searchable_text=_build_searchable_text(raw),
        )
        items.append(item)

    _catalog = items
    _catalog_by_name = {item.name.lower().strip(): item for item in items}
    _catalog_by_url = {item.link.strip(): item for item in items}

    logger.info(f"Catalog loaded: {len(items)} items (filtered from {len(raw_items)} raw)")
    return items


# ── Public accessors ────────────────────────────────────────────────────────

def get_catalog() -> list[CatalogItem]:
    return _catalog


def get_catalog_by_name() -> dict[str, CatalogItem]:
    return _catalog_by_name


def get_catalog_by_url() -> dict[str, CatalogItem]:
    return _catalog_by_url


def find_item_by_name(name: str) -> CatalogItem | None:
    """Look up an item by exact or partial name match."""
    # Exact match
    item = _catalog_by_name.get(name.lower().strip())
    if item:
        return item

    # Substring / partial match
    name_lower = name.lower().strip()
    for key, item in _catalog_by_name.items():
        if name_lower in key or key in name_lower:
            return item

    return None


def find_items_by_names(names: list[str]) -> list[CatalogItem]:
    """Find multiple items by name, best-effort fuzzy matching."""
    results = []
    for name in names:
        item = find_item_by_name(name)
        if item:
            results.append(item)
    return results
