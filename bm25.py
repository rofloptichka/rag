import os
from typing import List, Dict, Any

from meilisearch import Client

MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_MASTER_KEY", "")

_client = Client(MEILI_URL, MEILI_KEY)


def _kb_index_name(company_id: str) -> str:
    return f"kb_{company_id}"


def _ensure_kb_index(company_id: str):
    name = _kb_index_name(company_id)
    try:
        _client.get_index(name)
    except Exception:
        _client.create_index(name, {"primaryKey": "id"})
        idx = _client.index(name)
        idx.update_settings({
            "searchableAttributes": ["title", "text"],
            "displayedAttributes": ["id", "title", "filename", "url"],
            "filterableAttributes": ["filename"],
        })
    return _client.index(name)


def meili_upsert(company_id: str, docs: List[Dict[str, Any]]):
    idx = _ensure_kb_index(company_id)
    if not docs:
        return
    idx.add_documents(docs)


def meili_search(company_id: str, query: str, limit: int = 50) -> List[str]:
    idx = _ensure_kb_index(company_id)
    res = idx.search(query, {"limit": limit, "attributesToRetrieve": ["id"]})
    return [hit.get("id") for hit in (res or {}).get("hits", []) if hit.get("id") is not None]


