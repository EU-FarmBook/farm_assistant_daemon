# app/services/search_service.py

import logging

import httpx

from typing import Dict, Any, List, Optional

from app.config import get_settings
from app.schemas import AskIn

logger = logging.getLogger("farm-assistant.search")
S = get_settings()


def _rag_url() -> str:
    return f"{S.OPENSEARCH_API_URL.rstrip('/')}{S.OS_RAG_API_PATH}"

def build_search_payload(inp: AskIn) -> Dict[str, Any]:
    default_k = 5
    requested_k = inp.top_k if inp.top_k is not None else inp.k
    if not isinstance(requested_k, int) or requested_k <= 0:
        requested_k = default_k

    # Pull a slightly wider candidate set than the final grounded context so
    # Farm Assistant can discard weak OpenSearch hits before prompting the LLM.
    retrieval_k = max(int(requested_k), int(S.RETRIEVAL_CANDIDATE_K))

    payload: Dict[str, Any] = {
        "search_term": inp.question,
        "page": 1,
        "dev": False,
        "model": "msmarco",
        "sort_by": "score_desc",
        "k": retrieval_k,
    }

    # In semantic relevance mode, ask scout to attach a calibrated per-chunk
    # semantic_score so the sufficiency gate can threshold meaning, not word overlap.
    if (S.RELEVANCE_MODE or "").strip().lower() == "semantic":
        payload["include_semantic_score"] = True

    return payload

async def fetch_os_page(
    client: httpx.AsyncClient,
    base_payload: Dict[str, Any],
    page_no: int,
    headers: Dict[str, str],
    auth: Optional[httpx.Auth]
) -> Dict[str, Any]:
    payload = dict(base_payload)
    payload["page"] = page_no
    url = _rag_url()
    res = await client.post(url, json=payload, headers=headers, auth=auth)
    res.raise_for_status()
    return res.json()

async def collect_os_items(
    client: httpx.AsyncClient,
    base_payload: Dict[str, Any],
    pages: List[int],
    headers: Dict[str, str],
    auth: Optional[httpx.Auth]
) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    seen_ids = set()
    for p in pages:
        os_json = await fetch_os_page(client, base_payload, p, headers, auth)
        items = (
            os_json.get("data")
            or os_json.get("results")
            or os_json.get("hits")
            or os_json.get("items")
            or []
        )
        for it in items:
            _id = (it.get("_id") if isinstance(it, dict) else None) or id(it)
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            all_items.append(it)
    logger.info(f"Collected total_items={len(all_items)} from pages={pages}")
    return all_items
