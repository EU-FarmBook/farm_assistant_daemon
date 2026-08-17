#!/usr/bin/env python
"""
MCP bridge: the agent's only door to the outside world.

Runs inside the Hermes container as a stdio MCP server (mcp 1.28.1 + httpx
0.28.1 are already in that image; no extra install). It holds no OpenSearch
credentials and no user token — it forwards to the adapter, which does the
privileged work and knows whose turn is in flight.

Env, seeded per profile at provisioning:
    EUF_ADAPTER_URL   adapter base URL on the compose network
    EUF_BRIDGE_KEY    shared secret with the adapter (== API_SERVER_KEY)
    EUF_PROFILE       the Hermes profile this server runs for; the adapter uses
                      it to find the in-flight turn, so it MUST be per-profile
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

ADAPTER_URL = os.environ.get("EUF_ADAPTER_URL", "http://adapter:8100").rstrip("/")
PROFILE = os.environ.get("EUF_PROFILE", "")

# The key is read from disk on EVERY call, never captured at import.
#
# Hermes spawns this as a long-lived subprocess with the profile's env frozen at
# spawn time. With the key in that env, rotating it left every running MCP
# server presenting the old one — the adapter answered 401, the agent quietly
# stopped being able to search, and the only clue was a 401 in a log nobody was
# watching. A file re-read per call makes a key change take effect immediately,
# with no restart of the agent container.
BRIDGE_KEY_FILE = os.environ.get("EUF_BRIDGE_KEY_FILE", "/opt/data/bridge.key")


def _bridge_key() -> str:
    try:
        with open(BRIDGE_KEY_FILE, encoding="utf-8") as fh:
            key = fh.read().strip()
            if key:
                return key
    except OSError:
        pass
    # Fall back to the spawn-time env so an older profile still works.
    return os.environ.get("EUF_BRIDGE_KEY", "")


mcp = FastMCP("eu-farmbook")


def _headers() -> dict:
    return {"X-Bridge-Key": _bridge_key(), "X-EUF-Profile": PROFILE}


@mcp.tool()
async def search_eu_farmbook(query: str, top_k: int = 5) -> dict:
    """Search EU-FarmBook knowledge objects and return numbered passages to cite.

    Use this before answering any question that asks for facts, figures, practices,
    regulations or project information. Cite the passages you use as [1], [2], ...
    If it returns no passages, say EU-FarmBook has no material on the question
    instead of answering from your own knowledge.

    Args:
        query: The search query, in the language of the source material where possible.
        top_k: How many passages to return (default 5).
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{ADAPTER_URL}/internal/tools/search",
            json={"query": query, "top_k": top_k},
            headers=_headers(),
        )
        if r.status_code != 200:
            return {"ok": False, "error": f"Search backend returned HTTP {r.status_code}", "passages": []}
        return r.json()


@mcp.tool()
async def remember_about_user(fact: str) -> dict:
    """Store one durable fact about the user for future conversations.

    Only for things the user said about themselves that will still matter later —
    where they farm, what they grow, their role, a standing preference. One clear
    sentence. Never store your own answers, retrieved passages, sensitive personal
    data, or anything the user did not state about themselves.

    Args:
        fact: A single self-contained sentence about the user.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{ADAPTER_URL}/internal/tools/remember",
            json={"fact": fact},
            headers=_headers(),
        )
        if r.status_code != 200:
            return {"ok": False, "error": f"Memory backend returned HTTP {r.status_code}"}
        return r.json()


if __name__ == "__main__":
    mcp.run()
