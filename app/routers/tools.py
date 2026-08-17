# app/routers/tools.py
"""
The private tool surface the MCP bridge calls.

Reachable only from the agent container, on the compose network, and only with
the shared bridge key. It is NOT part of the public chat API and is excluded
from the schema: nothing a browser can reach should be able to drive a
retrieval or write a memory note out of band.
"""

import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services import tool_server
from app.services.profile_registry import is_valid_profile_name

S = get_settings()
logger = logging.getLogger("farm-assistant-hermes.tools")
router = APIRouter(prefix="/internal/tools", include_in_schema=False)


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = None


class RememberIn(BaseModel):
    fact: str = Field(min_length=1)


class ForgetIn(BaseModel):
    marker: str = Field(min_length=1, examples=["M2"])


def _authorize(bridge_key: Optional[str], profile: Optional[str]) -> str:
    expected = S.HERMES_API_KEY  # the bridge shares the agent's key; one secret, one blast radius
    if not expected or not bridge_key or not hmac.compare_digest(bridge_key, expected):
        # Almost always a STALE key rather than an attack: Hermes froze the old
        # value into a long-running MCP subprocess. Symptom is an agent that can
        # no longer search while everything else works, so name the cause here.
        logger.error(
            "Bridge call rejected for profile=%s: key mismatch. If HERMES_API_KEY "
            "changed, the agent's MCP server may hold the old one — it re-reads "
            "/opt/data/bridge.key per call, so check that file is current.",
            profile,
        )
        raise HTTPException(status_code=401, detail="Unauthorized.")
    if not profile or not is_valid_profile_name(profile):
        raise HTTPException(status_code=400, detail="Missing or invalid profile.")
    return profile


@router.post("/search")
async def search(
    body: SearchIn,
    x_bridge_key: Optional[str] = Header(default=None),
    x_euf_profile: Optional[str] = Header(default=None),
):
    profile = _authorize(x_bridge_key, x_euf_profile)
    return await tool_server.search_eu_farmbook(body.query, profile=profile, top_k=body.top_k)


@router.post("/forget")
async def forget(
    body: ForgetIn,
    x_bridge_key: Optional[str] = Header(default=None),
    x_euf_profile: Optional[str] = Header(default=None),
):
    profile = _authorize(x_bridge_key, x_euf_profile)
    return await tool_server.forget_about_user(body.marker, profile=profile)


@router.post("/remember")
async def remember(
    body: RememberIn,
    x_bridge_key: Optional[str] = Header(default=None),
    x_euf_profile: Optional[str] = Header(default=None),
):
    profile = _authorize(x_bridge_key, x_euf_profile)
    return await tool_server.remember_about_user(body.fact, profile=profile)
