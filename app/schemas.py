# app/schemas.py
#
# The wire contract, copied field-for-field from farm_assistant_um for the parts
# the v3 shell consumes. It is copied rather than trimmed to "what Hermes needs"
# on purpose: the whole value of the pilot is that v2 and v3 are swappable behind
# the same UI, and that only holds while the payloads are identical.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskIn(BaseModel):
    question: str
    page: Optional[int] = Field(default=None, examples=[1])
    k: Optional[int] = Field(default=None, examples=[5])
    model: Optional[str] = None
    include_fulltext: Optional[bool] = Field(default=None, examples=[True])
    sort_by: Optional[str] = Field(default=None, examples=["score_desc"])
    dev: Optional[bool] = Field(default=False, examples=[False])
    max_tokens: Optional[int] = Field(default=None, examples=[2000])
    temperature: Optional[float] = Field(default=None, examples=[0.4])
    top_k: Optional[int] = Field(default=None, examples=[4])


class SourceItem(BaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    subtitle: Optional[str] = None
    description: Optional[str] = None
    project: Optional[str] = None
    license: Optional[str] = None
    keywords: Optional[list[str]] = None
    topics: Optional[list[str]] = None
    themes: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    creators: Optional[list[str]] = None
    date_of_completion: Optional[str] = None
    display_url: Optional[str] = None
    category: Optional[str] = None  # KO category, e.g. "Document"
    sid: str | None = None


class ChatSessionCreateIn(BaseModel):
    title: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSessionPatchIn(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageFeedbackIn(BaseModel):
    feedback: str = Field(examples=["up", "down", "none"])
    meta: Dict[str, Any] = Field(default_factory=dict)


class DocumentExportIn(BaseModel):
    title: str = ""
    content: str = ""
    format: str = Field(examples=["pdf", "docx", "csv", "xlsx", "pptx"])
    sources: List[SourceItem] = Field(default_factory=list)


class ExportIntentIn(BaseModel):
    query: str = ""
    previous_assistant_message: str = ""


class TitleIn(BaseModel):
    question: str = ""
    answer: str = ""


class ChatTurnLogIn(BaseModel):
    session_uuid: Optional[str] = None
    user_message: str
    assistant_message: str
    meta: Dict[str, Any] = Field(default_factory=dict)


# --- Memory surfaces (v3-only) -------------------------------------------
#
# v2 reads memory from Django rows. v3's memory lives in the user's Hermes
# profile as MEMORY.md / USER.md, so the settings dialog needs a different
# shape: two documents, not a list of notes.

class MemoryDocument(BaseModel):
    """One of the agent's memory files, as text."""
    name: str                      # "MEMORY.md" | "USER.md"
    content: str
    char_count: int
    char_limit: int                # from config.yaml; over-limit content is truncated at read time by Hermes
    updated_at: Optional[str] = None


class MemoryDocumentsOut(BaseModel):
    profile: str
    documents: List[MemoryDocument]


class MemoryDocumentPatchIn(BaseModel):
    """
    A user edit to one of their own memory files.

    The user is allowed to rewrite these wholesale — that is the mneme model:
    the agent writes, the human reviews and corrects. `expected_char_count`
    makes the write a compare-and-swap so a concurrent agent write is not
    silently clobbered by a stale editor buffer.
    """
    content: str
    expected_char_count: Optional[int] = None
