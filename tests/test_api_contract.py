"""
Contract test: every endpoint the v3 UI calls must exist on this service.

This exists because it already went wrong once. The v3 shell is a copy of the v2
shell, so it calls the v2 settings-dialog endpoints — and the adapter had only
its own `/memory/documents` variants, which meant Personalization, Custom
instructions and the Memory tab would all have 404'd against a service that
looked healthy.

If you add a call in `src/services/farm_assistant_v3/client.ts`, add its path
here. A failure means the UI is about to ask for something nobody serves.
"""

from app.main import app

# (method, path) as the frontend proxy forwards them, minus the /chatbot/api
# prefix handling FastAPI does for us.
REQUIRED = [
    # Streaming chat
    ("get", "/chatbot/api/chats/message/stream"),
    ("get", "/chatbot/api/chats/{session_id}/message/stream"),
    # Session list / detail / transcript
    ("get", "/chatbot/api/chats"),
    ("get", "/chatbot/api/chats/{session_id}"),
    ("delete", "/chatbot/api/chats/{session_id}"),
    ("post", "/chatbot/api/chats/log-turn"),
    # Settings dialog — Personalization + Custom instructions
    ("get", "/chatbot/api/users/me/settings"),
    ("patch", "/chatbot/api/users/me/settings"),
    # Settings dialog — Memory
    ("get", "/chatbot/api/users/me/memory"),
    ("delete", "/chatbot/api/users/me/memory/{note_id}"),
    ("post", "/chatbot/api/users/me/memory/summary"),
]


def test_every_endpoint_the_ui_calls_exists():
    paths = app.openapi()["paths"]
    missing = [
        f"{method.upper()} {path}"
        for method, path in REQUIRED
        if path not in paths or method not in paths[path]
    ]
    assert not missing, f"UI calls endpoints this service does not serve: {missing}"


def test_internal_tool_surface_is_not_published():
    # The MCP bridge endpoints carry their own key and must never appear in the
    # public schema — nothing a browser reaches should drive a retrieval or a
    # memory write out of band.
    paths = app.openapi()["paths"]
    assert not [p for p in paths if p.startswith("/internal/")]


def test_the_endpoints_added_for_suggestions_and_attachments_exist():
    """
    These three were each added because the copied v2 shell already called them
    and got nothing: follow-up chips rendered as empty grey pills, the `+`
    button failed, and the empty chat had no openers.
    """
    paths = app.openapi()["paths"]
    for method, path in [
        ("post", "/chatbot/api/follow-ups"),
        ("get", "/chatbot/api/users/me/suggestions"),
        ("post", "/chatbot/api/files/document"),
        ("delete", "/chatbot/api/files/document/{doc_id}"),
        ("get", "/chatbot/api/chats/{session_id}/attachments"),
    ]:
        assert path in paths and method in paths[path], f"missing {method.upper()} {path}"
