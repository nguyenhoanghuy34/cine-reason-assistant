from __future__ import annotations

from typing import Literal, TypedDict


Intent = Literal[
    "GENERAL",
    "PERSONAL",
    "RELATED_USERS",
    "OTHER",
]


class AgentState(TypedDict, total=False):
    user_id: int | None
    query: str

    intent: Intent
    intent_reason: str

    response: str