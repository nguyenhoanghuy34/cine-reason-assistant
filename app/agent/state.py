from __future__ import annotations

from typing import Any, Literal, TypedDict


Intent = Literal[
    "GENERAL",
    "PERSONAL",
    "RELATED_USERS",
    "OTHER",
]


class AgentState(TypedDict, total=False):

    user_id: int | None

    query: str

    chat_history: list[str]

    intent: Intent

    intent_reason: str

    related_users_evidence: dict[str, Any]

    evidence: dict[str, Any]

    response: str

    target_user_ids: list[int]
    movie_titles: list[str]
    preferred_genres: list[str]
    excluded_genres: list[str]
    include_terms: list[str]
    exclude_terms: list[str]
    needs_recommendations: bool
    needs_genre_analysis: bool
