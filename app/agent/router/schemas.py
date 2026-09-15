from typing import Literal

from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    intent: Literal[
        "GENERAL",
        "PERSONAL",
        "RELATED_USERS",
        "OTHER",
    ]

    reason: str

    target_user_ids: list[int] = Field(default_factory=list, description="Explicit dataset user IDs requested, resolved from conversation. Never invent IDs.")
    movie_titles: list[str] = Field(default_factory=list, description="Movie titles whose ratings are requested, resolved from conversation.")
    preferred_genres: list[str] = Field(default_factory=list, description="Genres the user asks for, such as Thriller, Mystery, Drama.")
    excluded_genres: list[str] = Field(default_factory=list, description="Genres the user wants to avoid, such as Animation.")
    include_terms: list[str] = Field(default_factory=list, description="Natural-language concepts to search for in plot, tags, title or genres.")
    exclude_terms: list[str] = Field(default_factory=list, description="Natural-language concepts the user wants to avoid.")
    needs_recommendations: bool = False
    needs_genre_analysis: bool = False
