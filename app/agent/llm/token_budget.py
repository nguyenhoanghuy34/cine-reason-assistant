from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MAX_STRING_CHARS = 1200
MAX_LIST_ITEMS = 8
MAX_DICT_ITEMS = 40
MAX_HISTORY_ITEMS = 8


IMPORTANT_KEYS = {
    "user_id",
    "current_user_id",
    "query",
    "question",
    "intent",
    "title",
    "genres",
    "tags",
    "rating",
    "average_rating",
    "num_ratings",
    "overview",
    "summary",
    "reason",
    "constraints",
    "candidate_movies",
    "movies",
    "results",
    "ratings",
    "top_rated_movies",
    "low_rated_movies",
    "top_2_genres",
    "favorite_genres",
    "least_favorite_genres",
    "unwatched_genres",
    "underexposed_genres",
    "related_user_ids",
    "similar_users",
    "similarity_ranking",
    "similarity_details",
    "user_profile",
    "user_rating_history",
    "user_tags",
    "movie_info",
    "movie_summary",
    "recommendations",
    "genre_analysis",
}


def compact_text(text: Any, max_chars: int = MAX_STRING_CHARS) -> str:
    value = str(text).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def compact_history(history: Sequence[Any], max_items: int = MAX_HISTORY_ITEMS) -> list[str]:
    recent = list(history or [])[-max_items:]
    return [
        compact_text(item, 500)
        for item in recent
        if isinstance(item, str) and item.strip()
    ]


def compact_value(
    value: Any,
    *,
    max_string_chars: int = MAX_STRING_CHARS,
    max_list_items: int = MAX_LIST_ITEMS,
    max_dict_items: int = MAX_DICT_ITEMS,
) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return compact_text(value, max_string_chars)

    if isinstance(value, Mapping):
        ordered_keys = [
            key for key in value.keys()
            if str(key) in IMPORTANT_KEYS
        ]
        ordered_keys.extend(
            key for key in value.keys()
            if key not in ordered_keys
        )
        return {
            str(key): compact_value(
                value[key],
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for key in ordered_keys[:max_dict_items]
        }

    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [
            compact_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in list(value)[:max_list_items]
        ]

    return compact_text(value, max_string_chars)
