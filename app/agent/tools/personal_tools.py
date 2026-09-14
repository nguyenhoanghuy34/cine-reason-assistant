from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMP_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "temp-data"
)


def get_user_profile(user_id: int) -> dict[str, Any]:
    """
    Read the temporary user profile parquet file.
    """

    temp_path = (
        TEMP_DATA_DIR
        / f"user_{user_id}.parquet"
    )

    if not temp_path.exists():
        raise FileNotFoundError(
            f"Temporary data for user {user_id} not found: "
            f"{temp_path}"
        )

    data = pd.read_parquet(temp_path)

    row = data[data["user_id"] == user_id]

    if row.empty:
        raise ValueError(
            f"User {user_id} not found in temporary data."
        )

    row = row.iloc[0]

    return {
        "user_id": int(row["user_id"]),
        "high_rated_genres": row["high_rated_genres"],
        "high_rated_movies": row["high_rated_movies"],
        "user_tags": row["user_tags"],
        "low_rated_genres": row["low_rated_genres"],
        "unwatched_matching_movie_ids": row[
            "unwatched_matching_movie_ids"
        ],
        "top_2_genres": row["top_2_genres"],
    }


def get_user_summary(user_id: int) -> str:
    """
    Read the temporary TXT summary generated for the user.
    """

    summary_path = (
        TEMP_DATA_DIR
        / f"user_{user_id}_summary.txt"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"User summary not found for user {user_id}: "
            f"{summary_path}"
        )

    summary = summary_path.read_text(
        encoding="utf-8"
    ).strip()

    if not summary:
        raise ValueError(
            f"User summary is empty for user {user_id}."
        )

    return summary