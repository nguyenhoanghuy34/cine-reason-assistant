from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROFILE_FILE = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
    / "user_profiles.parquet"
)


def _parse_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else []

    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []

    return []


def _parse_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    return {}


def _safe_int(value) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def get_blind_spot_evidence(user_id: int) -> dict:
    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Profile file not found: {PROFILE_FILE}"
        )

    df = pd.read_parquet(PROFILE_FILE)

    user = df[df["user_id"] == user_id]

    if user.empty:
        raise ValueError(f"User {user_id} not found")

    row = user.iloc[0]

    watch_counts = _parse_dict(row["genre_watch_counts"])
    avg_ratings = _parse_dict(row["genre_avg_ratings"])
    high_counts = _parse_dict(row["genre_high_rated_counts"])
    high_ratios = _parse_dict(row["genre_high_rating_ratio"])

    unwatched_genres = _parse_list(row["unwatched_genres"])
    underexposed_genres = _parse_list(row["underexposed_genres"])

    all_genres = set(watch_counts)

    all_genres.update(unwatched_genres)
    all_genres.update(underexposed_genres)

    genre_analysis = []

    for genre in sorted(all_genres):
        watch_count = _safe_int(watch_counts.get(genre, 0))
        avg_rating = avg_ratings.get(genre)
        high_count = _safe_int(high_counts.get(genre, 0))
        high_ratio = high_ratios.get(genre)

        if watch_count == 0:
            exposure = "UNWATCHED"
        elif watch_count <= 2:
            exposure = "UNDEREXPOSED"
        else:
            exposure = "EXPLORED"

        genre_analysis.append(
            {
                "genre": genre,
                "watch_count": watch_count,
                "avg_rating": avg_rating,
                "high_rated_count": high_count,
                "high_rating_ratio": high_ratio,
                "exposure": exposure,
            }
        )

    return {
        "user_id": user_id,
        "genre_analysis": genre_analysis,
        "unwatched_genres": unwatched_genres,
        "underexposed_genres": underexposed_genres,
    }
