from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from app.agent.tools.related_user_tools import get_related_user_ids


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def get_similar_users(user_id: int, limit: int = 10) -> dict[str, Any]:
    related_ids = get_related_user_ids(user_id)[:limit]
    return {
        "user_id": user_id,
        "similar_user_ids": related_ids,
        "similarity_source": "precomputed user_similarity.parquet ranking",
    }


def get_user_similarity_details(user_ids: list[int], limit: int = 10) -> dict[str, Any]:
    user_ids = [int(user_id) for user_id in user_ids if user_id is not None]
    if len(user_ids) < 2:
        return {"user_ids": user_ids, "pairs": []}

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    ratings = ratings[ratings["userId"].isin(user_ids)]

    pairs = []
    for left, right in combinations(user_ids, 2):
        left_ratings = ratings[ratings["userId"] == left][["movieId", "rating"]]
        right_ratings = ratings[ratings["userId"] == right][["movieId", "rating"]]
        shared = left_ratings.merge(
            right_ratings,
            on="movieId",
            suffixes=("_left", "_right"),
        )
        if shared.empty:
            pairs.append({
                "left_user_id": left,
                "right_user_id": right,
                "shared_rating_count": 0,
                "average_abs_rating_gap": None,
                "shared_movies": [],
            })
            continue

        shared["abs_gap"] = (shared["rating_left"] - shared["rating_right"]).abs()
        examples = shared.sort_values("abs_gap").head(limit).merge(movies, on="movieId")
        shared_movies = []
        for _, row in examples.iterrows():
            shared_movies.append({
                "title": str(row["title"]),
                "left_rating": float(row["rating_left"]),
                "right_rating": float(row["rating_right"]),
                "abs_gap": round(float(row["abs_gap"]), 2),
            })

        pairs.append({
            "left_user_id": left,
            "right_user_id": right,
            "shared_rating_count": int(len(shared)),
            "average_abs_rating_gap": round(float(shared["abs_gap"].mean()), 2),
            "shared_movies": shared_movies,
        })

    return {"user_ids": user_ids, "pairs": pairs}
