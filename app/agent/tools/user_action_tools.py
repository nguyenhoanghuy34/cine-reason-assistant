from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TAGS_PATH = DATA_DIR / "tags.csv"


def get_user_rating_history(user_id: int, limit: int = 25) -> dict[str, Any]:
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    user_ratings = ratings[ratings["userId"] == user_id].copy()
    user_ratings = user_ratings.sort_values(["rating", "timestamp"], ascending=[False, False])
    user_ratings = user_ratings.head(limit).merge(movies, on="movieId", how="left")

    items = []
    for _, row in user_ratings.iterrows():
        items.append({
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "rating": float(row["rating"]),
        })

    return {
        "user_id": user_id,
        "top_rated_movies": items,
        "rating_count": int((ratings["userId"] == user_id).sum()),
    }


def get_user_tags(user_id: int, limit: int = 25) -> dict[str, Any]:
    tags = pd.read_csv(TAGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    user_tags = tags[tags["userId"] == user_id].copy()
    if user_tags.empty:
        return {"user_id": user_id, "tags": []}

    user_tags = user_tags.sort_values("timestamp", ascending=False).head(limit)
    user_tags = user_tags.merge(movies, on="movieId", how="left")

    items = []
    for _, row in user_tags.iterrows():
        items.append({
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "tag": str(row["tag"]),
        })

    return {"user_id": user_id, "tags": items}
