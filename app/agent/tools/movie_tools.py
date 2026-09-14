from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MOVIES_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
    / "movies_with_plots.csv"
)


def get_candidate_movies(
    movie_ids: list[int],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get movie information for candidate movie IDs.

    Returns movie evidence only.
    Recommendation decisions are handled by the LLM.
    """

    if movie_ids is None or len(movie_ids) == 0:
        return []

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Movie dataset not found: {MOVIES_PATH}"
        )

    movies = pd.read_csv(MOVIES_PATH)

    candidates = movies[
        movies["movieId"].isin(movie_ids)
    ].copy()

    candidates = candidates.head(limit)

    results = []

    for _, row in candidates.iterrows():
        results.append(
            {
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "year": row["year"],
                "genres": row["genres"],
                "plot": row["plot"],
            }
        )

    return results