from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

CLEAN_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
)

SOURCE_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)

USER_PROFILES_PATH = (
    CLEAN_DATA_DIR
    / "user_profiles.parquet"
)

MOVIES_PATH = (
    SOURCE_DATA_DIR
    / "movies_with_plots.csv"
)


def _normalize_movie_ids(value) -> list[int]:
    """
    Convert unwatched_matching_movie_ids
    into a clean list of integers.
    """

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            value = eval(
                value,
                {
                    "__builtins__": {}
                },
                {},
            )
        except Exception:
            return []

    try:
        return [
            int(movie_id)
            for movie_id in value
        ]
    except (TypeError, ValueError):
        return []


def _normalize_genres(value) -> list[str]:
    """
    Convert top_2_genres into a clean list.
    """

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            value = eval(
                value,
                {
                    "__builtins__": {}
                },
                {},
            )
        except Exception:
            return [
                genre.strip()
                for genre in value.split(",")
                if genre.strip()
            ]

    try:
        return [
            str(genre)
            for genre in value
        ]
    except TypeError:
        return []


def get_recommendation_candidates(
    user_id: int,
    max_candidates: int = 50,
) -> dict[str, Any]:

    # ============================================================
    # 1. LOAD USER PROFILE
    # ============================================================

    user_profiles = pd.read_parquet(
        USER_PROFILES_PATH
    )

    user_profile = user_profiles[
        user_profiles["user_id"] == user_id
    ]

    if user_profile.empty:
        raise ValueError(
            f"User {user_id} not found in "
            "user_profiles.parquet"
        )

    user_profile = user_profile.iloc[0]

    # ============================================================
    # 2. GET USER PREFERENCE SIGNALS
    # ============================================================

    top_2_genres = _normalize_genres(
        user_profile["top_2_genres"]
    )

    movie_ids = _normalize_movie_ids(
        user_profile[
            "unwatched_matching_movie_ids"
        ]
    )

    # ============================================================
    # 3. LIMIT CANDIDATES
    # ============================================================

    movie_ids = movie_ids[
        :max_candidates
    ]

    if not movie_ids:
        return {
            "user_id": user_id,
            "top_2_genres": top_2_genres,
            "candidate_movies": [],
        }

    # ============================================================
    # 4. LOAD MOVIE DATA
    # ============================================================

    movies = pd.read_csv(
        MOVIES_PATH
    )

    movies["movieId"] = movies[
        "movieId"
    ].astype(int)

    # ============================================================
    # 5. FILTER BY MOVIE IDS
    # ============================================================

    candidates = movies[
        movies["movieId"].isin(movie_ids)
    ].copy()

    # ============================================================
    # 6. PRESERVE ORIGINAL MOVIE ID ORDER
    # ============================================================

    movie_order = {
        movie_id: index
        for index, movie_id
        in enumerate(movie_ids)
    }

    candidates["_order"] = candidates[
        "movieId"
    ].map(movie_order)

    candidates = candidates.sort_values(
        "_order"
    )

    # ============================================================
    # 7. BUILD LLM EVIDENCE
    # ============================================================

    candidate_movies = []

    for row in candidates.itertuples(
        index=False
    ):
        candidate_movies.append(
            {
                "movie_id": int(
                    row.movieId
                ),
                "title": str(
                    row.title
                ),
                "year": (
                    int(row.year)
                    if pd.notna(row.year)
                    else None
                ),
                "genres": str(
                    row.genres
                ),
                "plot": str(
                    row.plot
                ),
            }
        )

    # ============================================================
    # 8. RETURN
    # ============================================================

    return {
        "user_id": user_id,
        "top_2_genres": top_2_genres,
        "candidate_movies": candidate_movies,
    }