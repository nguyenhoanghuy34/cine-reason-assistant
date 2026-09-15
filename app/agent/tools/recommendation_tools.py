from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

TAGS_PATH = (
    SOURCE_DATA_DIR
    / "tags.csv"
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
            value = ast.literal_eval(value)
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
            value = ast.literal_eval(value)
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


def get_recommendation_evidence(
    user_id: int,
    max_candidates: int = 50,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible alias used by the agent nodes.
    """
    return get_recommendation_candidates(
        user_id=user_id,
        max_candidates=max_candidates,
        constraints=constraints,
    )


def _normalize_terms(values) -> list[str]:
    if not values:
        return []
    return [
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    ]


def _add_tags(movies: pd.DataFrame) -> pd.DataFrame:
    if not TAGS_PATH.exists():
        movies["tags"] = ""
        return movies

    tags = pd.read_csv(TAGS_PATH)
    if tags.empty or "movieId" not in tags or "tag" not in tags:
        movies["tags"] = ""
        return movies

    tag_text = (
        tags.dropna(subset=["tag"])
        .assign(tag=lambda frame: frame["tag"].astype(str))
        .groupby("movieId")["tag"]
        .apply(lambda values: " ".join(sorted(set(values.str.lower()))))
        .reset_index(name="tags")
    )
    return movies.merge(tag_text, on="movieId", how="left").fillna({"tags": ""})


def _matches_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms)


def _matches_none(text: str, terms: list[str]) -> bool:
    return not any(term in text for term in terms)


def _score_candidates(candidates: pd.DataFrame, constraints: dict[str, Any]) -> pd.DataFrame:
    preferred_genres = _normalize_terms(constraints.get("preferred_genres"))
    excluded_genres = _normalize_terms(constraints.get("excluded_genres"))
    include_terms = _normalize_terms(constraints.get("include_terms"))
    exclude_terms = _normalize_terms(constraints.get("exclude_terms"))

    if not any([preferred_genres, excluded_genres, include_terms, exclude_terms]):
        candidates["_constraint_score"] = 0
        return candidates

    candidates = _add_tags(candidates.copy())
    candidates["_search_text"] = (
        candidates["title"].astype(str) + " "
        + candidates["genres"].astype(str) + " "
        + candidates["plot"].astype(str) + " "
        + candidates["tags"].astype(str)
    ).str.lower()
    candidates["_genre_text"] = candidates["genres"].astype(str).str.lower()

    if excluded_genres:
        candidates = candidates[
            candidates["_genre_text"].map(lambda text: _matches_none(text, excluded_genres))
        ]
    if exclude_terms:
        candidates = candidates[
            candidates["_search_text"].map(lambda text: _matches_none(text, exclude_terms))
        ]
    if preferred_genres:
        candidates = candidates[
            candidates["_genre_text"].map(lambda text: any(term in text for term in preferred_genres))
        ]

    def score(row) -> int:
        text = row["_search_text"]
        genre_text = row["_genre_text"]
        genre_score = sum(3 for term in preferred_genres if term in genre_text)
        term_score = sum(1 for term in include_terms if term in text)
        return genre_score + term_score

    candidates["_constraint_score"] = candidates.apply(score, axis=1)
    if include_terms:
        matching = candidates[candidates["_search_text"].map(lambda text: _matches_all(text, include_terms))]
        if not matching.empty:
            candidates = matching

    return candidates.sort_values(["_constraint_score", "_order"], ascending=[False, True])


def get_recommendation_candidates(
    user_id: int,
    max_candidates: int = 50,
    constraints: dict[str, Any] | None = None,
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

    candidates = candidates.sort_values("_order")
    constraints = constraints or {}
    candidates = _score_candidates(candidates, constraints)

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
        "constraints": constraints,
        "candidate_movies": candidate_movies,
    }
