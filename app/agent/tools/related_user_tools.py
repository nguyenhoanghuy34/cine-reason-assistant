from __future__ import annotations

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

DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)


def get_related_user_ids(
    user_id: int,
) -> list[int]:
    """
    Get users with similar movie taste.
    """

    similarity_path = (
        CLEAN_DATA_DIR
        / "user_similarity.parquet"
    )

    if not similarity_path.exists():
        raise FileNotFoundError(
            f"User similarity file not found: "
            f"{similarity_path}"
        )

    data = pd.read_parquet(
        similarity_path
    )

    row = data[
        data["user_id"] == user_id
    ]

    if row.empty:
        raise ValueError(
            f"User {user_id} not found in "
            f"user_similarity.parquet"
        )

    related_user_ids = row.iloc[0][
        "related_user_ids"
    ]

    if related_user_ids is None:
        return []

    return [
        int(related_id)
        for related_id in related_user_ids
        if int(related_id) != user_id
    ]


def get_top_movies_from_related_users(
    user_id: int,
    top_users: int = 10,
    top_movies: int = 15,
    min_rating: float = 4.0,
) -> dict[str, Any]:
    """
    Get highly rated movies from users with similar taste.

    Process:
    1. Find related users.
    2. Take the first top_users.
    3. Get their ratings.
    4. Keep high ratings.
    5. Aggregate ratings by movie.
    6. Join movie metadata.
    7. Return top movies.
    """

    related_user_ids = get_related_user_ids(
        user_id
    )

    if not related_user_ids:
        return {
            "user_id": user_id,
            "related_user_ids": [],
            "movies": [],
        }

    # --------------------------------------------------------------
    # Limit number of related users
    # --------------------------------------------------------------

    related_user_ids = related_user_ids[
        :top_users
    ]

    # --------------------------------------------------------------
    # Read ratings
    # --------------------------------------------------------------

    ratings_path = (
        DATA_DIR
        / "ratings.csv"
    )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: "
            f"{ratings_path}"
        )

    ratings = pd.read_csv(
        ratings_path
    )

    related_ratings = ratings[
        ratings["userId"].isin(
            related_user_ids
        )
    ].copy()

    if related_ratings.empty:
        return {
            "user_id": user_id,
            "related_user_ids": related_user_ids,
            "movies": [],
        }

    # --------------------------------------------------------------
    # Keep highly rated movies
    # --------------------------------------------------------------

    high_ratings = related_ratings[
        related_ratings["rating"] >= min_rating
    ].copy()

    if high_ratings.empty:
        return {
            "user_id": user_id,
            "related_user_ids": related_user_ids,
            "movies": [],
        }

    # --------------------------------------------------------------
    # Aggregate by movie
    # --------------------------------------------------------------

    movie_stats = (
        high_ratings
        .groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count"),
            users=("userId", "count"),
        )
        .reset_index()
    )

    # Strongest evidence:
    # - high average rating
    # - more similar users rating the movie
    movie_stats = movie_stats.sort_values(
        by=[
            "rating_count",
            "average_rating",
        ],
        ascending=[
            False,
            False,
        ],
    )

    movie_stats = movie_stats.head(
        top_movies
    )

    # --------------------------------------------------------------
    # Read movie metadata
    # --------------------------------------------------------------

    movies_path = (
        DATA_DIR
        / "movies.csv"
    )

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Movies file not found: "
            f"{movies_path}"
        )

    movies = pd.read_csv(
        movies_path
    )

    movie_stats = movie_stats.merge(
        movies[
            [
                "movieId",
                "title",
                "genres",
            ]
        ],
        on="movieId",
        how="left",
    )

    # --------------------------------------------------------------
    # Build compact evidence
    # --------------------------------------------------------------

    movie_evidence = []

    for _, row in movie_stats.iterrows():

        movie_evidence.append(
            {
                "movie_id": int(
                    row["movieId"]
                ),
                "title": str(
                    row["title"]
                ),
                "genres": str(
                    row["genres"]
                ),
                "average_rating": round(
                    float(
                        row["average_rating"]
                    ),
                    2,
                ),
                "rating_count": int(
                    row["rating_count"]
                ),
            }
        )

    return {
        "user_id": user_id,
        "related_user_ids": related_user_ids,
        "movies": movie_evidence,
    }


def get_movie_opinions(user_ids: list[int], titles: list[str]) -> dict[str, Any]:
    """Retrieve all recorded ratings for requested films, including low ratings."""
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    results = []
    for title in titles:
        matches = movies[movies["title"].str.contains(title, case=False, regex=False, na=False)]
        selected = ratings[
            ratings["userId"].isin(user_ids) & ratings["movieId"].isin(matches["movieId"])
        ].merge(matches[["movieId", "title"]], on="movieId")
        results.append({
            "requested_title": title,
            "matched_movies": matches[["movieId", "title"]].to_dict("records"),
            "ratings": selected[["userId", "movieId", "title", "rating"]].to_dict("records"),
        })
    return {"user_ids": user_ids, "results": results}
