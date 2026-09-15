from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
MOVIES_WITH_PLOTS_PATH = DATA_DIR / "movies_with_plots.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TAGS_PATH = DATA_DIR / "tags.csv"


def _movie_matches(titles: list[str]) -> pd.DataFrame:
    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    if not titles:
        return movies.head(0)

    masks = [
        movies["title"].str.contains(title, case=False, regex=False, na=False)
        for title in titles
        if title
    ]
    if not masks:
        return movies.head(0)

    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return movies[mask].copy()


def get_movie_info(titles: list[str], limit: int = 5) -> dict[str, Any]:
    matches = _movie_matches(titles).head(limit)
    if matches.empty:
        return {"requested_titles": titles, "movies": []}

    ratings = pd.read_csv(RATINGS_PATH)
    stats = (
        ratings.groupby("movieId")
        .agg(average_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )

    tags = pd.read_csv(TAGS_PATH)
    tag_text = (
        tags.dropna(subset=["tag"])
        .assign(tag=lambda frame: frame["tag"].astype(str))
        .groupby("movieId")["tag"]
        .apply(lambda values: sorted(set(values.str.lower()))[:12])
        .reset_index(name="tags")
    )

    matches = matches.merge(stats, on="movieId", how="left")
    matches = matches.merge(tag_text, on="movieId", how="left")

    movies = []
    for _, row in matches.iterrows():
        tags_value = row["tags"] if isinstance(row["tags"], list) else []
        movies.append({
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "genres": str(row["genres"]),
            "average_rating": round(float(row["average_rating"]), 2)
            if pd.notna(row["average_rating"]) else None,
            "rating_count": int(row["rating_count"]) if pd.notna(row["rating_count"]) else 0,
            "tags": tags_value,
        })

    return {"requested_titles": titles, "movies": movies}


def get_movie_summary(titles: list[str], limit: int = 3) -> dict[str, Any]:
    matches = _movie_matches(titles).head(limit)
    summaries = []
    for _, row in matches.iterrows():
        summaries.append({
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "genres": str(row["genres"]),
            "plot": str(row["plot"]),
        })
    return {"requested_titles": titles, "summaries": summaries}
