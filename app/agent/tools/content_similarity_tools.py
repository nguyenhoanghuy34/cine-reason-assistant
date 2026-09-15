from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.agent.tools.movie_data_tools import resolve_movie_titles


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
MOVIES_WITH_PLOTS_PATH = DATA_DIR / "movies_with_plots.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def _movie_text(row) -> str:
    return " ".join([
        str(row.get("title", "")),
        str(row.get("genres", "")),
        str(row.get("plot", "")),
    ])


def find_content_matches(
    query: str,
    limit: int = 5,
    exclude_movie_ids: set[int] | None = None,
) -> dict[str, Any]:
    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    exclude_movie_ids = exclude_movie_ids or set()
    pool = movies[~movies["movieId"].astype(int).isin(exclude_movie_ids)].copy()
    if pool.empty:
        return {"query": query, "matches": []}

    corpus = [_movie_text(row) for _, row in pool.iterrows()]
    vectors = TfidfVectorizer(stop_words="english", max_features=8000).fit_transform([query, *corpus])
    scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
    pool["_content_score"] = scores
    pool = pool.sort_values("_content_score", ascending=False).head(limit)

    ratings = pd.read_csv(RATINGS_PATH)
    stats = (
        ratings.groupby("movieId")
        .agg(average_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    pool = pool.merge(stats, on="movieId", how="left")

    matches = []
    for _, row in pool.iterrows():
        matches.append({
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": str(row["genres"]),
            "plot": str(row["plot"]),
            "content_score": round(float(row["_content_score"]), 3),
            "average_rating": round(float(row["average_rating"]), 2)
            if pd.notna(row["average_rating"]) else None,
            "rating_count": int(row["rating_count"]) if pd.notna(row["rating_count"]) else 0,
        })

    return {"query": query, "matches": matches}


def compare_movie_themes(titles: list[str]) -> dict[str, Any]:
    resolved_titles = []
    for title in titles:
        resolved_titles.extend(resolve_movie_titles(title, [title]))
    resolved_titles = list(dict.fromkeys(resolved_titles))

    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    selected = movies[movies["title"].isin(resolved_titles)].copy()
    if len(selected) < 2:
        return {"requested_titles": titles, "resolved_titles": resolved_titles, "comparisons": []}

    corpus = [_movie_text(row) for _, row in selected.iterrows()]
    vectors = TfidfVectorizer(stop_words="english", max_features=8000).fit_transform(corpus)
    scores = cosine_similarity(vectors)
    rows = selected.to_dict("records")
    comparisons = []

    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            left = rows[left_index]
            right = rows[right_index]
            comparisons.append({
                "left_title": str(left["title"]),
                "right_title": str(right["title"]),
                "left_genres": str(left["genres"]),
                "right_genres": str(right["genres"]),
                "similarity_score": round(float(scores[left_index][right_index]), 3),
                "left_plot": str(left["plot"]),
                "right_plot": str(right["plot"]),
            })

    return {
        "requested_titles": titles,
        "resolved_titles": resolved_titles,
        "comparisons": comparisons,
    }


def find_similar_to_movie(
    source_title: str,
    limit: int = 5,
    excluded_genres: list[str] | None = None,
) -> dict[str, Any]:
    resolved = resolve_movie_titles(source_title, [source_title])
    if not resolved:
        return {"source_title": source_title, "resolved_title": None, "matches": []}

    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    source = movies[movies["title"] == resolved[0]]
    if source.empty:
        return {"source_title": source_title, "resolved_title": resolved[0], "matches": []}

    query = _movie_text(source.iloc[0])
    exclude_ids = {int(source.iloc[0]["movieId"])}
    result = find_content_matches(query, limit=limit * 3, exclude_movie_ids=exclude_ids)
    excluded = {genre.lower() for genre in excluded_genres or []}
    matches = [
        item for item in result["matches"]
        if not excluded or all(genre not in item["genres"].lower() for genre in excluded)
    ][:limit]
    return {
        "source_title": source_title,
        "resolved_title": resolved[0],
        "matches": matches,
    }
