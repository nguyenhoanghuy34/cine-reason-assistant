from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"
MOVIES_WITH_PLOTS_PATH = DATA_DIR / "movies_with_plots.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
TAGS_PATH = DATA_DIR / "tags.csv"
FUZZY_TITLE_THRESHOLD = 0.82


def _normalize_title(value: str) -> str:
    text = re.sub(r"\(\d{4}\)", "", str(value)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_variants(value: str) -> set[str]:
    normalized = _normalize_title(value)
    variants = {normalized} if normalized else set()
    aka_matches = re.findall(r"a k a ([^)]+)", normalized)
    variants.update(_normalize_title(match) for match in aka_matches if match.strip())
    for article in ("the", "a", "an"):
        suffix = f" {article}"
        if normalized.endswith(suffix):
            variants.add(f"{article} {normalized[:-len(suffix)]}".strip())
    return {variant for variant in variants if variant}


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _best_fuzzy_title(
    requested: str,
    title_rows: list[tuple[str, set[str]]],
) -> tuple[str, float] | None:
    requested_variants = _title_variants(requested)
    best_title = ""
    best_score = 0.0

    for title, variants in title_rows:
        for requested_variant in requested_variants:
            for variant in variants:
                score = _similarity(requested_variant, variant)
                if score > best_score:
                    best_title = title
                    best_score = score

    if best_title and best_score >= FUZZY_TITLE_THRESHOLD:
        return best_title, best_score
    return None


def resolve_movie_titles(query: str, candidate_titles: list[str] | None = None) -> list[str]:
    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    if movies.empty or "title" not in movies:
        return []

    query_text = _normalize_title(query)
    candidates = {
        variant
        for title in candidate_titles or []
        if str(title).strip()
        for variant in _title_variants(title)
    }

    title_rows = [
        (str(row["title"]), _title_variants(str(row["title"])))
        for _, row in movies[["title"]].dropna().iterrows()
    ]
    resolved = []

    for title, variants in sorted(title_rows, key=lambda item: max(map(len, item[1]), default=0), reverse=True):
        if not variants or title in resolved:
            continue
        exact_candidate = bool(variants & candidates)
        mentioned_in_query = bool(
            query_text
            and any(
                len(variant) >= 3
                and re.search(rf"\b{re.escape(variant)}\b", query_text)
                for variant in variants
            )
        )
        if exact_candidate or mentioned_in_query:
            resolved.append(title)

    if not resolved:
        for candidate in candidate_titles or []:
            fuzzy_match = _best_fuzzy_title(candidate, title_rows)
            if fuzzy_match and fuzzy_match[0] not in resolved:
                resolved.append(fuzzy_match[0])

    return resolved


def get_title_resolution_evidence(
    query: str,
    candidate_titles: list[str] | None = None,
) -> dict[str, Any]:
    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    if movies.empty or "title" not in movies:
        return {"requested_titles": candidate_titles or [], "resolved_titles": []}

    title_rows = [
        (str(row["title"]), _title_variants(str(row["title"])))
        for _, row in movies[["title"]].dropna().iterrows()
    ]
    resolved = resolve_movie_titles(query, candidate_titles)
    requested = candidate_titles or []
    matches = []

    for title in resolved:
        title_variants = dict(title_rows).get(title, _title_variants(title))
        exact = any(
            bool(title_variants & _title_variants(candidate))
            for candidate in requested
        )
        if exact:
            matches.append({
                "requested": requested,
                "resolved_title": title,
                "match_type": "exact",
                "confidence": 1.0,
            })
            continue

        best_requested = ""
        best_score = 0.0
        for candidate in requested:
            for candidate_variant in _title_variants(candidate):
                for title_variant in title_variants:
                    score = _similarity(candidate_variant, title_variant)
                    if score > best_score:
                        best_requested = candidate
                        best_score = score
        matches.append({
            "requested": best_requested or requested,
            "resolved_title": title,
            "match_type": "fuzzy",
            "confidence": round(best_score, 2),
        })

    return {
        "requested_titles": requested,
        "resolved_titles": resolved,
        "matches": matches,
    }


def _movie_matches(titles: list[str]) -> pd.DataFrame:
    movies = pd.read_csv(MOVIES_WITH_PLOTS_PATH)
    if not titles:
        return movies.head(0)

    requested = {
        variant
        for title in titles
        if title
        for variant in _title_variants(title)
    }
    masks = [
        movies["title"].map(lambda title: bool(_title_variants(title) & requested))
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
