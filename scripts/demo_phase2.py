#!/usr/bin/env python3
"""Small deterministic demonstration of the Phase 2 services."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.movie_retrieval import MovieRetrievalService
from backend.services.user_profile import UserProfileService


def main() -> None:
    repository = MovieRepository(MovieLensDataLoader().load())
    profile = UserProfileService(repository).get_user_profile(1)
    assert profile is not None
    print(f"User 1: {profile.rating_count} ratings; average={profile.average_rating}")
    print("Favorite genres:", [(item.genre, item.count, item.average_rating, item.preference_score) for item in profile.favorite_genres])
    print("Weak genres:", [(item.genre, item.count, item.average_rating, item.preference_score) for item in profile.weak_genres])
    retrieval = MovieRetrievalService(repository)
    for query in ("dark psychological thriller with a twist", "science fiction movies", "action movies from the 1990s"):
        print(f"\n{query!r}")
        for result in retrieval.search_movies(query, top_k=3):
            print(f"- {result.title} ({result.year}), score={result.score}, genres={list(result.genres)}")


if __name__ == "__main__":
    main()
