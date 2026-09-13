"""Small LangChain tool adapters around deterministic backend services."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain_core.tools import BaseTool, tool

from backend.data_layer.repository import MovieRepository
from backend.services.collaborative_filtering import CollaborativeFilteringService
from backend.services.movie_retrieval import MovieRetrievalService
from backend.services.recommendation import HybridRecommendationEngine
from backend.services.user_profile import UserProfileService


class MovieToolbox:
    """Expose bounded, JSON-serializable facts; tools never read CSV files."""

    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository
        self._retrieval = MovieRetrievalService(repository)
        self._profiles = UserProfileService(repository)
        self._collaborative = CollaborativeFilteringService(repository)
        self._recommendations = HybridRecommendationEngine(repository)
        self.tools: list[BaseTool] = self._build_tools()
        self.by_name = {item.name: item for item in self.tools}

    def _build_tools(self) -> list[BaseTool]:
        @tool
        def get_movie(movie_id: int) -> dict[str, Any]:
            """Get verified metadata and a bounded plot for a MovieLens movie ID."""
            movie = self._retrieval.get_movie(movie_id)
            return {"ok": True, "movie": _movie(movie)} if movie else _error(f"Movie {movie_id} was not found.")

        @tool
        def search_movies(query: str, top_k: int = 5) -> dict[str, Any]:
            """Search movie title, genres, plot, and tags with deterministic TF-IDF retrieval."""
            if top_k < 1:
                return _error("top_k must be at least 1.")
            return {"ok": True, "results": [_search(item) for item in self._retrieval.search_movies(query, min(top_k, 10))]}

        @tool
        def find_similar_movies(movie_id: int, top_k: int = 5) -> dict[str, Any]:
            """Find content-similar movies for an existing movie ID."""
            if self._retrieval.get_movie(movie_id) is None:
                return _error(f"Movie {movie_id} was not found.")
            return {"ok": True, "results": [_search(item) for item in self._retrieval.find_similar_movies(movie_id, min(max(top_k, 1), 10))]}

        @tool
        def get_user_profile(user_id: int) -> dict[str, Any]:
            """Get a compact, rating-derived profile for a known MovieLens user."""
            profile = self._profiles.get_user_profile(user_id)
            return {"ok": True, "profile": _profile(profile, self._all_genres())} if profile else _error(f"User {user_id} has no rating history in this dataset.")

        @tool
        def get_user_ratings(user_id: int, limit: int = 10) -> dict[str, Any]:
            """Get a bounded list of a user's ratings; use the profile for aggregate taste."""
            ratings = self._profiles.get_user_ratings(user_id)
            if ratings is None:
                return _error(f"User {user_id} has no rating history in this dataset.")
            rows = []
            for rating in ratings[: min(max(limit, 1), 25)]:
                movie = self._retrieval.get_movie(rating.movie_id)
                rows.append({"movie_id": rating.movie_id, "title": movie.title if movie else None, "rating": rating.rating})
            return {"ok": True, "rating_count": len(ratings), "ratings": rows}

        @tool
        def analyze_user_taste(user_id: int) -> dict[str, Any]:
            """Get favorite and weak genres derived from the user's ratings."""
            profile = self._profiles.analyze_user_taste(user_id)
            return {"ok": True, "profile": _profile(profile, self._all_genres())} if profile else _error(f"User {user_id} has no rating history in this dataset.")

        @tool
        def find_similar_users(user_id: int, top_k: int = 10) -> dict[str, Any]:
            """Find rating-taste neighbors, including overlap count for confidence."""
            if self._profiles.get_user_ratings(user_id) is None:
                return _error(f"User {user_id} has no rating history in this dataset.")
            return {"ok": True, "results": [asdict(item) for item in self._collaborative.find_similar_users(user_id, min(max(top_k, 1), 30))]}

        @tool
        def get_similar_users_opinion(user_id: int, movie_id: int) -> dict[str, Any]:
            """Aggregate how similar users rated one movie, including count and distribution."""
            opinion = self._collaborative.get_similar_users_opinion(user_id, movie_id)
            return {"ok": True, "opinion": asdict(opinion)} if opinion else _error("User or movie was not found in the dataset.")

        @tool
        def recommend_movies(user_id: int, query: str | None = None, genre: str | None = None, exclude_genres: list[str] | None = None, top_k: int = 5) -> dict[str, Any]:
            """Return unseen hybrid recommendations and numeric evidence for a known user."""
            results = self._recommendations.recommend_movies(user_id, query, genre, exclude_genres, min(max(top_k, 1), 10))
            if not results and self._profiles.get_user_ratings(user_id) is None:
                return _error(f"User {user_id} has no rating history in this dataset.")
            return {"ok": True, "recommendations": [_recommendation(item) for item in results]}

        return [get_movie, search_movies, find_similar_movies, get_user_profile, get_user_ratings, analyze_user_taste, find_similar_users, get_similar_users_opinion, recommend_movies]

    def _all_genres(self) -> list[str]:
        return sorted({genre for movie in self._repository.get_movies() for genre in movie.genres})


def _movie(movie) -> dict[str, Any]:
    return {"movie_id": movie.movie_id, "title": movie.title, "year": movie.year, "genres": list(movie.genres), "plot": (movie.plot or "")[:600]}


def _search(item) -> dict[str, Any]:
    return {"movie_id": item.movie_id, "title": item.title, "score": item.score, "genres": list(item.genres), "year": item.year, "plot": (item.plot or "")[:300]}


def _profile(profile, all_genres: list[str]) -> dict[str, Any]:
    observed = {item.genre for item in profile.genre_statistics}
    return {
        "user_id": profile.user_id, "rating_count": profile.rating_count, "average_rating": profile.average_rating,
        "favorite_genres": [asdict(item) for item in profile.favorite_genres],
        "weak_genres": [asdict(item) for item in profile.weak_genres],
        "unrated_genres": [genre for genre in all_genres if genre not in observed],
        "highly_rated_movies": [_movie(item) for item in profile.highly_rated_movies[:5]],
    }


def _recommendation(item) -> dict[str, Any]:
    return {"movie_id": item.movie_id, "title": item.title, "final_score": item.final_score, "genres": list(item.genres), "year": item.year, "evidence": asdict(item.evidence)}


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}
