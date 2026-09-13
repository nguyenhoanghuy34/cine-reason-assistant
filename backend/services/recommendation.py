"""Deterministic hybrid recommendations with component-level evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p

from backend.data_layer.models import Movie
from backend.data_layer.repository import MovieRepository
from .collaborative_filtering import CollaborativeFilteringService, SimilarUser
from .movie_retrieval import MovieRetrievalService
from .user_profile import UserProfileService


@dataclass(frozen=True, slots=True)
class RecommendationEvidence:
    collaborative_score: float
    content_score: float
    genre_score: float
    quality_score: float
    supporting_movies: tuple[str, ...]
    similar_user_rating: float | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    movie_id: int
    title: str
    final_score: float
    genres: tuple[str, ...]
    year: int | None
    evidence: RecommendationEvidence


class HybridRecommendationEngine:
    """Ranks unseen movies from collaborative, content, taste, and quality signals."""

    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository
        self._profiles = UserProfileService(repository)
        self._retrieval = MovieRetrievalService(repository)
        self._collaborative = CollaborativeFilteringService(repository)
        self._movies = repository.get_movies()
        self._quality = self._build_quality_scores()

    def recommend_movies(
        self,
        user_id: int,
        query: str | None = None,
        genre: str | None = None,
        exclude_genres: tuple[str, ...] | list[str] | None = None,
        top_k: int = 10,
    ) -> list[Recommendation]:
        profile = self._profiles.get_user_profile(user_id)
        if profile is None or top_k <= 0:
            return []
        seen = {rating.movie_id for rating in self._repository.get_user_ratings(user_id) or ()}
        excluded = {item.casefold() for item in exclude_genres or ()}
        required_genre = genre.casefold() if genre else None
        content_scores, supporting_movies, query_candidates = self._content_signal(profile, query)
        neighbor_ratings = self._neighbor_movie_ratings(user_id)
        genre_scores = {item.genre: item.preference_score for item in profile.genre_statistics}
        recommendations: list[Recommendation] = []
        for movie in self._movies:
            if movie.movie_id in seen or (required_genre and not any(value.casefold() == required_genre for value in movie.genres)):
                continue
            if any(value.casefold() in excluded for value in movie.genres):
                continue
            if query is not None and movie.movie_id not in query_candidates:
                continue
            collaborative_score, neighbor_rating = self._collaborative_score(movie.movie_id, neighbor_ratings)
            content_score = content_scores.get(movie.movie_id, 0.0)
            genre_score = sum(genre_scores.get(value, 0.0) for value in movie.genres) / len(movie.genres) if movie.genres else 0.0
            quality_score = self._quality.get(movie.movie_id, 0.0)
            final_score = 0.40 * collaborative_score + 0.30 * content_score + 0.20 * genre_score + 0.10 * quality_score
            recommendations.append(Recommendation(
                movie.movie_id, movie.title, round(final_score, 4), movie.genres, movie.year,
                RecommendationEvidence(round(collaborative_score, 4), round(content_score, 4), round(genre_score, 4), round(quality_score, 4), supporting_movies.get(movie.movie_id, ()), neighbor_rating),
            ))
        return sorted(recommendations, key=lambda item: (-item.final_score, item.title))[:top_k]

    def _content_signal(self, profile, query: str | None) -> tuple[dict[int, float], dict[int, tuple[str, ...]], set[int]]:
        scores: dict[int, float] = {}
        support: dict[int, tuple[str, ...]] = {}
        if query:
            results = self._retrieval.search_movies(query, top_k=500)
            maximum = max((result.score for result in results), default=1.0)
            for result in results:
                scores[result.movie_id] = result.score / maximum
            return scores, support, set(scores)
        # A bounded set of strongest positive examples keeps online recommendation
        # latency predictable while retaining human-readable supporting evidence.
        seeds = sorted(
            profile.highly_rated_movies,
            key=lambda movie: (-self._rating_for(profile.user_id, movie.movie_id), movie.title),
        )[:10]
        for movie in seeds:
            for result in self._retrieval.find_similar_movies(movie.movie_id, top_k=100):
                weighted = result.score * (self._rating_for(profile.user_id, movie.movie_id) / 5.0)
                if weighted > scores.get(result.movie_id, 0.0):
                    scores[result.movie_id] = weighted
                    support[result.movie_id] = (movie.title,)
        return scores, support, {movie.movie_id for movie in self._movies}

    def _rating_for(self, user_id: int, movie_id: int) -> float:
        return next(rating.rating for rating in self._repository.get_user_ratings(user_id) or () if rating.movie_id == movie_id)

    def _neighbor_movie_ratings(self, user_id: int) -> dict[int, list[tuple[float, float]]]:
        grouped: dict[int, list[tuple[float, float]]] = {}
        for neighbor in self._collaborative.find_similar_users(user_id, top_k=50):
            for rating in self._repository.get_user_ratings(neighbor.user_id) or ():
                grouped.setdefault(rating.movie_id, []).append((neighbor.similarity, rating.rating))
        return grouped

    @staticmethod
    def _collaborative_score(movie_id: int, ratings: dict[int, list[tuple[float, float]]]) -> tuple[float, float | None]:
        values = ratings.get(movie_id, [])
        if not values:
            return 0.0, None
        weight = sum(similarity for similarity, _ in values)
        predicted = sum(similarity * rating for similarity, rating in values) / weight
        # Shrink sparse neighbor evidence toward zero rather than letting one
        # neighbor's 5/5 become a fully confident collaborative score.
        confidence = weight / (weight + 0.5)
        return (predicted / 5.0) * confidence, round(predicted, 3)

    def _build_quality_scores(self) -> dict[int, float]:
        statistics = {}
        max_count = 1
        for movie in self._movies:
            ratings = self._repository.get_movie_ratings(movie.movie_id) or ()
            if ratings:
                statistics[movie.movie_id] = (sum(item.rating for item in ratings) / len(ratings), len(ratings))
                max_count = max(max_count, len(ratings))
        return {
            movie_id: 0.7 * (average / 5.0) + 0.3 * (log1p(count) / log1p(max_count))
            for movie_id, (average, count) in statistics.items()
        }
