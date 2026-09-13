"""User-rating analysis without recommendation or LLM logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from backend.data_layer.models import Movie, Rating
from backend.data_layer.repository import MovieRepository

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 2.0


@dataclass(frozen=True, slots=True)
class GenrePreference:
    genre: str
    count: int
    average_rating: float
    high_rating_ratio: float
    preference_score: float


@dataclass(frozen=True, slots=True)
class UserTasteProfile:
    user_id: int
    rating_count: int
    average_rating: float
    highly_rated_movies: tuple[Movie, ...]
    low_rated_movies: tuple[Movie, ...]
    genre_statistics: tuple[GenrePreference, ...]
    favorite_genres: tuple[GenrePreference, ...]
    weak_genres: tuple[GenrePreference, ...]


class UserProfileService:
    """Creates explainable profiles from a user's historical ratings."""

    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository

    def get_user_ratings(self, user_id: int) -> tuple[Rating, ...] | None:
        return self._repository.get_user_ratings(user_id)

    def get_user_profile(self, user_id: int) -> UserTasteProfile | None:
        ratings = self.get_user_ratings(user_id)
        if ratings is None:
            return None
        statistics = self.analyze_genre_preferences(user_id)
        rated_movies = [(rating, self._repository.get_movie(rating.movie_id)) for rating in ratings]
        highly_rated = tuple(movie for rating, movie in rated_movies if movie is not None and rating.rating >= HIGH_RATING_THRESHOLD)
        low_rated = tuple(movie for rating, movie in rated_movies if movie is not None and rating.rating <= LOW_RATING_THRESHOLD)
        favorites = tuple(sorted(statistics, key=lambda item: (-item.preference_score, -item.average_rating, -item.count, item.genre))[:3])
        weak = tuple(sorted(statistics, key=lambda item: (item.preference_score, item.average_rating, -item.count, item.genre))[:3])
        return UserTasteProfile(
            user_id=user_id,
            rating_count=len(ratings),
            average_rating=round(sum(r.rating for r in ratings) / len(ratings), 3),
            highly_rated_movies=highly_rated,
            low_rated_movies=low_rated,
            genre_statistics=statistics,
            favorite_genres=favorites,
            weak_genres=weak,
        )

    def analyze_user_taste(self, user_id: int) -> UserTasteProfile | None:
        """Alias kept as an explicit capability for future tool exposure."""
        return self.get_user_profile(user_id)

    def analyze_genre_preferences(self, user_id: int) -> tuple[GenrePreference, ...]:
        ratings = self.get_user_ratings(user_id)
        if ratings is None:
            return ()
        ratings_by_genre: defaultdict[str, list[float]] = defaultdict(list)
        for rating in ratings:
            movie = self._repository.get_movie(rating.movie_id)
            if movie is not None:
                for genre in movie.genres:
                    ratings_by_genre[genre].append(rating.rating)
        preferences = []
        for genre, values in ratings_by_genre.items():
            average = sum(values) / len(values)
            high_ratio = sum(value >= HIGH_RATING_THRESHOLD for value in values) / len(values)
            # Rating quality is primary; high-rating consistency is a secondary signal.
            # Count remains separately visible and cannot make a genre favorite by itself.
            score = (average / 5.0) * (0.5 + 0.5 * high_ratio)
            preferences.append(GenrePreference(genre, len(values), round(average, 3), round(high_ratio, 3), round(score, 3)))
        return tuple(sorted(preferences, key=lambda item: item.genre))
