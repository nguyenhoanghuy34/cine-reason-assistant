"""Read-only indexed access to normalized MovieLens data."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .loader import MovieLensData
from .models import Movie, Rating, Tag


class MovieRepository:
    """Repository boundary for upper layers; it never reads CSV files itself."""

    def __init__(self, data: MovieLensData) -> None:
        self._movies = {movie.movie_id: movie for movie in data.movies}
        self._ratings_by_user: dict[int, tuple[Rating, ...]] = _group(data.ratings, "user_id")
        self._ratings_by_movie: dict[int, tuple[Rating, ...]] = _group(data.ratings, "movie_id")
        self._tags_by_user: dict[int, tuple[Tag, ...]] = _group(data.tags, "user_id")
        self._tags_by_movie: dict[int, tuple[Tag, ...]] = _group(data.tags, "movie_id")

    def get_movie(self, movie_id: int) -> Movie | None:
        return self._movies.get(movie_id)

    def get_movie_by_title(self, title: str) -> Movie | None:
        normalized = title.casefold().strip()
        return next((movie for movie in self._movies.values() if movie.title.casefold() == normalized), None)

    def get_movies(self, *, genre: str | None = None, year: int | None = None, limit: int | None = None) -> list[Movie]:
        movies = self._movies.values()
        genre_key = genre.casefold().strip() if genre else None
        result = [movie for movie in movies if (year is None or movie.year == year) and (genre_key is None or any(item.casefold() == genre_key for item in movie.genres))]
        return result[:limit] if limit is not None else result

    def get_user_ratings(self, user_id: int) -> tuple[Rating, ...] | None:
        return self._ratings_by_user.get(user_id)

    def get_user_tags(self, user_id: int) -> tuple[Tag, ...] | None:
        return self._tags_by_user.get(user_id)

    def get_movie_ratings(self, movie_id: int) -> tuple[Rating, ...] | None:
        return self._ratings_by_movie.get(movie_id)

    def get_movie_tags(self, movie_id: int) -> tuple[Tag, ...] | None:
        return self._tags_by_movie.get(movie_id)


def _group(items: Iterable[Rating | Tag], attribute: str) -> dict[int, tuple]:
    grouped: defaultdict[int, list] = defaultdict(list)
    for item in items:
        grouped[getattr(item, attribute)].append(item)
    return {key: tuple(values) for key, values in grouped.items()}
