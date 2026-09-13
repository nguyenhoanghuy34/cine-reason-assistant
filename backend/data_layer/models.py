"""Typed, immutable representations of normalized MovieLens rows."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    plot: str | None


@dataclass(frozen=True, slots=True)
class Rating:
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Tag:
    user_id: int
    movie_id: int
    tag: str | None
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Link:
    movie_id: int
    imdb_id: int | None
    tmdb_id: int | None
