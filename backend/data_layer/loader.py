"""Cached CSV loader with parsing and normalization at the data boundary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .models import Link, Movie, Rating, Tag

REQUIRED_FILES = (
    "movies_with_plots.csv",
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
)


class DatasetError(ValueError):
    """Raised when a required file or its basic schema cannot be read."""


@dataclass(frozen=True, slots=True)
class MovieLensData:
    movies: tuple[Movie, ...]
    basic_movies: tuple[Movie, ...]
    ratings: tuple[Rating, ...]
    tags: tuple[Tag, ...]
    links: tuple[Link, ...]


class MovieLensDataLoader:
    """Loads the five supplied CSV files once per resolved dataset directory."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else self.default_data_dir()

    @staticmethod
    def default_data_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "ml-latest-small-filtered"

    def load(self) -> MovieLensData:
        return _load_dataset(self.data_dir.resolve())

    @staticmethod
    def clear_cache() -> None:
        """Clear process-local cache; useful in tests or after replacing files."""
        _load_dataset.cache_clear()


def _require_files(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        raise DatasetError(f"Missing required dataset file(s) in {data_dir}: {', '.join(missing)}")


def _read_rows(path: Path, fields: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            actual_fields = set(reader.fieldnames or [])
            missing = fields - actual_fields
            if missing:
                raise DatasetError(f"{path.name} is missing column(s): {', '.join(sorted(missing))}")
            return list(reader)
    except OSError as exc:
        raise DatasetError(f"Unable to read {path}: {exc}") from exc


def _integer(value: str | None, field: str, filename: str, row_number: int, *, nullable: bool = False) -> int | None:
    if value is None or not value.strip():
        if nullable:
            return None
        raise DatasetError(f"{filename}, row {row_number}: {field} is required")
    try:
        return int(float(value))
    except ValueError as exc:
        raise DatasetError(f"{filename}, row {row_number}: invalid {field}={value!r}") from exc


def _timestamp(value: str | None, filename: str, row_number: int) -> datetime:
    seconds = _integer(value, "timestamp", filename, row_number)
    assert seconds is not None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise DatasetError(f"{filename}, row {row_number}: invalid timestamp={value!r}") from exc


def _genres(value: str | None) -> tuple[str, ...]:
    if not value or value.strip() == "(no genres listed)":
        return ()
    return tuple(item.strip() for item in value.split("|") if item.strip())


@lru_cache(maxsize=8)
def _load_dataset(data_dir: Path) -> MovieLensData:
    _require_files(data_dir)
    movie_rows = _read_rows(data_dir / "movies_with_plots.csv", {"movieId", "title", "year", "genres", "plot"})
    basic_rows = _read_rows(data_dir / "movies.csv", {"movieId", "title", "genres"})
    rating_rows = _read_rows(data_dir / "ratings.csv", {"userId", "movieId", "rating", "timestamp"})
    tag_rows = _read_rows(data_dir / "tags.csv", {"userId", "movieId", "tag", "timestamp"})
    link_rows = _read_rows(data_dir / "links.csv", {"movieId", "imdbId", "tmdbId"})

    movies = tuple(
        Movie(
            movie_id=_integer(row["movieId"], "movieId", "movies_with_plots.csv", i),  # type: ignore[arg-type]
            title=row["title"].strip(),
            year=_integer(row["year"], "year", "movies_with_plots.csv", i, nullable=True),
            genres=_genres(row["genres"]),
            plot=row["plot"].strip() or None,
        )
        for i, row in enumerate(movie_rows, start=2)
    )
    basic_movies = tuple(
        Movie(_integer(row["movieId"], "movieId", "movies.csv", i), row["title"].strip(), None, _genres(row["genres"]), None)  # type: ignore[arg-type]
        for i, row in enumerate(basic_rows, start=2)
    )
    ratings = tuple(
        Rating(
            _integer(row["userId"], "userId", "ratings.csv", i),  # type: ignore[arg-type]
            _integer(row["movieId"], "movieId", "ratings.csv", i),  # type: ignore[arg-type]
            float(row["rating"]),
            _timestamp(row["timestamp"], "ratings.csv", i),
        )
        for i, row in enumerate(rating_rows, start=2)
    )
    tags = tuple(
        Tag(
            _integer(row["userId"], "userId", "tags.csv", i),  # type: ignore[arg-type]
            _integer(row["movieId"], "movieId", "tags.csv", i),  # type: ignore[arg-type]
            row["tag"].strip() or None,
            _timestamp(row["timestamp"], "tags.csv", i),
        )
        for i, row in enumerate(tag_rows, start=2)
    )
    links = tuple(
        Link(
            _integer(row["movieId"], "movieId", "links.csv", i),  # type: ignore[arg-type]
            _integer(row["imdbId"], "imdbId", "links.csv", i, nullable=True),
            _integer(row["tmdbId"], "tmdbId", "links.csv", i, nullable=True),
        )
        for i, row in enumerate(link_rows, start=2)
    )
    return MovieLensData(movies, basic_movies, ratings, tags, links)
