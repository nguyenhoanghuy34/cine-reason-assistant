"""Non-destructive integrity checks and dataset summary metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .loader import MovieLensData


@dataclass(frozen=True, slots=True)
class ValidationReport:
    metrics: dict[str, int | dict[float, int]]
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_data(data: MovieLensData) -> ValidationReport:
    """Inspect data only. No records are removed, changed, or deduplicated."""
    issues: list[str] = []
    movie_ids = [movie.movie_id for movie in data.movies]
    movie_id_set = set(movie_ids)
    basic_movie_ids = [movie.movie_id for movie in data.basic_movies]
    basic_movie_id_set = set(basic_movie_ids)
    _issue_if(issues, len(movie_ids) != len(movie_id_set), "movies_with_plots.csv has duplicate movieId values")
    _issue_if(issues, len(basic_movie_ids) != len(basic_movie_id_set), "movies.csv has duplicate movieId values")
    _issue_if(issues, any(movie.movie_id <= 0 for movie in data.movies), "movies_with_plots.csv contains invalid movieId values")
    _issue_if(issues, any(not movie.title for movie in data.movies), "movies_with_plots.csv contains missing titles")
    _issue_if(issues, any(not movie.title for movie in data.basic_movies), "movies.csv contains missing titles")
    _issue_if(issues, any(r.user_id <= 0 for r in data.ratings) or any(t.user_id <= 0 for t in data.tags), "ratings.csv or tags.csv contains invalid userId values")
    _issue_if(issues, any(r.rating < 0.5 or r.rating > 5.0 for r in data.ratings), "ratings.csv has ratings outside 0.5-5.0")
    _issue_if(issues, any(t.tag is None for t in data.tags), "tags.csv contains missing tag values")
    _issue_if(issues, any(link.imdb_id is None and link.tmdb_id is None for link in data.links), "links.csv contains rows with both external IDs missing")

    for name, rows in (("ratings.csv", data.ratings), ("tags.csv", data.tags), ("links.csv", data.links)):
        duplicate_count = len(rows) - len(set(rows))
        _issue_if(issues, duplicate_count > 0, f"{name} has {duplicate_count} exact duplicate record(s)")

    rating_movie_ids = {rating.movie_id for rating in data.ratings}
    tag_movie_ids = {tag.movie_id for tag in data.tags}
    link_movie_ids = {link.movie_id for link in data.links}
    for name, ids in (("ratings.csv", rating_movie_ids), ("tags.csv", tag_movie_ids), ("links.csv", link_movie_ids)):
        unknown = ids - movie_id_set
        _issue_if(issues, bool(unknown), f"{name} references {len(unknown)} movieId(s) absent from movies_with_plots.csv")

    # movies.csv is the unfiltered MovieLens catalogue, so records exclusive to it
    # are reported as coverage rather than treated as corrupt data.
    missing_from_basic = movie_id_set - basic_movie_id_set
    _issue_if(issues, bool(missing_from_basic), f"movies_with_plots.csv has {len(missing_from_basic)} movieId(s) absent from movies.csv")

    distribution = dict(sorted(Counter(rating.rating for rating in data.ratings).items()))
    metrics: dict[str, int | dict[float, int]] = {
        "movies": len(data.movies), "users": len({r.user_id for r in data.ratings}),
        "basic_movies": len(data.basic_movies),
        "ratings": len(data.ratings), "tags": len(data.tags), "rating_distribution": distribution,
        "movies_without_ratings": len(movie_id_set - rating_movie_ids),
        "movies_without_tags": len(movie_id_set - tag_movie_ids),
        "movies_without_links": len(movie_id_set - link_movie_ids),
        "basic_movies_outside_plot_catalogue": len(basic_movie_id_set - movie_id_set),
        "missing_plots": sum(movie.plot is None for movie in data.movies),
        "missing_tag_values": sum(tag.tag is None for tag in data.tags),
        "links_missing_imdb_id": sum(link.imdb_id is None for link in data.links),
        "links_missing_tmdb_id": sum(link.tmdb_id is None for link in data.links),
    }
    return ValidationReport(metrics, tuple(issues))


def _issue_if(issues: list[str], condition: bool, message: str) -> None:
    if condition:
        issues.append(message)
