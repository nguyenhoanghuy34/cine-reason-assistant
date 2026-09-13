"""Local TF-IDF movie retrieval over repository-provided metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.data_layer.models import Movie
from backend.data_layer.repository import MovieRepository


@dataclass(frozen=True, slots=True)
class SearchResult:
    movie_id: int
    title: str
    score: float
    genres: tuple[str, ...]
    year: int | None
    plot: str | None


class MovieRetrievalService:
    """Content search and similarity using title, genre, plot, and tags only."""

    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository
        self._movies = repository.get_movies()
        self._movie_index = {movie.movie_id: index for index, movie in enumerate(self._movies)}
        self._known_genres = {genre.casefold().replace("-", " "): genre for movie in self._movies for genre in movie.genres}
        corpus = [self._document(movie) for movie in self._movies]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(corpus)

    def get_movie(self, movie_id: int) -> Movie | None:
        return self._repository.get_movie(movie_id)

    def search_movies(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not query.strip() or top_k <= 0:
            return []
        similar_title = re.fullmatch(r"\s*movies?\s+similar\s+to\s+(.+?)\s*", query, flags=re.IGNORECASE)
        if similar_title:
            movie = self._movie_from_title(similar_title.group(1))
            return self.find_similar_movies(movie.movie_id, top_k) if movie else []
        query_vector = self._vectorizer.transform([self._expand_query(query)])
        if query_vector.nnz == 0:
            return []
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        results = self._rank(
            scores,
            top_k,
            exclude_index=None,
            decade=_decade_from_query(query),
            required_genre=self._genre_from_query(query),
        )
        title_match = self._movie_from_title(query)
        if title_match is not None:
            exact = SearchResult(title_match.movie_id, title_match.title, 1.0, title_match.genres, title_match.year, title_match.plot)
            results = [exact] + [result for result in results if result.movie_id != title_match.movie_id]
        return results[:top_k]

    def find_similar_movies(self, movie_id: int, top_k: int = 10) -> list[SearchResult]:
        index = self._movie_index.get(movie_id)
        if index is None or top_k <= 0:
            return []
        scores = cosine_similarity(self._matrix[index], self._matrix).ravel()
        return self._rank(scores, top_k, exclude_index=index, decade=None, required_genre=None)

    def _document(self, movie: Movie) -> str:
        tags = self._repository.get_movie_tags(movie.movie_id) or ()
        tag_text = " ".join(tag.tag for tag in tags if tag.tag)
        genres = " ".join(genre.replace("-", " ") for genre in movie.genres)
        # Repeating concise metadata makes it useful alongside much longer plots.
        return " ".join((movie.title, movie.title, genres, genres, tag_text, movie.plot or ""))

    def _rank(self, scores, top_k: int, exclude_index: int | None, decade: int | None, required_genre: str | None) -> list[SearchResult]:
        ranked = sorted(range(len(self._movies)), key=lambda index: (-scores[index], self._movies[index].title))
        results = []
        for index in ranked:
            movie = self._movies[index]
            if index == exclude_index or scores[index] <= 0:
                continue
            if decade is not None and (movie.year is None or movie.year // 10 * 10 != decade):
                continue
            if required_genre is not None and required_genre not in movie.genres:
                continue
            results.append(SearchResult(movie.movie_id, movie.title, round(float(scores[index]), 4), movie.genres, movie.year, movie.plot))
            if len(results) == top_k:
                break
        return results

    def _movie_from_title(self, title: str) -> Movie | None:
        exact = self._repository.get_movie_by_title(title)
        if exact:
            return exact
        normalized = title.casefold().strip()
        article_match = re.fullmatch(r"(the|a|an)\s+(.+)", normalized)
        alternatives = [normalized]
        if article_match:
            alternatives.append(f"{article_match.group(2)}, {article_match.group(1)}")
        return next((movie for movie in self._movies if any(candidate == movie.title.casefold() or candidate in movie.title.casefold() for candidate in alternatives)), None)

    @staticmethod
    def _expand_query(query: str) -> str:
        normalized = query.casefold().replace("science fiction", "sci fi")
        normalized = re.sub(r"\bfunny\b", "funny comedy humorous", normalized)
        return normalized

    def _genre_from_query(self, query: str) -> str | None:
        normalized = self._expand_query(query)
        # Use genres found in the supplied data, longest first to avoid partial matches.
        for text, genre in sorted(self._known_genres.items(), key=lambda item: -len(item[0])):
            if re.search(rf"\b{re.escape(text)}\b", normalized):
                return genre
        return None


def _decade_from_query(query: str) -> int | None:
    match = re.search(r"\b(19\d0|20\d0)s\b", query)
    return int(match.group(1)) if match else None
