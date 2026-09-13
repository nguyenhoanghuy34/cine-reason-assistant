import unittest

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.movie_retrieval import MovieRetrievalService


class MovieRetrievalServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = MovieRetrievalService(MovieRepository(MovieLensDataLoader().load()))

    def test_movie_search_returns_structured_results(self) -> None:
        results = self.service.search_movies("science fiction movies", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0].movie_id, int)
        self.assertGreater(results[0].score, 0)

    def test_similar_movie_search(self) -> None:
        results = self.service.find_similar_movies(1, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(result.movie_id != 1 for result in results))

    def test_explicit_genre_is_respected(self) -> None:
        results = self.service.search_movies("action movies from the 1990s", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(all("Action" in result.genres and 1990 <= result.year <= 1999 for result in results))

    def test_query_with_no_match(self) -> None:
        self.assertEqual(self.service.search_movies("qzxvnonexistentword", top_k=3), [])

    def test_missing_movie(self) -> None:
        self.assertIsNone(self.service.get_movie(-1))
        self.assertEqual(self.service.find_similar_movies(-1), [])
