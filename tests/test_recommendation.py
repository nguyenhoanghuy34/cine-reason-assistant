import unittest

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.recommendation import HybridRecommendationEngine


class HybridRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = MovieRepository(MovieLensDataLoader().load())
        cls.engine = HybridRecommendationEngine(cls.repository)

    def test_seen_movies_are_excluded(self) -> None:
        seen = {rating.movie_id for rating in self.repository.get_user_ratings(1) or ()}
        results = self.engine.recommend_movies(1, top_k=10)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(item.movie_id not in seen for item in results))

    def test_genre_filter(self) -> None:
        results = self.engine.recommend_movies(1, genre="Comedy", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertTrue(all("Comedy" in item.genres for item in results))

    def test_excluded_genre(self) -> None:
        results = self.engine.recommend_movies(1, exclude_genres=["Horror"], top_k=10)
        self.assertTrue(all("Horror" not in item.genres for item in results))

    def test_query_with_personalization(self) -> None:
        results = self.engine.recommend_movies(1, query="science fiction movies", top_k=5)
        self.assertGreater(len(results), 0)
        self.assertTrue(all("Sci-Fi" in item.genres for item in results))
        self.assertTrue(all(item.evidence.content_score > 0 for item in results))

    def test_missing_user(self) -> None:
        self.assertEqual(self.engine.recommend_movies(-1), [])
