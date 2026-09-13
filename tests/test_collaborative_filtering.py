import unittest

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.collaborative_filtering import CollaborativeFilteringService


class CollaborativeFilteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = MovieRepository(MovieLensDataLoader().load())
        cls.service = CollaborativeFilteringService(cls.repository)

    def test_similar_users_for_user_with_many_ratings(self) -> None:
        results = self.service.find_similar_users(1, top_k=5)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(item.user_id != 1 and item.similarity > 0 and item.overlap_count >= 2 for item in results))

    def test_similar_users_for_user_with_few_ratings(self) -> None:
        few_ratings_user = min(self.repository.get_user_ids(), key=lambda user_id: len(self.repository.get_user_ratings(user_id) or ()))
        results = self.service.find_similar_users(few_ratings_user, top_k=5)
        self.assertIsInstance(results, list)
        self.assertTrue(all(item.overlap_count >= 2 for item in results))

    def test_missing_user(self) -> None:
        self.assertEqual(self.service.find_similar_users(-1), [])
        self.assertIsNone(self.service.get_similar_users_opinion(-1, 1))

    def test_similar_user_opinion(self) -> None:
        opinion = self.service.get_similar_users_opinion(1, 1)
        self.assertIsNotNone(opinion)
        assert opinion is not None
        self.assertEqual(opinion.movie_id, 1)
        self.assertGreaterEqual(opinion.similar_users_count, opinion.rated_users_count)
        if opinion.rated_users_count:
            self.assertIsNotNone(opinion.average_rating)
            self.assertEqual(sum(opinion.rating_distribution.values()), opinion.rated_users_count)
