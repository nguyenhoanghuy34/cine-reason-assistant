import unittest

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.user_profile import UserProfileService


class UserProfileServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = UserProfileService(MovieRepository(MovieLensDataLoader().load()))

    def test_user_one_profile(self) -> None:
        profile = self.service.get_user_profile(1)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.user_id, 1)
        self.assertGreater(profile.rating_count, 0)
        self.assertGreater(len(profile.genre_statistics), 0)
        self.assertTrue(all(item.count > 0 for item in profile.genre_statistics))

    def test_missing_user(self) -> None:
        self.assertIsNone(self.service.get_user_profile(-1))
        self.assertEqual(self.service.analyze_genre_preferences(-1), ())
