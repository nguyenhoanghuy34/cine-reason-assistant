import unittest

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


class MovieRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = MovieRepository(MovieLensDataLoader().load())

    def test_get_movie(self) -> None:
        movie = self.repository.get_movie(1)
        self.assertIsNotNone(movie)
        self.assertEqual(movie.title, "Toy Story")
        self.assertIn("Animation", movie.genres)

    def test_get_user_ratings(self) -> None:
        ratings = self.repository.get_user_ratings(1)
        self.assertIsNotNone(ratings)
        self.assertGreater(len(ratings), 0)
        self.assertTrue(all(rating.user_id == 1 for rating in ratings))

    def test_get_movie_ratings(self) -> None:
        ratings = self.repository.get_movie_ratings(1)
        self.assertIsNotNone(ratings)
        self.assertGreater(len(ratings), 0)
        self.assertTrue(all(rating.movie_id == 1 for rating in ratings))

    def test_invalid_movie_id(self) -> None:
        self.assertIsNone(self.repository.get_movie(-999))
        self.assertIsNone(self.repository.get_movie_ratings(-999))

    def test_invalid_user_id(self) -> None:
        self.assertIsNone(self.repository.get_user_ratings(-999))
        self.assertIsNone(self.repository.get_user_tags(-999))


if __name__ == "__main__":
    unittest.main()
