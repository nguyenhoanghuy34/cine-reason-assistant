import unittest

from backend.agent.tools import MovieToolbox
from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


class MovieToolboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.toolbox = MovieToolbox(MovieRepository(MovieLensDataLoader().load()))

    def test_tools_return_bounded_structured_output(self) -> None:
        movie = self.toolbox.by_name["get_movie"].invoke({"movie_id": 1})
        search = self.toolbox.by_name["search_movies"].invoke({"query": "science fiction", "top_k": 3})
        profile = self.toolbox.by_name["get_user_profile"].invoke({"user_id": 1})
        self.assertTrue(movie["ok"])
        self.assertEqual(movie["movie"]["title"], "Toy Story")
        self.assertTrue(search["ok"])
        self.assertLessEqual(len(search["results"]), 3)
        self.assertTrue(profile["ok"])

    def test_tool_errors_are_structured(self) -> None:
        result = self.toolbox.by_name["get_movie"].invoke({"movie_id": -1})
        self.assertFalse(result["ok"])
        self.assertIn("error", result)
