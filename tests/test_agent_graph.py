import unittest

from backend.agent.graph import MovieAssistantGraph
from backend.agent.tools import MovieToolbox
from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


class MovieAssistantGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = MovieAssistantGraph(MovieToolbox(MovieRepository(MovieLensDataLoader().load())))

    def test_movie_opinion_routes_to_required_tools(self) -> None:
        state = self.agent.invoke(1, "What do people with similar taste to mine think of Inception?")
        self.assertEqual(state["intent"], "movie_opinion")
        self.assertEqual([item["tool"] for item in state["tool_results"]], state["plan"])
        self.assertIn("Evidence:", state["final_answer"])

    def test_recommendation_is_grounded(self) -> None:
        state = self.agent.invoke(1, "What should I watch tonight?")
        self.assertEqual(state["intent"], "recommendation")
        self.assertIn("Confidence:", state["final_answer"])
        self.assertIn("recommend_movies", state["evidence"])

    def test_missing_history_is_not_hallucinated(self) -> None:
        state = self.agent.invoke(-1, "I haven't rated any movies. What should I watch?")
        self.assertIn("can’t verify", state["final_answer"])

    def test_required_conversation_examples_complete(self) -> None:
        examples = [
            "Why would I like Inception?",
            "I liked Toy Story but I'm tired of animated movies.",
            "I want a dark psychological thriller with a twist.",
            "What genres am I missing?",
            "Compare Inception and The Matrix for me.",
        ]
        for query in examples:
            state = self.agent.invoke(1, query)
            self.assertTrue(state["final_answer"])
            self.assertTrue(state["tool_results"])
