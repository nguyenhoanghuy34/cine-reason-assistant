from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from app.agent import graph
from app.agent.nodes import personal
from app.agent.tools import blind_spot_tools
from app.agent.tools import movie_data_tools
from app.agent.tools import recommendation_tools
from app.agent.tools import related_user_tools
from app.agent.tools import user_action_tools


def test_related_users_exclude_only_current_user(monkeypatch):
    monkeypatch.setattr(related_user_tools.Path, "exists", lambda _: True)
    monkeypatch.setattr(related_user_tools.pd, "read_parquet", lambda _: pd.DataFrame({
        "user_id": [15], "related_user_ids": [[15, 2, 7]],
    }))
    assert related_user_tools.get_related_user_ids(15) == [2, 7]


def test_memory_keeps_repeated_turns_and_isolates_users(monkeypatch):
    monkeypatch.setattr(graph.router, "route", lambda state: {"intent": "GENERAL", "evidence": {}})
    monkeypatch.setattr(graph, "general_node", lambda state: {"response": "Zodiac"})
    monkeypatch.setattr(graph, "agent", graph.build_graph(MemorySaver()))
    first = graph.invoke_with_memory(15, "Gợi ý phim")
    second = graph.invoke_with_memory(15, "Gợi ý phim")
    other = graph.invoke_with_memory(2, "Vì sao?")
    assert len(first["chat_history"]) == 2
    assert len(second["chat_history"]) == 4
    assert second["chat_history"].count("User: Gợi ý phim") == 2
    assert other["chat_history"] == ["User: Vì sao?", "Assistant: Zodiac"]


def test_personal_recommendations_receive_candidates_and_history(monkeypatch):
    monkeypatch.setattr(personal, "get_personal_evidence", lambda _: {"user_id": 15})
    def recommendations(user_id, constraints=None):
        return {"candidate_movies": [{"title": "Heat"}], "constraints": constraints}

    monkeypatch.setattr(personal, "get_recommendation_evidence", recommendations)
    answer = Mock(return_value={"response": "LLM answer"})
    monkeypatch.setattr(personal, "llm_answer_node", answer)
    personal.personal_node({"user_id": 15, "query": "Gợi ý tiếp", "needs_recommendations": True,
                            "excluded_genres": ["Animation"],
                            "chat_history": ["User: Tôi thích hoạt hình."]})
    supplied = answer.call_args.args[0]
    assert supplied["evidence"]["recommendations"]["candidate_movies"]
    assert supplied["evidence"]["recommendations"]["constraints"]["excluded_genres"] == ["Animation"]
    assert supplied["chat_history"] == ["User: Tôi thích hoạt hình."]


def test_recommendation_constraints_filter_excluded_genres(monkeypatch):
    monkeypatch.setattr(recommendation_tools, "_add_tags", lambda movies: movies.assign(tags=""))
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story", "Heat"],
        "genres": ["Adventure|Animation|Children", "Action|Crime|Thriller"],
        "plot": ["toys become friends", "a dark crime thriller"],
        "_order": [0, 1],
    })
    filtered = recommendation_tools._score_candidates(
        movies,
        {"excluded_genres": ["Animation"], "preferred_genres": ["Thriller"]},
    )
    assert filtered["title"].tolist() == ["Heat"]


def test_blind_spot_handles_missing_numeric_values(monkeypatch):
    class ExistingProfile:
        def exists(self):
            return True

    monkeypatch.setattr(blind_spot_tools, "PROFILE_FILE", ExistingProfile())
    monkeypatch.setattr(blind_spot_tools.pd, "read_parquet", lambda _: pd.DataFrame({
        "user_id": [15],
        "genre_watch_counts": [{"(no genres listed)": None, "Sci-Fi": 3}],
        "genre_avg_ratings": [{"(no genres listed)": None, "Sci-Fi": 4.0}],
        "genre_high_rated_counts": [{"(no genres listed)": None, "Sci-Fi": 2}],
        "genre_high_rating_ratio": [{"(no genres listed)": None, "Sci-Fi": 0.67}],
        "unwatched_genres": [["Documentary"]],
        "underexposed_genres": [["Western"]],
    }))
    result = blind_spot_tools.get_blind_spot_evidence(15)
    missing = [
        row for row in result["genre_analysis"]
        if row["genre"] == "(no genres listed)"
    ][0]
    assert missing["watch_count"] == 0
    assert missing["high_rated_count"] == 0


def test_movie_info_tool_returns_metadata(monkeypatch):
    monkeypatch.setattr(movie_data_tools.pd, "read_csv", lambda path: pd.DataFrame({
        "movieId": [1],
        "title": ["Heat"],
        "year": [1995],
        "genres": ["Action|Crime|Thriller"],
        "plot": ["A detective pursues a thief."],
        "rating": [4.0],
        "tag": ["crime"],
    }) if str(path).endswith("movies_with_plots.csv") else pd.DataFrame({
        "movieId": [1],
        "rating": [4.0],
        "tag": ["crime"],
    }))
    result = movie_data_tools.get_movie_summary(["Heat"])
    assert result["summaries"][0]["title"] == "Heat"


def test_user_rating_history_tool_returns_actions(monkeypatch):
    def read_csv(path):
        if str(path).endswith("ratings.csv"):
            return pd.DataFrame({
                "userId": [15, 15],
                "movieId": [1, 2],
                "rating": [5.0, 3.0],
                "timestamp": [20, 10],
            })
        return pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Heat", "Toy Story"],
            "genres": ["Action", "Animation"],
        })

    monkeypatch.setattr(user_action_tools.pd, "read_csv", read_csv)
    result = user_action_tools.get_user_rating_history(15)
    assert result["top_rated_movies"][0]["title"] == "Heat"


def test_memory_survives_reopening_database(monkeypatch):
    import os
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    monkeypatch.setattr(graph.router, "route", lambda state: {"intent": "GENERAL"})
    monkeypatch.setattr(graph, "general_node", lambda state: {"response": "Zodiac"})
    path = "app/data/test-memory-persistence.sqlite"
    if os.path.exists(path):
        os.unlink(path)
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        monkeypatch.setattr(graph, "agent", graph.build_graph(SqliteSaver(connection)))
        graph.invoke_with_memory(15, "Tôi thích trinh thám")
    finally:
        connection.close()
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        monkeypatch.setattr(graph, "agent", graph.build_graph(SqliteSaver(connection)))
        result = graph.invoke_with_memory(15, "Tôi thích gì?")
        assert result["chat_history"][0] == "User: Tôi thích trinh thám"
        assert len(result["chat_history"]) == 4
    finally:
        connection.close()
