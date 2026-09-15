from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver

from app.agent import graph
from app.agent.nodes import personal
from app.agent.router.intent_router import _fallback_intent
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


def test_rating_history_question_routes_to_personal_without_llm():
    result = _fallback_intent("What movies have I rated highly?")
    assert result.intent == "PERSONAL"
    assert result.needs_user_behavior is True


def test_memory_keeps_repeated_turns_and_isolates_users(monkeypatch):
    monkeypatch.setattr(graph.router, "route", lambda state: {"intent": "GENERAL", "evidence": {}})
    monkeypatch.setattr(graph, "general_node", lambda state: {"response": "Zodiac"})
    monkeypatch.setattr(graph, "agent", graph.build_graph(MemorySaver()))
    first = graph.invoke_with_memory(15, "Recommend a movie")
    second = graph.invoke_with_memory(15, "Recommend a movie")
    other = graph.invoke_with_memory(2, "Why?")
    assert len(first["chat_history"]) == 2
    assert len(second["chat_history"]) == 4
    assert second["chat_history"].count("User: Recommend a movie") == 2
    assert other["chat_history"] == ["User: Why?", "Assistant: Zodiac"]


def test_personal_recommendations_receive_candidates_and_history(monkeypatch):
    monkeypatch.setattr(personal, "get_personal_evidence", lambda _: {"user_id": 15})
    def recommendations(user_id, constraints=None):
        return {"candidate_movies": [{"title": "Heat"}], "constraints": constraints}

    monkeypatch.setattr(personal, "get_recommendation_evidence", recommendations)
    answer = Mock(return_value={"response": "LLM answer"})
    monkeypatch.setattr(personal, "llm_answer_node", answer)
    personal.personal_node({"user_id": 15, "query": "Gợi ý tiếp", "needs_recommendations": True,
                            "excluded_genres": ["Animation"],
                            "chat_history": ["User: I like animated movies."]})
    supplied = answer.call_args.args[0]
    assert supplied["evidence"]["recommendations"]["candidate_movies"]
    assert supplied["evidence"]["recommendations"]["constraints"]["excluded_genres"] == ["Animation"]
    assert supplied["chat_history"] == ["User: I like animated movies."]


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


def test_movie_title_resolution_uses_canonical_titles(monkeypatch):
    def read_csv(path):
        if str(path).endswith("movies_with_plots.csv"):
            return pd.DataFrame({
                "movieId": [1, 2, 3],
                "title": ["Heat (1995)", "Body Heat (1981)", "Toy Story (1995)"],
                "year": [1995, 1981, 1995],
                "genres": ["Action", "Crime", "Animation"],
                "plot": ["A detective pursues a thief.", "Noir thriller.", "Toys become friends."],
            })
        return pd.DataFrame({"movieId": [], "rating": []})

    monkeypatch.setattr(movie_data_tools.pd, "read_csv", read_csv)

    resolved = movie_data_tools.resolve_movie_titles(
        "What is Heat about?",
        ["Heat"],
    )
    assert resolved == ["Heat (1995)"]
    assert movie_data_tools.get_movie_summary(resolved)["summaries"][0]["title"] == "Heat (1995)"


def test_movie_title_resolution_handles_trailing_articles(monkeypatch):
    monkeypatch.setattr(movie_data_tools.pd, "read_csv", lambda _: pd.DataFrame({
        "movieId": [318, 81520],
        "title": ["Shawshank Redemption, The", "Undisputed III: Redemption"],
        "year": [1994, 2010],
        "genres": ["Crime|Drama", "Action|Crime|Drama"],
        "plot": ["Andy and Red survive prison through hope.", "A prison fighting story."],
    }))

    resolved = movie_data_tools.resolve_movie_titles(
        "What is The Shawshank Redemption about?",
        ["The Shawshank Redemption"],
    )
    assert resolved == ["Shawshank Redemption, The"]
    summary = movie_data_tools.get_movie_summary(resolved)
    assert summary["summaries"][0]["movie_id"] == 318


def test_movie_title_resolution_handles_typos_with_fuzzy_match(monkeypatch):
    monkeypatch.setattr(movie_data_tools.pd, "read_csv", lambda _: pd.DataFrame({
        "movieId": [318, 81520],
        "title": ["Shawshank Redemption, The", "Undisputed III: Redemption"],
        "year": [1994, 2010],
        "genres": ["Crime|Drama", "Action|Crime|Drama"],
        "plot": ["Andy and Red survive prison through hope.", "A prison fighting story."],
    }))

    resolved = movie_data_tools.resolve_movie_titles(
        "What is Shawshnk Redemtion about?",
        ["Shawshnk Redemtion"],
    )
    evidence = movie_data_tools.get_title_resolution_evidence(
        "What is Shawshnk Redemtion about?",
        ["Shawshnk Redemtion"],
    )

    assert resolved == ["Shawshank Redemption, The"]
    assert evidence["matches"][0]["match_type"] == "fuzzy"
    assert evidence["matches"][0]["confidence"] >= 0.82


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
        graph.invoke_with_memory(15, "I like mystery movies")
    finally:
        connection.close()
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        monkeypatch.setattr(graph, "agent", graph.build_graph(SqliteSaver(connection)))
        result = graph.invoke_with_memory(15, "What do I like?")
        assert result["chat_history"][0] == "User: I like mystery movies"
        assert len(result["chat_history"]) == 4
    finally:
        connection.close()
