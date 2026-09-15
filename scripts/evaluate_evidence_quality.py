#!/usr/bin/env python3
"""Offline quantitative evidence evaluation with English questions and ground truth.

This script does not call the LLM. It checks whether deterministic routing,
movie-title resolution, and evidence tools produce the expected evidence for
representative English questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes.general import general_node
from app.agent.nodes.personal import personal_node
from app.agent.nodes.related_users import related_users_node
from app.agent.router.intent_router import _fallback_intent


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    user_id: int
    question: str
    state: dict
    expected_intent: str
    expected_evidence_keys: set[str]
    expected_titles: set[str] = frozenset()
    expected_answer_terms: set[str] = frozenset()


def _stub_answer(state: dict) -> dict:
    evidence = state.get("evidence", {})
    terms = []
    for key in ("movie_summary", "movie_info"):
        data = evidence.get(key, {})
        for item in data.get("summaries", []) + data.get("movies", []):
            if item.get("title"):
                terms.append(item["title"])
    for item in evidence.get("user_rating_history", {}).get("top_rated_movies", [])[:3]:
        terms.append(item.get("title", ""))
    for item in evidence.get("similar_users", {}).get("movies", [])[:3]:
        terms.append(item.get("title", ""))
    return {"response": "Answer: " + "; ".join(term for term in terms if term)}


def _collect_titles(value) -> set[str]:
    titles = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "title" and item:
                titles.add(str(item))
            else:
                titles.update(_collect_titles(item))
    elif isinstance(value, list):
        for item in value:
            titles.update(_collect_titles(item))
    return titles


def _precision_recall(expected: set[str], actual: set[str]) -> tuple[float, float]:
    if not expected:
        return 1.0, 1.0
    hits = len(expected & actual)
    precision = hits / len(actual) if actual else 0.0
    recall = hits / len(expected)
    return precision, recall


def _route_case(case: EvaluationCase) -> dict:
    routed = _fallback_intent(case.question)
    state = {"user_id": case.user_id, "query": case.question, **case.state}
    if routed is not None:
        state.update({
            "intent": routed.intent,
            "intent_reason": routed.reason,
            "target_user_ids": routed.target_user_ids,
            "movie_titles": routed.movie_titles,
            "preferred_genres": routed.preferred_genres,
            "excluded_genres": routed.excluded_genres,
            "include_terms": routed.include_terms,
            "exclude_terms": routed.exclude_terms,
            "needs_recommendations": routed.needs_recommendations,
            "needs_genre_analysis": routed.needs_genre_analysis,
            "needs_movie_info": routed.needs_movie_info,
            "needs_movie_summary": routed.needs_movie_summary,
            "needs_user_behavior": routed.needs_user_behavior,
            "needs_similarity": routed.needs_similarity,
        })
    return state


def main() -> None:
    cases = [
        EvaluationCase(
            name="personal_high_ratings",
            user_id=30,
            question="What movies have I rated highly?",
            state={"needs_user_behavior": True, "movie_titles": []},
            expected_intent="PERSONAL",
            expected_evidence_keys={"user_profile", "user_rating_history", "user_tags"},
            expected_answer_terms={"top_rated_movies"},
        ),
        EvaluationCase(
            name="movie_summary_exact_title",
            user_id=30,
            question="What is The Shawshank Redemption about?",
            state={
                "intent": "GENERAL",
                "movie_titles": ["The Shawshank Redemption"],
                "needs_movie_summary": True,
            },
            expected_intent="GENERAL",
            expected_evidence_keys={"title_resolution", "movie_summary"},
            expected_titles={"Shawshank Redemption, The"},
        ),
        EvaluationCase(
            name="movie_summary_typo_fuzzy",
            user_id=30,
            question="What is Shawshnk Redemtion about?",
            state={
                "intent": "GENERAL",
                "movie_titles": ["Shawshnk Redemtion"],
                "needs_movie_summary": True,
            },
            expected_intent="GENERAL",
            expected_evidence_keys={"title_resolution", "movie_summary"},
            expected_titles={"Shawshank Redemption, The"},
        ),
        EvaluationCase(
            name="similar_users_recommendations",
            user_id=30,
            question="What movies do users similar to me enjoy?",
            state={
                "intent": "RELATED_USERS",
                "target_user_ids": [],
                "movie_titles": [],
                "needs_similarity": True,
            },
            expected_intent="RELATED_USERS",
            expected_evidence_keys={"current_user_profile", "similar_users", "similarity_ranking"},
        ),
    ]

    original_nodes = {
        "general": general_node.__globals__["llm_answer_node"],
        "personal": personal_node.__globals__["llm_answer_node"],
        "related": related_users_node.__globals__["llm_answer_node"],
    }
    general_node.__globals__["llm_answer_node"] = _stub_answer
    personal_node.__globals__["llm_answer_node"] = _stub_answer
    related_users_node.__globals__["llm_answer_node"] = _stub_answer

    rows = []
    try:
        for case in cases:
            state = _route_case(case)
            if case.expected_intent == "PERSONAL":
                result = personal_node(state)
            elif case.expected_intent == "RELATED_USERS":
                result = related_users_node(state)
            else:
                result = general_node(state)

            evidence = result.get("evidence", {})
            actual_keys = set(evidence.keys())
            actual_titles = _collect_titles(evidence)
            key_precision, key_recall = _precision_recall(case.expected_evidence_keys, actual_keys)
            title_precision, title_recall = _precision_recall(case.expected_titles, actual_titles)
            passed = (
                case.expected_evidence_keys <= actual_keys
                and case.expected_titles <= actual_titles
            )
            rows.append({
                "case": case.name,
                "passed": passed,
                "key_precision": round(key_precision, 3),
                "key_recall": round(key_recall, 3),
                "title_precision": round(title_precision, 3),
                "title_recall": round(title_recall, 3),
                "expected_keys": sorted(case.expected_evidence_keys),
                "actual_keys": sorted(actual_keys),
                "expected_titles": sorted(case.expected_titles),
                "actual_titles_sample": sorted(actual_titles)[:5],
            })
    finally:
        general_node.__globals__["llm_answer_node"] = original_nodes["general"]
        personal_node.__globals__["llm_answer_node"] = original_nodes["personal"]
        related_users_node.__globals__["llm_answer_node"] = original_nodes["related"]

    passed_count = sum(row["passed"] for row in rows)
    print("Quantitative evidence evaluation")
    print("Language: English questions, English answer contract")
    print("Ground truth: expected intent, required evidence keys, and expected resolved titles")
    print(f"Cases: {len(rows)}")
    print(f"Pass rate: {passed_count / len(rows):.3f}")
    print()

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
