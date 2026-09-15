from types import SimpleNamespace
from unittest.mock import Mock

from app.agent.nodes import llm_answer
from app.main import _format_evidence


def test_irrelevant_question_is_answered_by_llm(monkeypatch):
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="I can help with movie questions.")
    monkeypatch.setattr(llm_answer, "llm", model)
    result = llm_answer.llm_answer_node({"query": "Thời tiết hôm nay?", "intent": "OTHER"})
    assert result["response"] == model.invoke.return_value.content
    model.invoke.assert_called_once()


def test_english_followup_reaches_llm_with_both_roles(monkeypatch):
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="Because that movie matches your stated taste.")
    monkeypatch.setattr(llm_answer, "llm", model)
    history = ["User: I like mystery movies.", "Assistant: You could watch Zodiac."]
    llm_answer.llm_answer_node({"query": "Why did you choose that movie?", "chat_history": history})
    prompt = model.invoke.call_args.args[0][1][1]
    assert "Zodiac" in prompt
    assert "I like mystery movies" in prompt
    assert "Why did you choose that movie?" in prompt


def test_output_evidence_is_deterministic():
    result = {
        "intent": "PERSONAL",
        "user_id": 15,
        "evidence": {
            "recommendations": {
                "candidate_movies": [{"title": "Heat"}, {"title": "Memento"}],
                "constraints": {"excluded_genres": ["Animation"]},
            }
        },
    }
    evidence = _format_evidence(result)
    assert "intent=PERSONAL" in evidence
    assert "user_id=15" in evidence
    assert "candidates=Heat (n/a); Memento (n/a)" in evidence
    assert "excluded_genres" in evidence
    assert "grounding=grounded" not in evidence


def test_output_evidence_marks_grounded_recommendations():
    result = {
        "intent": "PERSONAL",
        "user_id": 15,
        "response": "You should watch Heat.",
        "evidence": {
            "recommendations": {
                "candidate_movies": [{"title": "Heat", "average_rating": 4.0}],
            }
        },
    }
    evidence = _format_evidence(result)
    assert "candidates=Heat (4.0)" in evidence
    assert "grounding=grounded" in evidence
