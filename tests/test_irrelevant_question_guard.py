from types import SimpleNamespace
from unittest.mock import Mock

from app.agent.nodes import llm_answer
from app.main import _format_evidence


def test_irrelevant_question_is_answered_by_llm(monkeypatch):
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="Tôi có thể giúp bạn về phim.")
    monkeypatch.setattr(llm_answer, "llm", model)
    result = llm_answer.llm_answer_node({"query": "Thời tiết hôm nay?", "intent": "OTHER"})
    assert result["response"] == model.invoke.return_value.content
    model.invoke.assert_called_once()


def test_vietnamese_followup_reaches_llm_with_both_roles(monkeypatch):
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content="Vì phim đó phù hợp với sở thích bạn đã nói.")
    monkeypatch.setattr(llm_answer, "llm", model)
    history = ["User: Tôi thích phim trinh thám.", "Assistant: Bạn có thể xem Zodiac."]
    llm_answer.llm_answer_node({"query": "Vì sao chọn phim đó?", "chat_history": history})
    prompt = model.invoke.call_args.args[0][1][1]
    assert "Zodiac" in prompt
    assert "Tôi thích phim trinh thám" in prompt
    assert "Vì sao chọn phim đó?" in prompt


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
        "response": "Bạn nên xem Heat.",
        "evidence": {
            "recommendations": {
                "candidate_movies": [{"title": "Heat", "average_rating": 4.0}],
            }
        },
    }
    evidence = _format_evidence(result)
    assert "candidates=Heat (4.0)" in evidence
    assert "grounding=grounded" in evidence
