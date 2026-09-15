from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.agent.llm.client import create_llm
from app.agent.llm.prompts import INTENT_ROUTER_PROMPT
from app.agent.llm.token_budget import compact_history, compact_value
from app.agent.router.schemas import IntentClassification
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_user_profile


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content).strip()


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Router did not return JSON.")
    return json.loads(match.group(0))


def _fallback_intent(query: str) -> IntentClassification | None:
    text = query.lower()

    personal_patterns = [
        "have i rated",
        "i rated",
        "my rated",
        "my ratings",
        "movies have i",
        "what movies have i",
    ]
    high_rating_patterns = [
        "rated highly",
        "highly rated",
        "rated high",
        "top rated",
        "highest rated",
        "liked",
    ]

    if any(pattern in text for pattern in personal_patterns):
        return IntentClassification(
            intent="PERSONAL",
            reason="The question asks about the current user's rating history.",
            needs_user_behavior=True,
        )

    if "recommend" in text or "suggest" in text or "should i watch" in text:
        return IntentClassification(
            intent="PERSONAL",
            reason="The question asks for personalized recommendations.",
            needs_recommendations=True,
            needs_user_behavior=any(pattern in text for pattern in high_rating_patterns),
        )

    return None


class IntentRouter:

    def __init__(self):
        self.llm = create_llm()

    def route(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]
        user_id = state.get("user_id")

        result = _fallback_intent(query)
        if result is None:
            user_profile = {}

            if user_id is not None:

                try:
                    user_profile = get_user_profile(user_id)
                except (FileNotFoundError, ValueError):
                    user_profile = {"user_id": user_id, "status": "profile unavailable"}

            history = compact_history(state.get("chat_history", []), max_items=6)
            history_text = "\n".join(
                f"- {item}" for item in history if isinstance(item, str) and item.strip()
            )

            prompt = INTENT_ROUTER_PROMPT.format(
                user_id=user_id,
                query=query,
                chat_history=history_text or "No previous conversation yet.",
                user_summary=compact_value(user_profile, max_list_items=5, max_dict_items=24),
            )

            response = self.llm.invoke(prompt)
            try:
                result = IntentClassification.model_validate(
                    _extract_json(_extract_text(response.content))
                )
            except (ValueError, json.JSONDecodeError, ValidationError):
                result = IntentClassification(
                    intent="OTHER",
                    reason="Router could not parse the model classification.",
                )

        return {
            **state,
            "intent": result.intent,
            "intent_reason": result.reason,
            "target_user_ids": result.target_user_ids,
            "movie_titles": result.movie_titles,
            "preferred_genres": result.preferred_genres,
            "excluded_genres": result.excluded_genres,
            "include_terms": result.include_terms,
            "exclude_terms": result.exclude_terms,
            "needs_recommendations": result.needs_recommendations,
            "needs_genre_analysis": result.needs_genre_analysis,
            "needs_movie_info": result.needs_movie_info,
            "needs_movie_summary": result.needs_movie_summary,
            "needs_user_behavior": result.needs_user_behavior,
            "needs_similarity": result.needs_similarity,
            "evidence": {},
            "related_users_evidence": {},
            "response": "",
        }
