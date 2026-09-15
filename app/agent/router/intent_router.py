from __future__ import annotations

from app.agent.llm.client import create_llm
from app.agent.llm.prompts import INTENT_ROUTER_PROMPT
from app.agent.router.schemas import IntentClassification
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_user_profile


class IntentRouter:

    def __init__(self):

        self.llm = create_llm().with_structured_output(
            IntentClassification
        )

    def route(
        self,
        state: AgentState,
    ) -> AgentState:

        query = state["query"]
        user_id = state.get("user_id")

        user_profile = {}

        if user_id is not None:

            try:
                user_profile = get_user_profile(user_id)
            except (FileNotFoundError, ValueError):
                user_profile = {"user_id": user_id, "status": "profile unavailable"}

        history = state.get("chat_history", [])
        history_text = "\n".join(
            f"- {item}" for item in history if isinstance(item, str) and item.strip()
        )

        prompt = INTENT_ROUTER_PROMPT.format(
            user_id=user_id,
            query=query,
            chat_history=history_text or "No previous conversation yet.",
            user_summary=user_profile,
        )

        result = self.llm.invoke(
            prompt
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
            "evidence": {},
            "related_users_evidence": {},
            "response": "",
        }
