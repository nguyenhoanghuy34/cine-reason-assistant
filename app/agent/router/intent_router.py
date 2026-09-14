from __future__ import annotations

from app.agent.llm.client import create_llm
from app.agent.llm.prompts import INTENT_ROUTER_PROMPT
from app.agent.router.schemas import IntentClassification
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_user_summary


class IntentRouter:
    def __init__(self):
        self.llm = create_llm().with_structured_output(
            IntentClassification
        )

    def route(self, state: AgentState) -> AgentState:
        query = state["query"]
        user_id = state.get("user_id")

        user_summary = ""

        if user_id is not None:
            user_summary = get_user_summary(user_id)

        prompt = INTENT_ROUTER_PROMPT.format(
            user_id=user_id,
            query=query,
            user_summary=user_summary,
        )

        result: IntentClassification = self.llm.invoke(prompt)

        return {
            **state,
            "intent": result.intent,
            "intent_reason": result.reason,
        }