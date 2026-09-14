from app.agent.llm.client import create_llm
from app.agent.llm.prompts import INTENT_ROUTER_PROMPT
from app.agent.router.schemas import IntentClassification
from app.agent.state import AgentState


class IntentRouter:
    def __init__(self):
        self.llm = create_llm().with_structured_output(
            IntentClassification
        )

    def route(self, state: AgentState) -> AgentState:
        query = state["query"]
        user_id = state.get("user_id")

        prompt = INTENT_ROUTER_PROMPT.format(
            user_id=user_id,
            query=query,
        )

        result: IntentClassification = self.llm.invoke(prompt)

        return {
            **state,
            "intent": result.intent,
            "intent_reason": result.reason,
        }