from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.state import AgentState


def other_node(state: AgentState) -> AgentState:
    return llm_answer_node(state)
