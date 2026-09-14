from app.agent.state import AgentState


def other_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": (
            "[OTHER ROUTE]\n"
            "This request is valid but is not supported by "
            "the current specialized routes yet."
        ),
    }