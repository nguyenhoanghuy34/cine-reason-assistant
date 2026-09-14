from app.agent.state import AgentState


def personal_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": (
            "[PERSONAL ROUTE]\n"
            "This query requires information about the user's "
            "personal movie history or preferences."
        ),
    }