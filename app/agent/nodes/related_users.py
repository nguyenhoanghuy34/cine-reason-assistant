from app.agent.state import AgentState


def related_users_node(state: AgentState) -> AgentState:
    return {
        **state,
        "response": (
            "[RELATED_USERS ROUTE]\n"
            "This query requires finding users with similar "
            "movie taste and analyzing their opinions."
        ),
    }