from langgraph.graph import END, START, StateGraph

from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.other import other_node
from app.agent.nodes.personal import personal_node
from app.agent.nodes.related_users import related_users_node
from app.agent.router.intent_router import IntentRouter
from app.agent.state import AgentState


router = IntentRouter()


def intent_router_node(state: AgentState) -> AgentState:
    return router.route(state)


def route_by_intent(state: AgentState) -> str:
    intent = state["intent"]

    if intent == "PERSONAL":
        return "personal"

    if intent == "RELATED_USERS":
        return "related_users"

    if intent == "OTHER":
        return "other"

    return "general"


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node(
        "intent_router",
        intent_router_node,
    )

    graph.add_node(
        "general",
        llm_answer_node,
    )

    graph.add_node(
        "personal",
        personal_node,
    )

    graph.add_node(
        "related_users",
        related_users_node,
    )

    graph.add_node(
        "other",
        other_node,
    )

    # Start → Router
    graph.add_edge(
        START,
        "intent_router",
    )

    # Router → Intent-specific route
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "general": "general",
            "personal": "personal",
            "related_users": "related_users",
            "other": "other",
        },
    )

    # Routes → END
    graph.add_edge("general", END)
    graph.add_edge("personal", END)
    graph.add_edge("related_users", END)
    graph.add_edge("other", END)

    return graph.compile()


agent = build_graph()