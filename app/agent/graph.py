import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.other import other_node
from app.agent.nodes.personal import personal_node
from app.agent.nodes.related_users import related_users_node
from app.agent.router.intent_router import IntentRouter
from app.agent.state import AgentState


router = IntentRouter()


def get_chat_history(state: AgentState) -> list[str]:
    history = state.get("chat_history", [])
    return [item for item in history if isinstance(item, str) and item.strip()]


def intent_router_node(state: AgentState) -> AgentState:
    return router.route(state)


def remember_turn(state: AgentState) -> AgentState:
    return {"chat_history": get_chat_history(state) + [
        f"User: {state['query']}",
        f"Assistant: {state['response']}",
    ]}


def route_by_intent(state: AgentState) -> str:
    intent = state["intent"]

    if intent == "PERSONAL":
        return "personal"

    if intent == "RELATED_USERS":
        return "related_users"

    if intent == "OTHER":
        return "other"

    return "general"


def build_graph(checkpointer=None):
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

    graph.add_node("remember_turn", remember_turn)
    for node in ("general", "personal", "related_users", "other"):
        graph.add_edge(node, "remember_turn")
    graph.add_edge("remember_turn", END)

    if checkpointer is None:
        path = Path(__file__).resolve().parents[1] / "data" / "agent-memory.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    return graph.compile(checkpointer=checkpointer)


agent = build_graph()


def invoke_with_memory(user_id: int, query: str, thread_id: str = "default"):
    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")
    # Namespace sessions by user so the default thread cannot share identities.
    config = {"configurable": {"thread_id": json.dumps([user_id, str(thread_id)])}}

    state = {
        "user_id": user_id,
        "query": query.strip(),
    }

    return agent.invoke(state, config=config)
