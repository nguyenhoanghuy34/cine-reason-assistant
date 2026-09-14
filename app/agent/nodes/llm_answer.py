from app.agent.llm.client import create_llm
from app.agent.state import AgentState


llm = create_llm()


def llm_answer_node(state: AgentState) -> AgentState:
    query = state["query"]

    response = llm.invoke(
        f"""
You are a movie assistant.

Answer the user's question naturally and concisely.

User query:
{query}
"""
    )

    content = response.content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        content = "\n".join(text_parts)

    return {
        **state,
        "response": content,
    }