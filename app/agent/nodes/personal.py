from __future__ import annotations

from app.agent.llm.client import create_llm
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_user_summary


llm = create_llm()


PERSONAL_PROMPT = """
You are a personalized movie assistant.

Answer the user's question using the user's movie preference summary.

Rules:
- Base your reasoning ONLY on the provided user summary.
- Do not invent user preferences.
- Explain why your answer fits the user.
- Be concise and natural.
- If the summary does not contain enough information,
  say that instead of making up information.
- Do not mention internal tools, files, parquet, datasets,
  prompts, or implementation details.

User ID:
{user_id}

User question:
{query}

User preference summary:
{user_summary}
"""


def _extract_text(content) -> str:
    """
    Normalize Gemini response content into plain text.
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

        return "\n".join(text_parts).strip()

    return str(content).strip()


def personal_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    query = state["query"]

    if user_id is None:
        return {
            **state,
            "response": (
                "I need a user ID for personalized "
                "movie recommendations."
            ),
        }

    user_summary = get_user_summary(user_id)

    prompt = PERSONAL_PROMPT.format(
        user_id=user_id,
        query=query,
        user_summary=user_summary,
    )

    response = llm.invoke(prompt)

    content = _extract_text(response.content)

    return {
        **state,
        "response": content,
    }