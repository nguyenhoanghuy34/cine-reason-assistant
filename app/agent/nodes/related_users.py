from __future__ import annotations

import json

from app.agent.llm.client import create_llm
from app.agent.state import AgentState

from app.agent.tools.related_user_tools import (
    get_top_movies_from_related_users,
)


llm = create_llm()


RELATED_USERS_PROMPT = """
You are a movie reasoning assistant.

The user is asking about movies liked by people
with similar movie taste.

Use ONLY the provided evidence.

Rules:
- Use the user summary to understand the user's taste.
- Use the related-user movie evidence as factual evidence.
- Do not invent ratings, users, movies, or preferences.
- Do not claim that all related users agree.
- Explain the recommendation or opinion using the evidence.
- If the evidence is insufficient, say so.
- Be concise and natural.
- Do not mention internal tools, files, parquet,
  datasets, prompts, or implementation details.

Current user ID:
{user_id}

User question:
{query}

Current user preference summary:
{user_summary}

Movies highly rated by users with similar taste:
{related_user_evidence}
"""


def _extract_text(content) -> str:

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":

                    text_parts.append(
                        block.get(
                            "text",
                            "",
                        )
                    )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


def related_users_node(
    state: AgentState,
) -> AgentState:

    user_id = state.get("user_id")
    query = state["query"]

    if user_id is None:

        return {
            **state,
            "response": (
                "I need a user ID to compare "
                "your taste with similar users."
            ),
        }

    # --------------------------------------------------------------
    # 1. Get current user's summary
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 2. Get movies liked by related users
    # --------------------------------------------------------------

    evidence = (
        get_top_movies_from_related_users(
            user_id=user_id,
            top_users=10,
            top_movies=15,
            min_rating=4.0,
        )
    )

    # --------------------------------------------------------------
    # 3. Build prompt
    # --------------------------------------------------------------

    prompt = RELATED_USERS_PROMPT.format(
        user_id=user_id,
        query=query,
        user_summary=user_summary,
        related_user_evidence=json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        ),
    )

    # --------------------------------------------------------------
    # 4. Ask Gemini to reason over evidence
    # --------------------------------------------------------------

    response = llm.invoke(
        prompt
    )

    content = _extract_text(
        response.content
    )

    return {
        **state,
        "related_users_evidence": evidence,
        "response": content,
    }