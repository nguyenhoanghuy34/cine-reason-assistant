from __future__ import annotations

import json

from app.agent.llm.client import create_llm
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_user_profile
from app.agent.tools.recommendation_tools import (
    get_recommendation_candidates,
)


llm = create_llm()


PERSONAL_PROMPT = """
You are a personalized movie assistant.

Answer the user's question using the provided evidence.

There are two types of evidence:

1. USER PROFILE
   - Information about the user's movie preferences.

2. MOVIE CANDIDATES
   - Movies that the Python system has already selected
     as unwatched candidates for this user.

Rules:
- Use ONLY the provided evidence.
- Do not invent user preferences.
- Do not invent movies.
- Do not invent movie IDs.
- Do not invent genres.
- Do not invent plot information.
- For recommendations, ONLY recommend movies
  from candidate_movies.
- Explain briefly why the recommendation fits
  the user's preferences.
- Be concise and natural.
- If there is not enough evidence, say so.
- Do not mention internal tools, files, parquet,
  datasets, prompts, or implementation details.

For recommendation questions:
- Use top_2_genres as an important preference signal.
- Analyze candidate movie genres.
- Analyze candidate movie plots when useful.
- Select the best matching movies.
- Return TOP 5 when at least 5 candidates are available.
- Rank recommendations from #1 to #5.

User ID:
{user_id}

User question:
{query}

User profile:
{user_profile}

Recommendation evidence:
{recommendation_evidence}
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
                        block.get("text", "")
                    )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


def personal_node(
    state: AgentState,
) -> AgentState:

    user_id = state.get("user_id")
    query = state["query"]

    # ============================================================
    # 1. USER ID CHECK
    # ============================================================

    if user_id is None:

        return {
            **state,
            "response": (
                "I need a user ID for "
                "personalized movie recommendations."
            ),
            "evidence": {},
        }

    # ============================================================
    # 2. LOAD USER PROFILE
    # ============================================================

    user_profile = get_user_profile(
        user_id
    )

    # ============================================================
    # 3. GET RECOMMENDATION EVIDENCE
    # ============================================================

    recommendation_evidence = (
        get_recommendation_candidates(
            user_id=user_id,
            max_candidates=50,
        )
    )

    # ============================================================
    # 4. BUILD PROMPT
    # ============================================================

    prompt = PERSONAL_PROMPT.format(
        user_id=user_id,
        query=query,
        user_profile=json.dumps(
            user_profile,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
        recommendation_evidence=json.dumps(
            recommendation_evidence,
            ensure_ascii=False,
            default=str,
            indent=2,
        ),
    )

    # ============================================================
    # 5. LLM
    # ============================================================

    response = llm.invoke(
        prompt
    )

    content = _extract_text(
        response.content
    )

    # ============================================================
    # 6. SAVE EVIDENCE FOR DISPLAY
    # ============================================================

    evidence = {
        "user_id": user_id,
        "top_2_genres": recommendation_evidence.get(
            "top_2_genres",
            [],
        ),
        "candidate_movies": recommendation_evidence.get(
            "candidate_movies",
            [],
        ),
    }

    # ============================================================
    # 7. RETURN
    # ============================================================

    return {
        **state,
        "response": content,
        "evidence": evidence,
    }