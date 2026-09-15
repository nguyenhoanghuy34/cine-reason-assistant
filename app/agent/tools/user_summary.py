from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from app.agent.llm.client import create_llm
from app.agent.llm.token_budget import compact_value
from app.agent.tools.personal_tools import get_user_profile


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEMP_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "temp-data"
)


SUMMARY_PROMPT = """
You are a movie preference analyst.

Summarize the user's movie profile into a short, useful profile
that another LLM can use later to answer personalized movie questions.

Use ONLY the information provided in the profile.

Rules:
- Do not invent preferences.
- Do not claim the user dislikes a genre just because it appears
  in low-rated genres.
- Identify the strongest preferences from top_2_genres.
- Use highly rated movies as concrete evidence.
- Mention secondary interests when supported by the data.
- Keep the summary concise.
- Do not recommend movies.
- Do not answer any user question.
- Output plain text only.

User profile:
{profile}
"""


def _extract_text(content: Any) -> str:
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


def create_user_summary(
    user_id: int,
    max_retries: int = 3,
) -> Path:
    """
    Create a compact user summary using Gemini
    and save it as a temporary TXT file.
    """

    profile = get_user_profile(user_id)

    prompt = SUMMARY_PROMPT.format(
        profile=json.dumps(
            compact_value(profile, max_list_items=8, max_dict_items=32),
            ensure_ascii=False,
            default=str,
        )
    )

    llm = create_llm()

    response = None

    for attempt in range(1, max_retries + 1):

        try:
            print(
                f"Gemini summary attempt "
                f"{attempt}/{max_retries}..."
            )

            response = llm.invoke(prompt)
            break

        except Exception as exc:

            error_text = str(exc)

            is_temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if not is_temporary_error:
                raise

            if attempt == max_retries:
                raise RuntimeError(
                    "Gemini failed to create the user summary "
                    f"after {max_retries} attempts."
                ) from exc

            wait_seconds = attempt * 3

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(
            "Gemini did not return a response."
        )

    summary = _extract_text(response.content)

    if not summary:
        raise ValueError(
            f"Gemini returned an empty summary "
            f"for user {user_id}."
        )

    TEMP_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        TEMP_DATA_DIR
        / f"user_{user_id}_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    return summary_path


def delete_user_summary(user_id: int) -> None:

    summary_path = (
        TEMP_DATA_DIR
        / f"user_{user_id}_summary.txt"
    )

    if summary_path.exists():
        summary_path.unlink()
