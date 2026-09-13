"""Optional Gemini adapter. It only writes from tool evidence; it does no ranking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ANSWER_SYSTEM_PROMPT = """You are the response writer for a Movie Discovery Assistant.
You receive only verified JSON evidence produced by deterministic MovieLens tools.

Rules:
- Use only the supplied evidence for every movie, rating, user, similarity, and recommendation claim.
- Do not calculate a new recommendation score, rating, similarity, or statistic.
- Never invent a title, year, genre, rating, user, plot, or dataset fact.
- Clearly label a conclusion as an inference when it is based on the evidence.
- Do not promise the user will like a movie.
- If evidence is absent, sparse, or reports an error, say that plainly.
- Keep the answer concise and use sections when useful: Recommendation, Why, Evidence, Confidence.
"""


def create_gemini_llm() -> Any | None:
    """Create Gemini only when its optional dependency and API key are available."""
    try:
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0,
    )
