from __future__ import annotations

import re
from typing import Any


def _candidate_titles(evidence: dict[str, Any]) -> list[str]:
    recommendations = evidence.get("recommendations")
    if isinstance(recommendations, dict):
        return [
            str(movie.get("title"))
            for movie in recommendations.get("candidate_movies", [])
            if movie.get("title")
        ]

    if isinstance(recommendations, list):
        titles = []
        for item in recommendations:
            data = item.get("data") if isinstance(item, dict) else {}
            titles.extend(_candidate_titles({"recommendations": data}))
        return titles

    return []


def check_recommendation_grounding(response: str, evidence: dict[str, Any]) -> dict[str, Any]:
    titles = _candidate_titles(evidence)
    if not titles:
        return {
            "status": "not_applicable",
            "reason": "no recommendation candidates in evidence",
        }

    mentioned = [
        title for title in titles
        if re.search(re.escape(title), response, flags=re.IGNORECASE)
    ]
    return {
        "status": "grounded" if mentioned else "unverified",
        "candidate_titles": titles[:10],
        "mentioned_candidate_titles": mentioned,
    }
