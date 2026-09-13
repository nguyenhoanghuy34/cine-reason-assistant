"""A bounded LangGraph workflow that grounds answers in tool evidence."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .llm import ANSWER_SYSTEM_PROMPT, create_gemini_llm
from .tools import MovieToolbox

SYSTEM_PROMPT = """You are a Movie Discovery Assistant. Use only supplied tool evidence for
movie metadata, ratings, users, similarity, and recommendations. Never invent a dataset fact.
State uncertainty when evidence is sparse. Distinguish facts from inferences, and never promise
that a user will like a film. If information is unavailable, say so plainly."""


class AgentState(TypedDict, total=False):
    user_id: int
    query: str
    intent: str
    plan: list[str]
    tool_results: list[dict[str, Any]]
    evidence: dict[str, Any]
    final_answer: str


class MovieAssistantGraph:
    """START → understand → plan → execute tools → collect evidence → answer → END."""

    def __init__(self, toolbox: MovieToolbox, llm: Any | None = None) -> None:
        self.toolbox = toolbox
        self._llm = llm if llm is not None else create_gemini_llm()
        builder = StateGraph(AgentState)
        builder.add_node("understand_query", self._understand_query)
        builder.add_node("build_plan", self._build_plan)
        builder.add_node("execute_tools", self._execute_tools)
        builder.add_node("collect_evidence", self._collect_evidence)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_edge(START, "understand_query")
        builder.add_edge("understand_query", "build_plan")
        builder.add_edge("build_plan", "execute_tools")
        builder.add_edge("execute_tools", "collect_evidence")
        builder.add_edge("collect_evidence", "generate_answer")
        builder.add_edge("generate_answer", END)
        self.graph = builder.compile()

    def invoke(self, user_id: int, query: str) -> AgentState:
        return self.graph.invoke({"user_id": user_id, "query": query})

    @staticmethod
    def _understand_query(state: AgentState) -> dict[str, Any]:
        query = state["query"].casefold()
        if "similar taste" in query and ("think" in query or "opinion" in query):
            intent = "movie_opinion"
        elif query.startswith("compare ") or " compare " in query:
            intent = "comparison"
        elif "why would i like" in query or query.startswith("why "):
            intent = "explanation"
        elif "similar to" in query:
            intent = "similar_movie"
        elif "similar users" in query:
            intent = "similar_users"
        elif "what genres" in query or "my taste" in query:
            intent = "taste_analysis"
        elif "profile" in query or "my ratings" in query:
            intent = "user_profile"
        elif any(term in query for term in ("should i watch", "recommend", "i liked", "what should i watch", "haven't rated", "have not rated")):
            intent = "recommendation"
        else:
            intent = "movie_search"
        return {"intent": intent}

    @staticmethod
    def _build_plan(state: AgentState) -> dict[str, Any]:
        intent = state["intent"]
        plans = {
            "recommendation": ["get_user_profile", "recommend_movies"],
            "movie_opinion": ["search_movies", "get_user_profile", "find_similar_users", "get_similar_users_opinion"],
            "explanation": ["search_movies", "get_movie", "get_user_profile", "find_similar_movies"],
            "comparison": ["search_movies", "get_movie"],
            "similar_movie": ["search_movies", "find_similar_movies"],
            "similar_users": ["get_user_profile", "find_similar_users"],
            "user_profile": ["get_user_profile", "get_user_ratings"],
            "taste_analysis": ["analyze_user_taste"],
            "movie_search": ["search_movies"],
        }
        return {"plan": plans[intent]}

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        movie_ids: list[int] = []
        for name in state["plan"]:
            arguments: dict[str, Any] = {"user_id": state["user_id"]}
            if name == "search_movies":
                if state["intent"] == "comparison":
                    searches = []
                    missing = []
                    for title in _comparison_titles(state["query"]):
                        found = self.toolbox.by_name[name].invoke({"query": title, "top_k": 5})
                        exact = [item for item in found.get("results", []) if _title_equivalent(title, item["title"])]
                        if exact:
                            searches.extend(exact[:1])
                        else:
                            missing.append(title)
                    output = {"ok": True, "results": searches, "missing_titles": missing}
                    results.append({"tool": name, "output": output})
                    movie_ids = [item["movie_id"] for item in output["results"]]
                    continue
                arguments = {"query": _title_query(state["query"]), "top_k": 5}
            elif name in {"get_movie", "find_similar_movies", "get_similar_users_opinion"}:
                if not movie_ids:
                    continue
                arguments = {"movie_id": movie_ids[0]}
                if name == "get_similar_users_opinion":
                    arguments["user_id"] = state["user_id"]
            elif name == "recommend_movies":
                arguments = {"user_id": state["user_id"], "query": _recommendation_query(state["query"]), "exclude_genres": _excluded_genres(state["query"]), "top_k": 5}
            try:
                output = self.toolbox.by_name[name].invoke(arguments)
            except Exception as exc:  # Tool failures become evidence, never hidden claims.
                output = {"ok": False, "error": f"{name} failed: {exc}"}
            results.append({"tool": name, "output": output})
            if name == "search_movies" and output.get("ok"):
                movie_ids = [item["movie_id"] for item in output["results"]]
        return {"tool_results": results}

    @staticmethod
    def _collect_evidence(state: AgentState) -> dict[str, Any]:
        return {"evidence": {item["tool"]: item["output"] for item in state["tool_results"]}}

    def _generate_answer(self, state: AgentState) -> dict[str, Any]:
        """Let Gemini summarize verified evidence; retain a safe local fallback."""
        fallback = self._grounded_answer(state)
        if self._llm is None or any(not value.get("ok") for value in state["evidence"].values()):
            return fallback
        prompt = json.dumps(
            {"user_query": state["query"], "intent": state["intent"], "plan": state["plan"], "evidence": state["evidence"]},
            ensure_ascii=False,
            default=str,
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=f"Write the final response from this verified evidence only:\n{prompt}"),
            ])
            content = response.content if isinstance(response.content, str) else str(response.content)
            return {"final_answer": content.strip() or fallback["final_answer"]}
        except Exception:
            # A provider/network failure must never turn into an unsupported claim.
            return fallback

    @staticmethod
    def _grounded_answer(state: AgentState) -> dict[str, Any]:
        evidence = state["evidence"]
        errors = list(dict.fromkeys(value["error"] for value in evidence.values() if not value.get("ok")))
        if errors:
            return {"final_answer": "I can't verify this from the supplied MovieLens data: " + " ".join(errors)}
        if state["intent"] == "recommendation":
            items = evidence.get("recommend_movies", {}).get("recommendations", [])
            if not items:
                return {"final_answer": "I don't have rating history for this user, so I can't produce a personalized recommendation from this dataset."}
            lines = ["Recommendation (inference from the hybrid ranking):"]
            for item in items[:3]:
                ev = item["evidence"]
                lines.append(f"- {item['title']} ({item['year']}) - score {item['final_score']}; collaborative={ev['collaborative_score']}, genre={ev['genre_score']}, quality={ev['quality_score']}.")
            lines.append("Confidence: moderate; scores are dataset-derived signals, not a guarantee you will enjoy these films.")
            return {"final_answer": "\n".join(lines)}
        if state["intent"] == "movie_opinion":
            opinion = evidence.get("get_similar_users_opinion", {}).get("opinion")
            if not opinion or opinion["rated_users_count"] == 0:
                return {"final_answer": "No similar-user rating evidence is available for that movie in this dataset."}
            return {"final_answer": f"Evidence: {opinion['rated_users_count']} of {opinion['similar_users_count']} similar users rated it; their average was {opinion['average_rating']}/5 and positive ratio was {opinion['positive_ratio']}. Distribution: {opinion['rating_distribution']}. Confidence depends on the {opinion['rated_users_count']} ratings."}
        if state["intent"] == "explanation":
            movie = evidence.get("get_movie", {}).get("movie")
            profile = evidence.get("get_user_profile", {}).get("profile", {})
            if not movie:
                return {"final_answer": "I could not verify that movie in the supplied MovieLens data."}
            favorites = {item["genre"] for item in profile.get("favorite_genres", [])}
            overlap = [genre for genre in movie["genres"] if genre in favorites]
            examples = [item["title"] for item in profile.get("highly_rated_movies", [])[:3]]
            related = evidence.get("find_similar_movies", {}).get("results", [])
            lines = [f"Verified facts: {movie['title']} ({movie['year']}) has genres {movie['genres']}. Your profile has {profile.get('rating_count', 0)} recorded ratings and favorite genres {[item['genre'] for item in profile.get('favorite_genres', [])]}."]
            if overlap:
                lines.append(f"Inference: shared genres {overlap} may fit your recorded taste; highly rated examples include {examples}.")
            else:
                lines.append("Inference: the available genre evidence shows no direct favorite-genre overlap, so confidence is low.")
            if related:
                lines.append(f"Content evidence: related retrieval includes {[item['title'] for item in related[:3]]}.")
            lines.append("This is not a guarantee that you will like it.")
            return {"final_answer": "\n".join(lines)}
        if state["intent"] == "taste_analysis":
            profile = evidence.get("analyze_user_taste", {}).get("profile", {})
            return {"final_answer": f"Evidence from {profile.get('rating_count', 0)} ratings: genres with no recorded ratings are {profile.get('unrated_genres', [])}; weaker rated genres are {profile.get('weak_genres', [])}. This is coverage of this dataset, not proof you dislike those genres."}
        if state["intent"] == "comparison":
            movies = evidence.get("search_movies", {}).get("results", [])
            missing = evidence.get("search_movies", {}).get("missing_titles", [])
            if missing:
                return {"final_answer": f"Verified comparison evidence: {[(item['title'], item['year'], item['genres']) for item in movies]}. I could not verify {missing} in this filtered dataset, so I won’t compare it using unrelated search results."}
            return {"final_answer": f"Verified comparison evidence: {[(item['title'], item['year'], item['genres']) for item in movies[:2]]}. These are metadata facts; a stronger preference comparison would need your rating evidence for both."}
        movies = evidence.get("search_movies", {}).get("results", [])
        if not movies:
            return {"final_answer": "I could not find matching evidence in the supplied MovieLens dataset."}
        return {"final_answer": "Evidence: " + "; ".join(f"{item['title']} ({item['year']}, {', '.join(item['genres'])})" for item in movies[:3]) + "."}


def _title_query(query: str) -> str:
    cleaned = re.sub(r"(?i)what do people with similar taste to mine think (of|about)|why would i like|compare|for me", "", query)
    return cleaned.strip(" ?.") or query


def _recommendation_query(query: str) -> str | None:
    if any(word in query.casefold() for word in ("tonight", "should i watch", "what should")):
        return None
    if "i liked" in query.casefold():
        return None
    return query


def _excluded_genres(query: str) -> list[str]:
    return ["Animation"] if "tired of animated" in query.casefold() or "tired of animation" in query.casefold() else []


def _comparison_titles(query: str) -> list[str]:
    match = re.search(r"(?i)compare\s+(.+?)\s+and\s+(.+?)(?:\s+for me)?[?.!]*$", query)
    return [match.group(1).strip(), match.group(2).strip()] if match else [_title_query(query)]


def _title_equivalent(requested: str, result: str) -> bool:
    def normalized(value: str) -> str:
        value = value.casefold().strip()
        article = re.fullmatch(r"(the|a|an)\s+(.+)", value)
        return f"{article.group(2)}, {article.group(1)}" if article else value
    return normalized(requested) == normalized(result)
