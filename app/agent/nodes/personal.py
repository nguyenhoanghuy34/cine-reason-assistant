from concurrent.futures import ThreadPoolExecutor

from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_personal_evidence
from app.agent.tools.recommendation_tools import get_recommendation_evidence
from app.agent.tools.blind_spot_tools import get_blind_spot_evidence
from app.agent.tools.movie_data_tools import get_title_resolution_evidence, resolve_movie_titles
from app.agent.tools.related_user_tools import get_movie_opinions
from app.agent.tools.user_action_tools import get_user_rating_history, get_user_tags


def collect_evidence(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def collect_evidence_parallel(tasks: dict):
    if not tasks:
        return {}

    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        futures = {
            key: executor.submit(loader, *args, **kwargs)
            for key, (loader, args, kwargs) in tasks.items()
        }
        return {
            key: collect_evidence(lambda future=future: future.result())
            for key, future in futures.items()
        }


def personal_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    movie_titles = resolve_movie_titles(state.get("query", ""), state.get("movie_titles", []))
    constraints = {
        "preferred_genres": state.get("preferred_genres", []),
        "excluded_genres": state.get("excluded_genres", []),
        "include_terms": state.get("include_terms", []),
        "exclude_terms": state.get("exclude_terms", []),
        "seed_titles": movie_titles,
    }
    evidence = {"user_id": user_id}
    if user_id is not None:
        tasks = {
            "user_profile": (get_personal_evidence, (user_id,), {}),
        }
        if movie_titles:
            evidence["title_resolution"] = get_title_resolution_evidence(
                state.get("query", ""),
                state.get("movie_titles", []),
            )
            tasks["movie_opinions"] = (
                get_movie_opinions,
                ([user_id], movie_titles),
                {},
            )
        if state.get("needs_user_behavior"):
            tasks["user_rating_history"] = (get_user_rating_history, (user_id,), {})
            tasks["user_tags"] = (get_user_tags, (user_id,), {})
        if state.get("needs_recommendations"):
            tasks["recommendations"] = (
                get_recommendation_evidence,
                (user_id,),
                {"constraints": constraints},
            )
        if state.get("needs_genre_analysis"):
            tasks["genre_analysis"] = (get_blind_spot_evidence, (user_id,), {})
        evidence.update(collect_evidence_parallel(tasks))
    else:
        evidence["status"] = "current user ID missing"
    return {"evidence": evidence, **llm_answer_node({**state, "evidence": evidence})}
