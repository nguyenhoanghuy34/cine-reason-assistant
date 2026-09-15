from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_personal_evidence
from app.agent.tools.recommendation_tools import get_recommendation_evidence
from app.agent.tools.blind_spot_tools import get_blind_spot_evidence
from app.agent.tools.related_user_tools import get_movie_opinions


def collect_evidence(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def personal_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    constraints = {
        "preferred_genres": state.get("preferred_genres", []),
        "excluded_genres": state.get("excluded_genres", []),
        "include_terms": state.get("include_terms", []),
        "exclude_terms": state.get("exclude_terms", []),
        "seed_titles": state.get("movie_titles", []),
    }
    evidence = {"user_id": user_id}
    if user_id is not None:
        evidence["user_profile"] = collect_evidence(get_personal_evidence, user_id)
        if state.get("movie_titles"):
            evidence["movie_opinions"] = collect_evidence(
                get_movie_opinions, [user_id], state["movie_titles"]
            )
        if state.get("needs_recommendations"):
            evidence["recommendations"] = collect_evidence(
                get_recommendation_evidence, user_id, constraints=constraints
            )
        if state.get("needs_genre_analysis"):
            evidence["genre_analysis"] = collect_evidence(get_blind_spot_evidence, user_id)
    else:
        evidence["status"] = "current user ID missing"
    return {"evidence": evidence, **llm_answer_node({**state, "evidence": evidence})}
