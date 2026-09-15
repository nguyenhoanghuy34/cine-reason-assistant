from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.personal import collect_evidence
from app.agent.state import AgentState
from app.agent.tools.personal_tools import get_personal_evidence
from app.agent.tools.recommendation_tools import get_recommendation_evidence
from app.agent.tools.related_user_tools import (
    get_related_user_ids, get_top_movies_from_related_users, get_movie_opinions,
)
from app.agent.tools.similarity_tools import get_similar_users, get_user_similarity_details


def related_users_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    targets = state.get("target_user_ids", [])
    constraints = {
        "preferred_genres": state.get("preferred_genres", []),
        "excluded_genres": state.get("excluded_genres", []),
        "include_terms": state.get("include_terms", []),
        "exclude_terms": state.get("exclude_terms", []),
        "seed_titles": state.get("movie_titles", []),
    }
    evidence = {"current_user_id": user_id, "target_user_ids": targets}
    if user_id is not None:
        evidence["current_user_profile"] = collect_evidence(get_personal_evidence, user_id)
    if targets:
        evidence["other_user_profiles"] = [
            {"user_id": target, "profile": collect_evidence(get_personal_evidence, target)}
            for target in targets
        ]
        if state.get("needs_recommendations"):
            evidence["recommendations"] = [
                {"user_id": target, "data": collect_evidence(
                    get_recommendation_evidence, target, constraints=constraints
                )}
                for target in targets
            ]
        if state.get("needs_similarity"):
            pair_ids = [user_id, *targets] if user_id is not None else targets
            evidence["similarity_details"] = collect_evidence(
                get_user_similarity_details, pair_ids
            )
    elif user_id is not None:
        evidence["similar_users"] = collect_evidence(get_top_movies_from_related_users, user_id)
        if state.get("needs_similarity"):
            evidence["similarity_ranking"] = collect_evidence(get_similar_users, user_id)
        related = collect_evidence(get_related_user_ids, user_id)
        targets = related[:10] if isinstance(related, list) else []
    if state.get("movie_titles"):
        evidence["movie_opinions"] = collect_evidence(
            get_movie_opinions, targets, state["movie_titles"]
        )
    return {
        "evidence": evidence,
        "related_users_evidence": evidence,
        **llm_answer_node({**state, "evidence": evidence}),
    }
