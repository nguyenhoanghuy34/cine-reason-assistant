from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.personal import collect_evidence, collect_evidence_parallel
from app.agent.state import AgentState
from app.agent.tools.movie_data_tools import get_title_resolution_evidence, resolve_movie_titles
from app.agent.tools.personal_tools import get_personal_evidence
from app.agent.tools.recommendation_tools import get_recommendation_evidence
from app.agent.tools.related_user_tools import (
    get_related_user_ids,
    get_top_movies_from_related_users,
    get_movie_opinions,
)
from app.agent.tools.similarity_tools import get_similar_users, get_user_similarity_details


def related_users_node(state: AgentState) -> AgentState:
    user_id = state.get("user_id")
    targets = state.get("target_user_ids", [])
    movie_titles = resolve_movie_titles(state.get("query", ""), state.get("movie_titles", []))
    constraints = {
        "preferred_genres": state.get("preferred_genres", []),
        "excluded_genres": state.get("excluded_genres", []),
        "include_terms": state.get("include_terms", []),
        "exclude_terms": state.get("exclude_terms", []),
        "seed_titles": movie_titles,
    }
    evidence = {"current_user_id": user_id, "target_user_ids": targets}
    tasks = {}

    if user_id is not None:
        tasks["current_user_profile"] = (get_personal_evidence, (user_id,), {})

    if targets:
        profiles = collect_evidence_parallel({
            target: (get_personal_evidence, (target,), {})
            for target in targets
        })
        evidence["other_user_profiles"] = [
            {"user_id": target, "profile": profiles[target]}
            for target in targets
        ]

        if state.get("needs_recommendations"):
            recommendations = collect_evidence_parallel({
                target: (
                    get_recommendation_evidence,
                    (target,),
                    {"constraints": constraints},
                )
                for target in targets
            })
            evidence["recommendations"] = [
                {"user_id": target, "data": recommendations[target]}
                for target in targets
            ]

        if state.get("needs_similarity"):
            pair_ids = [user_id, *targets] if user_id is not None else targets
            tasks["similarity_details"] = (
                get_user_similarity_details,
                (pair_ids,),
                {},
            )

    elif user_id is not None:
        tasks["similar_users"] = (get_top_movies_from_related_users, (user_id,), {})
        if state.get("needs_similarity"):
            tasks["similarity_ranking"] = (get_similar_users, (user_id,), {})
        related = collect_evidence(get_related_user_ids, user_id)
        targets = related[:10] if isinstance(related, list) else []

    if movie_titles:
        evidence["title_resolution"] = get_title_resolution_evidence(
            state.get("query", ""),
            state.get("movie_titles", []),
        )
        tasks["movie_opinions"] = (
            get_movie_opinions,
            (targets, movie_titles),
            {},
        )

    evidence.update(collect_evidence_parallel(tasks))
    return {
        "evidence": evidence,
        "related_users_evidence": evidence,
        **llm_answer_node({**state, "evidence": evidence}),
    }
