from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.personal import collect_evidence_parallel
from app.agent.state import AgentState
from app.agent.tools.movie_data_tools import (
    get_title_resolution_evidence,
    get_movie_info,
    get_movie_summary,
    resolve_movie_titles,
)


def general_node(state: AgentState) -> AgentState:
    evidence = {}
    titles = resolve_movie_titles(state.get("query", ""), state.get("movie_titles", []))
    tasks = {}

    if titles:
        evidence["title_resolution"] = get_title_resolution_evidence(
            state.get("query", ""),
            state.get("movie_titles", []),
        )
    if titles and state.get("needs_movie_info"):
        tasks["movie_info"] = (get_movie_info, (titles,), {})
    if titles and state.get("needs_movie_summary"):
        tasks["movie_summary"] = (get_movie_summary, (titles,), {})

    evidence.update(collect_evidence_parallel(tasks))
    return {"evidence": evidence, **llm_answer_node({**state, "evidence": evidence})}
