from app.agent.nodes.llm_answer import llm_answer_node
from app.agent.nodes.personal import collect_evidence
from app.agent.state import AgentState
from app.agent.tools.movie_data_tools import get_movie_info, get_movie_summary


def general_node(state: AgentState) -> AgentState:
    evidence = {}
    titles = state.get("movie_titles", [])

    if titles and state.get("needs_movie_info"):
        evidence["movie_info"] = collect_evidence(get_movie_info, titles)
    if titles and state.get("needs_movie_summary"):
        evidence["movie_summary"] = collect_evidence(get_movie_summary, titles)

    return {"evidence": evidence, **llm_answer_node({**state, "evidence": evidence})}
