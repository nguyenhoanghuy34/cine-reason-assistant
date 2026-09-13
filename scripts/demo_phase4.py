#!/usr/bin/env python3
"""Run the required Phase 4 conversation examples without an external LLM."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent.graph import MovieAssistantGraph
from backend.agent.tools import MovieToolbox
from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


def main() -> None:
    agent = MovieAssistantGraph(MovieToolbox(MovieRepository(MovieLensDataLoader().load())))
    examples = [
        (1, "What should I watch tonight?"),
        (1, "Why would I like Inception?"),
        (1, "What do people with similar taste to mine think of Inception?"),
        (1, "I liked Toy Story but I'm tired of animated movies."),
        (1, "I want a dark psychological thriller with a twist."),
        (1, "What genres am I missing?"),
        (1, "Compare Inception and The Matrix for me."),
        (-1, "I haven't rated any movies. What should I watch?"),
    ]
    for user_id, query in examples:
        state = agent.invoke(user_id, query)
        print(f"\nQ: {query}\nIntent: {state['intent']}\nTools: {[item['tool'] for item in state['tool_results']]}\nA: {state['final_answer']}")


if __name__ == "__main__":
    main()
