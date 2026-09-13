#!/usr/bin/env python3
"""Run five representative final-demo conversations against the real dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent.graph import MovieAssistantGraph
from backend.agent.tools import MovieToolbox
from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository


def main() -> None:
    agent = MovieAssistantGraph(MovieToolbox(MovieRepository(MovieLensDataLoader().load())))
    queries = [
        "What should I watch tonight?",
        "Why would I like Inception?",
        "What do people with similar taste to mine think of Inception?",
        "I liked Toy Story but I'm tired of animated movies.",
        "What genres am I missing?",
    ]
    for query in queries:
        state = agent.invoke(1, query)
        print(f"\nQ: {query}\nIntent: {state['intent']}\nTools: {[item['tool'] for item in state['tool_results']]}\nA: {state['final_answer']}")


if __name__ == "__main__":
    main()
