#!/usr/bin/env python3
"""Print five explainable hybrid recommendations for User 1."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.repository import MovieRepository
from backend.services.recommendation import HybridRecommendationEngine


def main() -> None:
    engine = HybridRecommendationEngine(MovieRepository(MovieLensDataLoader().load()))
    for item in engine.recommend_movies(1, top_k=5):
        print(f"{item.title} ({item.year}) | final_score={item.final_score} | genres={list(item.genres)}")
        print(f"  evidence={item.evidence}")


if __name__ == "__main__":
    main()
