#!/usr/bin/env python3
"""Run reproducible temporal offline and qualitative recommendation evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_layer.loader import MovieLensDataLoader
from backend.evaluation.offline import OfflineEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=None, help="Optional deterministic cap for a quicker run.")
    parser.add_argument("--min-ratings", type=int, default=10)
    args = parser.parse_args()
    evaluator = OfflineEvaluator(MovieLensDataLoader().load(), min_ratings=args.min_ratings)
    metrics = evaluator.evaluate(max_users=args.max_users)
    print("Temporal offline evaluation")
    print("Relevance definition: held-out explicit rating >= 4.0 (not a dataset-provided binary label).")
    print(asdict(metrics))
    few_ratings_user = min(evaluator.split.eligible_users, key=lambda user_id: len(evaluator.repository.get_user_ratings(user_id) or ()))
    for user_id in (1, 15, 30, few_ratings_user):
        print(f"\nQualitative recommendations for user {user_id} (train-only):")
        for recommendation in evaluator.qualitative_recommendations(user_id):
            print({
                "movie_id": recommendation.movie_id,
                "title": recommendation.title,
                "score": recommendation.final_score,
                "evidence": asdict(recommendation.evidence),
            })
    print("\nKnown failure cases: cold-start users have no personalized history; sparse histories can yield weak overlap; few similar users reduce collaborative confidence; quality/popularity can favor frequently rated movies; missing or sparse plot/tag/rating data weakens retrieval and evidence.")


if __name__ == "__main__":
    main()
