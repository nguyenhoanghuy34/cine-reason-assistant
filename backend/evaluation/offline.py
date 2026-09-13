"""Temporal offline evaluation using explicit MovieLens ratings as relevance data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, log2
from statistics import mean

from backend.data_layer.loader import MovieLensData
from backend.data_layer.models import Rating
from backend.data_layer.repository import MovieRepository
from backend.services.recommendation import HybridRecommendationEngine, Recommendation

RELEVANT_RATING = 4.0


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train_ratings: tuple[Rating, ...]
    test_ratings_by_user: dict[int, tuple[Rating, ...]]
    eligible_users: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    evaluated_users: int
    eligible_users: int


def temporal_split(data: MovieLensData, *, min_ratings: int = 10, holdout_ratio: float = 0.2, max_holdout: int = 5) -> TemporalSplit:
    """Hold out each eligible user's most recent ratings, preserving earlier ratings for train."""
    by_user: dict[int, list[Rating]] = {}
    for rating in data.ratings:
        by_user.setdefault(rating.user_id, []).append(rating)
    train: list[Rating] = []
    tests: dict[int, tuple[Rating, ...]] = {}
    eligible: list[int] = []
    for user_id, ratings in by_user.items():
        ordered = sorted(ratings, key=lambda item: (item.timestamp, item.movie_id))
        if len(ordered) < min_ratings:
            train.extend(ordered)
            continue
        holdout_count = min(max_holdout, max(1, ceil(len(ordered) * holdout_ratio)))
        train.extend(ordered[:-holdout_count])
        tests[user_id] = tuple(ordered[-holdout_count:])
        eligible.append(user_id)
    return TemporalSplit(tuple(train), tests, tuple(sorted(eligible)))


class OfflineEvaluator:
    """Evaluate the existing hybrid engine against temporally held-out explicit ratings."""

    def __init__(self, data: MovieLensData, *, min_ratings: int = 10, holdout_ratio: float = 0.2, max_holdout: int = 5) -> None:
        self.split = temporal_split(data, min_ratings=min_ratings, holdout_ratio=holdout_ratio, max_holdout=max_holdout)
        # Crucial: the hybrid engine sees only ratings that precede each user's held-out block.
        self.train_data = replace(data, ratings=self.split.train_ratings)
        self.repository = MovieRepository(self.train_data)
        self.engine = HybridRecommendationEngine(self.repository)

    def evaluate(self, *, max_users: int | None = None) -> EvaluationMetrics:
        users = self.split.eligible_users[:max_users] if max_users else self.split.eligible_users
        per_user: list[tuple[float, float, float, float, float, float]] = []
        for user_id in users:
            relevant = {rating.movie_id for rating in self.split.test_ratings_by_user[user_id] if rating.rating >= RELEVANT_RATING}
            if not relevant:
                continue
            recommendations = self.engine.recommend_movies(user_id, top_k=10)
            predicted = [item.movie_id for item in recommendations]
            p5, r5, n5 = _metrics_at_k(predicted, relevant, 5)
            p10, r10, n10 = _metrics_at_k(predicted, relevant, 10)
            per_user.append((p5, r5, n5, p10, r10, n10))
        if not per_user:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, len(users))
        columns = list(zip(*per_user))
        return EvaluationMetrics(*(round(mean(column), 4) for column in columns), len(per_user), len(users))

    def qualitative_recommendations(self, user_id: int, top_k: int = 5) -> list[Recommendation]:
        """Recommendations/evidence from the train-only engine for manual inspection."""
        return self.engine.recommend_movies(user_id, top_k=top_k)


def _metrics_at_k(predicted: list[int], relevant: set[int], k: int) -> tuple[float, float, float]:
    top = predicted[:k]
    hits = [movie_id in relevant for movie_id in top]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant)
    dcg = sum(1 / log2(index + 2) for index, hit in enumerate(hits) if hit)
    ideal_dcg = sum(1 / log2(index + 2) for index in range(min(len(relevant), k)))
    return precision, recall, dcg / ideal_dcg if ideal_dcg else 0.0
