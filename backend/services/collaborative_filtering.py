"""Explainable user-user collaborative filtering on rating overlap."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

from backend.data_layer.repository import MovieRepository

MIN_OVERLAP = 2
SHRINKAGE = 5
POSITIVE_RATING = 4.0


@dataclass(frozen=True, slots=True)
class SimilarUser:
    user_id: int
    similarity: float
    overlap_count: int


@dataclass(frozen=True, slots=True)
class SimilarUsersOpinion:
    movie_id: int
    similar_users_count: int
    rated_users_count: int
    average_rating: float | None
    positive_ratio: float | None
    rating_distribution: dict[float, int]


class CollaborativeFilteringService:
    """Find neighbors from co-rated movies using shrunk centered cosine similarity."""

    def __init__(self, repository: MovieRepository) -> None:
        self._repository = repository
        self._ratings_by_user = {
            user_id: {rating.movie_id: rating.rating for rating in repository.get_user_ratings(user_id) or ()}
            for user_id in repository.get_user_ids()
        }
        self._means = {user_id: sum(values.values()) / len(values) for user_id, values in self._ratings_by_user.items()}

    def find_similar_users(self, user_id: int, top_k: int = 30) -> list[SimilarUser]:
        target = self._ratings_by_user.get(user_id)
        if target is None or top_k <= 0:
            return []
        target_mean = self._means[user_id]
        neighbors: list[SimilarUser] = []
        for other_id, other in self._ratings_by_user.items():
            if other_id == user_id:
                continue
            common = target.keys() & other.keys()
            overlap = len(common)
            if overlap < MIN_OVERLAP:
                continue
            numerator = sum((target[movie] - target_mean) * (other[movie] - self._means[other_id]) for movie in common)
            target_norm = sqrt(sum((target[movie] - target_mean) ** 2 for movie in common))
            other_norm = sqrt(sum((other[movie] - self._means[other_id]) ** 2 for movie in common))
            if not target_norm or not other_norm:
                continue
            correlation = numerator / (target_norm * other_norm)
            similarity = correlation * (overlap / (overlap + SHRINKAGE))
            if similarity > 0:
                neighbors.append(SimilarUser(other_id, round(similarity, 4), overlap))
        return sorted(neighbors, key=lambda item: (-item.similarity, -item.overlap_count, item.user_id))[:top_k]

    def get_similar_users_opinion(self, user_id: int, movie_id: int, top_k: int = 30) -> SimilarUsersOpinion | None:
        if user_id not in self._ratings_by_user or self._repository.get_movie(movie_id) is None:
            return None
        neighbors = self.find_similar_users(user_id, top_k)
        ratings = [self._ratings_by_user[neighbor.user_id][movie_id] for neighbor in neighbors if movie_id in self._ratings_by_user[neighbor.user_id]]
        if not ratings:
            return SimilarUsersOpinion(movie_id, len(neighbors), 0, None, None, {})
        distribution = dict(sorted(Counter(ratings).items()))
        return SimilarUsersOpinion(
            movie_id=movie_id,
            similar_users_count=len(neighbors),
            rated_users_count=len(ratings),
            average_rating=round(sum(ratings) / len(ratings), 3),
            positive_ratio=round(sum(rating >= POSITIVE_RATING for rating in ratings) / len(ratings), 3),
            rating_distribution=distribution,
        )
