"""Deterministic business services built on the repository boundary."""

from .collaborative_filtering import CollaborativeFilteringService, SimilarUser, SimilarUsersOpinion
from .movie_retrieval import MovieRetrievalService, SearchResult
from .recommendation import HybridRecommendationEngine, Recommendation, RecommendationEvidence
from .user_profile import GenrePreference, UserProfileService, UserTasteProfile

__all__ = [
    "CollaborativeFilteringService",
    "GenrePreference",
    "HybridRecommendationEngine",
    "MovieRetrievalService",
    "Recommendation",
    "RecommendationEvidence",
    "SearchResult",
    "SimilarUser",
    "SimilarUsersOpinion",
    "UserProfileService",
    "UserTasteProfile",
]
