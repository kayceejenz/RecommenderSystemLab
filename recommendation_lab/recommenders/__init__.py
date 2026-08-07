"""Recommender models in the Recommendation System Lab series."""

from recommendation_lab.recommenders.base import BaseRecommender
from recommendation_lab.recommenders.popularity import PopularityRecommender
from recommendation_lab.recommenders.content_based import ContentBasedRecommender
from recommendation_lab.recommenders.user_based_cf import UserBasedRecommender
from recommendation_lab.recommenders.item_based_cf import ItemBasedRecommender

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "ContentBasedRecommender",
    "UserBasedRecommender",
    "ItemBasedRecommender",
]
