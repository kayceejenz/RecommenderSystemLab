"""Shared recommender interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd


class BaseRecommender(ABC):
    """Interface every recommender implements.

    A recommender learns from a ratings DataFrame with at least the columns
    user_id, movie_id, and rating, then either predicts a rating for a
    (user, movie) pair or recommends a ranked list of movies for a user.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__
        self.train: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> BaseRecommender:
        """Learn model parameters from the training ratings."""

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating user_id would give movie_id."""

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidates: Sequence[int] | None = None,
    ) -> list[int]:
        """Return the top-k movie_ids for user_id.

        candidates restricts the search space; when None, all movies known to
        the model are candidates. Movies the user rated in train are never
        recommended.
        """
