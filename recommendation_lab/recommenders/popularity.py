"""Popularity baseline recommender."""

from __future__ import annotations
from typing import Sequence
import pandas as pd
from recommendation_lab.recommenders.base import BaseRecommender

BAYESIAN_M = 10.0


class PopularityRecommender(BaseRecommender):
    """Rank movies by their popularity in the training set.

    This is the weakest baseline: every user gets the same list, so it is a
    benchmark later models should beat rather than a real recommender. It is a
    ranking model, so predict() falls back to the global mean rating.
    """

    def __init__(self, metric: str = "count", name: str | None = None) -> None:
        super().__init__(name)
        if metric not in {"count", "mean", "bayesian"}:
            raise ValueError(
                f"metric must be 'count', 'mean', or 'bayesian', got {metric!r}"
            )
        self.metric = metric
        self.popularity: pd.Series | None = None
        self.global_mean: float | None = None

    def fit(self, train: pd.DataFrame) -> BaseRecommender:
        self.train = train
        self.global_mean = train["rating"].mean()
        counts = train["movie_id"].value_counts()
        if self.metric == "count":
            self.popularity = counts
        elif self.metric == "mean":
            self.popularity = train.groupby("movie_id")["rating"].mean()
        else:
            means = train.groupby("movie_id")["rating"].mean()
            m = BAYESIAN_M
            self.popularity = (counts * means + m * self.global_mean) / (counts + m)
        self.popularity = self.popularity.sort_values(ascending=False)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        self._require_fit()
        return float(self.global_mean)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidates: Sequence[int] | None = None,
    ) -> list[int]:
        self._require_fit()
        rated = set(self.train.loc[self.train["user_id"] == user_id, "movie_id"])
        pool = self.popularity.index
        if candidates is not None:
            pool = pool.intersection(candidates)
        ranked = [m for m in pool if m not in rated]
        return ranked[:k]

    def _require_fit(self) -> None:
        if self.popularity is None:
            raise RuntimeError("fit() must be called before predict() or recommend()")
