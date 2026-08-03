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
        
        # calculate average of all the ratings in our training data (GM)
        self.global_mean = train["rating"].mean()
        
        # count each movie(N)
        counts = train["movie_id"].value_counts()
        
        # assign popularity based on the counts if metric is count
        if self.metric == "count":
            self.popularity = counts
            
        # assign the popularity based on the average rating of each movie
        elif self.metric == "mean": 
            self.popularity = train.groupby("movie_id")["rating"].mean()
            
        # a Bayesian average (shrinkage toward the global mean). Each movie's score pulls its observed mean toward the global mean by an amount controlled by m.
        else:
            # means = each movie's average rating in train.(M)
            means = train.groupby("movie_id")["rating"].mean()
            
            
            m = BAYESIAN_M
            
            # formulation: (N x M + m x GM) / (N + m)
            self.popularity = (counts * means + m * self.global_mean) / (counts + m)
            
        # order by descending
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
        
        # if a candidates set is supplied, restrict the pool to items in it (the evaluation passes all items minus the user's train items).
        if candidates is not None:
            pool = pool.intersection(candidates)
            
        # exclude movies the user has already seen (rated in train)
        ranked = [m for m in pool if m not in rated]
        return ranked[:k]

    def _require_fit(self) -> None:
        if self.popularity is None:
            raise RuntimeError("fit() must be called before predict() or recommend()")
