"""Content-based recommender using movie genres."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from recommendation_lab.recommenders.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Recommend movies similar to what a user already likes.

    Every movie is described by a genre vector, IDF-weighted so rare genres
    count more than ubiquitous ones. A user profile
    is the rating-weighted average of the genre vectors of the movies they
    rated in train, and recommendation ranks candidates by cosine similarity
    to that profile. Because only item features are used, movies with no
    ratings at all can still be recommended, which popularity cannot do.
    """

    def __init__(
        self,
        genre_weighting: str = "idf",
        center: float = 3.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        
        if genre_weighting not in {"idf", "binary"}:
            raise ValueError(
                f"genre_weighting must be 'idf' or 'binary', got {genre_weighting!r}"
            )
            
        self.genre_weighting = genre_weighting
        self.center = center
        self.genres: list[str] | None = None
        self.item_vectors: pd.DataFrame | None = None
        self.user_vectors: pd.DataFrame | None = None
        self.global_mean: float | None = None

    def fit(
        self,
        train: pd.DataFrame,
        items: pd.DataFrame | None = None,
    ) -> BaseRecommender:
        """Build genre and user profiles.

        items must be a DataFrame with movie_id and a genres column of lists;
        it supplies the content features the ratings alone do not carry.
        """
        self.train = train
        self.global_mean = train["rating"].mean()
        
        if items is None:
            raise ValueError("fit() requires the movies table with a 'genres' column")

        # get every distinct genre across the entire catalog, sorted; this is the feature vocabulary.
        genres = sorted({g for gl in items["genres"] for g in gl})
        self.genres = genres
        
        # count the numbers of document in which the term (i.e. genre) occurred (df)
        doc_freq = pd.Series(0.0, index=genres)
        for gl in items["genres"]:
            for g in gl:
                doc_freq[g] += 1.0
        
        # number of documents (N)
        n_movies = len(items)

        # assigns each genre in a movie its IDF weight log(N/df) (or 1.0 in binary mode).  a genre appears at most once per movie.
        rows, index = [], []
        for movie_id, gl in zip(items["movie_id"], items["genres"]):
            vec = np.zeros(len(genres))
            for g in gl:
                # formula: IDF: log(N/df) if using IDF mode else Binary mode simply sets the weight to 1.0.
                weight = (
                    np.log(n_movies / doc_freq[g])
                    if self.genre_weighting == "idf"
                    else 1.0
                )
                
                vec[genres.index(g)] = weight
            rows.append(vec)
            index.append(movie_id)
            
        # get all the vectors of all the items
        vectors = np.asarray(rows, dtype=float)
        
        # Norm is a linear algebra function used to calcuate the sum of a vector (NORM 1) and the length of each vector(NORM 2)
        # np.linalg.norm with no ord uses the L2 (Euclidean) norm, the length of each row. L1 (sum of abs values) is described but not used here.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        # calculate the item vector by dividing the vectors with the norm
        # formulation: w(t,d) = tf log[N/df] / Root(Sum(tf)^2 log([N/df])^2)
        self.item_vectors = pd.DataFrame(
            vectors / norms, index=pd.Index(index, name="movie_id"), columns=genres
        )

        merged = train[["user_id", "movie_id", "rating"]].merge(
            self.item_vectors.reset_index(),
            on="movie_id",
            how="inner",
        )
        
        # calculate the weight of each genre based on the rating on the training data and set our center as a threshold
        # assign weight of each genre based on the rating on the training data. mean-centers each rating by self.center (3.0)
        # ratings above center add genre weight, below subtract.
        weighted = merged[genres].multiply(merged["rating"] - self.center, axis=0)
        
        # get the vector sum of all user's rated movies
        profile = weighted.groupby(merged["user_id"]).sum()
        
        # calculate the norms for the users vector
        norms = np.linalg.norm(profile.to_numpy(), axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        # calculate user vector using same approach
        self.user_vectors = profile / norms
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
        if user_id not in self.user_vectors.index:
            return []
         
        # get user vector
        profile = self.user_vectors.loc[user_id].to_numpy()
        
        # get user's rated movies based on the training data
        rated = set(self.train.loc[self.train["user_id"] == user_id, "movie_id"])
        pool = self.item_vectors.index
        
        # if a candidates set is supplied, restrict the pool to items in it (the evaluation passes all items minus the user's train items).
        if candidates is not None:
            pool = pool.intersection(candidates)
            
        # drops movies the user already rated in train; returns [] only if every candidate has already been seen.
        pool = [m for m in pool if m not in rated]
        if not pool:
            return []
        
        # scores every candidate as the dot product of its unit vector with the user profile, equal to cosine similarity.
        scores = self.item_vectors.loc[pool].to_numpy() @ profile
        
        # sort the scores by descending order
        order = np.argsort(-scores)
        
        # return the pool
        return [pool[i] for i in order[:k]]

    def _require_fit(self) -> None:
        if self.item_vectors is None or self.user_vectors is None:
            raise RuntimeError("fit() must be called before predict() or recommend()")
