"""Item-based collaborative filtering recommender."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from recommendation_lab.recommenders.base import BaseRecommender


class ItemBasedRecommender(BaseRecommender):
    """Recommend items similar to the ones a user already liked.

    Item-based collaborative filtering is the transpose of the user-based
    model: instead of matching users to users, it matches items to items.
    Two items are similar when the same users rated them alike, and a movie
    is scored for a user by the ratings that user gave to the movie's nearest
    item neighbors.

    Similarity is the Pearson correlation over the users who rated both
    items. Items that share only a handful of co-rating users produce noisy,
    near-perfect correlations, so a minimum co-rated count (min_co_ratings)
    is enforced. The full item-item matrix is quadratic in the number of
    movies, so the model is pruned to the k nearest neighbors per item, which
    is also where the model's size lives.

    Prediction is a deviation-from-mean weighted average: the target item's
    mean rating plus the user's ratings on similar items, mean-centered by
    their own item means and weighted by similarity. Items with no similar
    item the user has rated fall back to the global mean.
    """

    def __init__(
        self,
        k_neighbors: int = 5, # how many closest "sister movies" to trust per item
        min_co_ratings: int = 100, # min shared users required to trust a similarity score
        weighting: str = "none", # "popularity" turns down volume on prolific users
        name: str | None = None, # optional model name for bookkeeping
    ) -> None:
        super().__init__(name) # let the base class store the name
        if weighting not in {"none", "popularity"}: # validate the weighting option up front
            raise ValueError(
                f"weighting must be 'none' or 'popularity', got {weighting!r}"
            )
        self.k_neighbors = k_neighbors # how many sister movies to trust per item
        self.min_co_ratings = min_co_ratings # min shared users before trusting a similarity
        self.weighting = weighting # the weighting mode chosen

        self.user_ids: np.ndarray | None = None # real user IDs
        self.item_ids: np.ndarray | None = None # real movie IDs
        self.item_mean: np.ndarray | None = None # each movie's average rating
        self.global_mean: float | None = None # fallback average across everyone
        self._centered: sparse.csr_matrix | None = None # ratings minus each movie's mean
        self._binary: sparse.csr_matrix | None = None # did-they-rate-it-at-all grid
        self._sim: sparse.csr_matrix | None = None # pruned item x item similarity
        self._sim_abs: sparse.csr_matrix | None = None # pruned similarity, absolute values

    def fit(self, train: pd.DataFrame) -> BaseRecommender:
        self.train = train # keep a reference for bookkeeping
        self.global_mean = float(train["rating"].mean()) # overall average rating, safety-net fallback

        self.user_ids = train["user_id"].unique() # distinct real user IDs
        self.item_ids = train["movie_id"].unique() # distinct real movie IDs
        u_idx = pd.Series(np.arange(len(self.user_ids)), index=self.user_ids) # real user ID to row slot
        i_idx = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids) # real movie ID to column slot

        rows = u_idx[train["user_id"]].to_numpy() # each rating's row slot (which user)
        cols = i_idx[train["movie_id"]].to_numpy() # each rating's column slot (which movie)
        vals = train["rating"].to_numpy() # the star rating itself
        ratings = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))
        ) # sparse user x movie grid: only filled-in boxes take memory

        item_counts = np.asarray(ratings.getnnz(axis=0)).ravel() # how many users rated each movie
        user_counts = np.asarray(ratings.getnnz(axis=1)).ravel() # how many movies each user rated
        item_mean = np.asarray(ratings.sum(axis=0)).ravel() / item_counts # each movie's average rating

        centered = ratings.copy().astype(np.float32) # copy so we don't mutate the original grid
        centered.data -= item_mean[ratings.indices] # subtract that movie's average to "above/below normal" scale

        w = (
            1.0 / user_counts.astype(np.float32) # prolific users get a quieter voice, picky ones louder
            if self.weighting == "popularity"
            else np.ones(len(self.user_ids), dtype=np.float32) # no reweighting: every user counts equally
        )
        weighted = centered.multiply(w[:, None]) # apply per-user weight across every movie's column

        sim = self._similarity(weighted, centered, ratings) # item x item "how sister-like" grid
        sim = self._enforce_min_co_ratings(sim, ratings) # zero out comparisons with too little overlap
        np.fill_diagonal(sim, 0.0) # a movie is never its own sister

        # prune the quadratic item x item matrix to the k nearest neighbors per item.
        idx_rows, idx_cols, idx_data = [], [], []
        for i in range(len(self.item_ids)):
            row = sim[i] # this movie's similarity to every other movie
            k = min(self.k_neighbors, len(row)) # can't pick more neighbors than movies that exist
            if k == 0:
                continue # no one to compare against, skip
            top = np.argpartition(-row, k - 1)[:k] # quickly grab the top-k most similar (unsorted)
            top = top[np.argsort(-row[top])] # now sort just those k, best sister first
            idx_rows.extend([i] * len(top)) # record the source movie for each kept edge
            idx_cols.extend(top.tolist()) # record the sister movie for each kept edge
            idx_data.extend(row[top].tolist()) # record the similarity for each kept edge
        self._sim = sparse.csr_matrix(
            (idx_data, (idx_rows, idx_cols)), shape=(len(self.item_ids), len(self.item_ids))
        ) # sparse pruned graph: each row has at most k nonzero similarities
        self._sim_abs = self._sim.copy() # absolute similarities for the normalizer
        self._sim_abs.data = np.abs(self._sim_abs.data)

        self.item_mean = item_mean.astype(np.float32) # save each movie's average for prediction time
        self._centered = centered.tocsr() # save mean-centered ratings for prediction time
        self._binary = (ratings > 0).astype(np.float32).tocsr() # save did-they-rate-it-at-all grid
        self._candidate_idx = np.arange(len(self.item_ids)) # full list of movie slots to recommend from
        return self

    def _similarity(
        self, weighted: sparse.csr_matrix, centered: sparse.csr_matrix, ratings: sparse.csr_matrix
    ) -> np.ndarray:
        """Pearson correlation between every pair of items.

        Each term of the correlation is restricted to users who rated both
        items: the centered matrices are zero at unrated cells, so a matrix
        product on the transposed grid accumulates only co-rated
        contributions. The weighted matrix is used for both the covariance
        and the per-item variance when popularity weighting is on.
        """
        w2 = weighted.multiply(weighted) # weighted-centered values, squared
        binary = (ratings > 0).astype(np.float32) # did-they-rate-it-at-all grid

        # numerator: agreement between every pair of movies,
        # only accumulating shared (co-rated) users because unrated cells are zero.
        numer = (weighted.T @ weighted).toarray()
        
        # denominator = sqrt(sum_w x_ui^2 * sum_w x_vi^2) over co-rated users.
        var_a = (w2.T @ binary).toarray() # item A's opinion "spread" restricted to item B's raters
        var_b = (binary.T @ w2).toarray() # item B's opinion "spread" restricted to item A's raters
        denom = np.sqrt(var_a * var_b) # normalizing term for the correlation
        np.maximum(denom, 1e-9, out=denom) # avoid divide-by-zero for pairs with no real overlap
        return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-9)
        # ^ similarity = agreement / spread, 0 where not computable

    def _enforce_min_co_ratings(self, sim: np.ndarray, ratings: sparse.csr_matrix) -> np.ndarray:
        binary = (ratings > 0).astype(np.float32) # did-they-rate-it-at-all grid
        common = (binary.T @ binary).toarray() # how many users rated each pair of movies
        sim[common < self.min_co_ratings] = 0.0 # too few shared users -> don't trust this similarity
        return sim

    def predict(self, user_id: int, movie_id: int) -> float:
        self._require_fit() # must be trained before predicting
        if self.user_ids is None or self.item_ids is None:
            return self.global_mean # not trained to fallback average
        if user_id not in self.user_ids or movie_id not in self.item_ids:
            return float(self.global_mean) # unseen user or movie to fallback average
        u = int(np.where(self.user_ids == user_id)[0][0]) # translate real user ID to row slot
        i = int(np.where(self.item_ids == movie_id)[0][0]) # translate real movie ID to column slot
        return float(self._user_scores(u)[i]) # look up this user's predicted score for this movie

    def _user_scores(self, u: int) -> np.ndarray:
        """Predicted rating for every item, cached per user.

        The deviation formula is applied to all items in one pass. The item
        graph is sparse, so a single sparse matrix-vector product gathers,
        for every movie, the user's ratings on that movie's neighbors weighted
        by similarity; the denominator accumulates the absolute weights so
        negative correlations still count. Items with no rated neighbor fall
        back to the global mean.
        """
        cached = getattr(self, "_score_cache", None) # grab the cache (may not exist yet)
        if cached is not None and cached[0] == u:
            return cached[1] # reuse last computed scores if same user asked again

        row_c = self._centered[u] # this user's ratings minus each movie's mean
        row_b = self._binary[u] # this user's did-they-rate-it-at-all row
        numer = (self._sim @ row_c.T).toarray().ravel()
        # ^ for each movie: sum of (similarity x user's centered rating) over sister movies the user rated
        denom = (self._sim_abs @ row_b.T).toarray().ravel()
        # ^ for each movie: sum of similarity weights that actually contributed (normalizer)
        scores = np.full(len(self.item_ids), self.global_mean, dtype=np.float64) # start with fallback everywhere
        ok = denom > 0 # movies with at least one rated sister movie
        scores[ok] = self.item_mean[ok] + numer[ok] / denom[ok]
        # ^ that movie's own average + sister ratings' weighted deviation to final predicted rating
        self._score_cache = (u, scores) # cache for repeat calls with the same user
        return scores

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidates: Sequence[int] | None = None,
    ) -> list[int]:
        self._require_fit() # must be trained before recommending
        if self.user_ids is None or self.item_ids is None:
            return [] # not trained to nothing to recommend
        if user_id not in self.user_ids:
            return [] # unknown user to nothing to recommend
        u = int(np.where(self.user_ids == user_id)[0][0]) # translate real user ID to row slot
        if self._binary[u].nnz == 0:
            return [] # user rated nothing, so no sister movies to lean on
        scores = self._user_scores(u) # predicted score for every movie

        # rank candidates, excluding items the user already rated in train.
        rated = set(self._binary[u].indices) # movies this user has already watched
        if candidates is not None:
            cand = set(candidates)
            pool = [i for i in self._candidate_idx if self.item_ids[i] in cand] # restrict to given candidate list
        else:
            pool = list(self._candidate_idx) # consider every movie
        pool = [i for i in pool if i not in rated] # drop movies already watched
        order = sorted(pool, key=lambda i: scores[i], reverse=True) # best predicted score first
        return [int(self.item_ids[i]) for i in order[:k]] # top-k real movie IDs

    def _require_fit(self) -> None:
        if self._centered is None or self._sim is None:
            raise RuntimeError("fit() must be called before predict() or recommend()")
