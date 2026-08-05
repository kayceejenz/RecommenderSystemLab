"""User-based collaborative filtering recommender."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from scipy import sparse
from recommendation_lab.recommenders.base import BaseRecommender


class UserBasedRecommender(BaseRecommender):
    """Recommend items liked by users whose opinions are similar.

    Collaborative filtering replaces hand-crafted item features with the
    structure of the user-item interaction matrix. User-based collaborative
    filtering is the nearest-neighbor member of that family: it finds the
    users most similar to a target user, then scores an item by how those
    neighbors rated it.

    Similarity is the Pearson correlation over the items two users have both
    rated. Users that share only a handful of co-rated items produce noisy,
    near-perfect correlations, so a minimum co-rated count (min_co_ratings)
    is enforced. By default every co-rated item contributes equally; with
    weighting="popularity" each item's contribution is scaled inversely to
    how many users rated it, so opinions about a popular movie count less
    and the correlation reflects agreement on the population as a whole.

    Prediction is a deviation-from-mean weighted average: the target user's
    mean rating plus the neighbors' mean-centered ratings weighted by their
    similarity. Items no neighbor has rated fall back to the global mean.
    """

    def __init__(
        self,
        k_neighbors: int = 5, # how many closest "taste-twins" to trust per user
        min_co_ratings: int = 5, # min shared movies required to trust a similarity score
        weighting: str = "none", # "popularity" turns down volume on blockbuster items
        name: str | None = None, # optional model name for bookkeeping
    ) -> None:
        super().__init__(name) # let the base class store the name
        if weighting not in {"none", "popularity"}: # validate the weighting option up front
            raise ValueError(
                f"weighting must be 'none' or 'popularity', got {weighting!r}"
            )
        self.k_neighbors = k_neighbors # how many twins to trust per user
        self.min_co_ratings = min_co_ratings # min shared movies before trusting a similarity
        self.weighting = weighting # the weighting mode chosen

        self.user_ids: np.ndarray | None = None # real user IDs
        self.item_ids: np.ndarray | None = None # real movie IDs
        self.user_mean: np.ndarray | None = None # each user's average rating
        self.global_mean: float | None = None # fallback average across everyone
        self._centered: sparse.csr_matrix | None = None # ratings minus each user's mean
        self._binary: sparse.csr_matrix | None = None # did-they-rate-it-at-all grid
        self._neighbor_idx: np.ndarray | None = None # each user's top-k twin IDs
        self._neighbor_sim: np.ndarray | None = None # each user's top-k twin scores

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

        counts = np.asarray(ratings.getnnz(axis=1)).ravel() # how many movies each user rated
        user_mean = np.asarray(ratings.sum(axis=1)).ravel() / counts # each user's average rating ("grumpiness")
        item_counts = np.asarray(ratings.getnnz(axis=0)).ravel() # how many users rated each movie (popularity)

        centered = ratings.copy().astype(np.float32) # copy so we don't mutate the original grid
        user_of_row = np.repeat(np.arange(len(self.user_ids)), np.diff(ratings.indptr))  # ^ maps each stored rating in .data back to which user it belongs to
        centered.data -= user_mean[user_of_row] # subtract that user's average to "above/below normal" scale

        w = (
            1.0 / item_counts.astype(np.float32) # rare movies get a louder voice, popular ones quieter
            if self.weighting == "popularity"
            else np.ones(len(self.item_ids), dtype=np.float32) # no reweighting: every movie counts equally
        )
        weighted = centered.multiply(w[None, :]) # apply per-movie weight across every user's row

        sim = self._similarity(weighted, centered, ratings) # user x user "how twin-like" grid
        sim = self._enforce_min_co_ratings(sim, ratings) # zero out comparisons with too little overlap
        np.fill_diagonal(sim, 0.0) # a user is never their own twin

        self._neighbor_idx = np.full(
            (len(self.user_ids), self.k_neighbors), -1, dtype=np.int64
        ) # each row: this user's top-k twin IDs, -1 means "no twin here"
        self._neighbor_sim = np.zeros((len(self.user_ids), self.k_neighbors)) # matching similarity scores
        for u in range(len(self.user_ids)):
            row = sim[u] # this user's similarity to every other user
            k = min(self.k_neighbors, len(row)) # can't pick more neighbors than users that exist
            if k == 0:
                continue # no one to compare against, skip
            top = np.argpartition(-row, k - 1)[:k] # quickly grab the top-k most similar (unsorted)
            top = top[np.argsort(-row[top])] # now sort just those k, best twin first
            self._neighbor_idx[u] = -1 # reset row before filling
            self._neighbor_idx[u, :k] = top # store this user's top-k twin IDs
            self._neighbor_sim[u] = 0.0 # reset row before filling
            self._neighbor_sim[u, :k] = row[top] # store matching similarity scores

        self.user_mean = user_mean.astype(np.float32) # save each user's average for prediction time
        self._centered = centered.tocsr() # save mean-centered ratings for prediction time
        self._binary = (ratings > 0).astype(np.float32).tocsr() # save did-they-rate-it-at-all grid
        self._candidate_idx = np.arange(len(self.item_ids)) # full list of movie slots to recommend from
        return self

    def _similarity(
        self, weighted: sparse.csr_matrix, centered: sparse.csr_matrix, ratings: sparse.csr_matrix
    ) -> np.ndarray:
        """Pearson correlation between every pair of users.

        Each term of the correlation is restricted to items both users rated:
        the centered matrices are zero at unrated cells, so a matrix product
        accumulates only co-rated contributions. The weighted matrix is used
        for both the covariance and the per-user variance when popularity
        weighting is on.
        """
        w2 = weighted.multiply(weighted) # weighted-centered values, squared
        binary = (ratings > 0).astype(np.float32) # did-they-rate-it-at-all grid

        # numerator: agreement between every pair of users,
        # only accumulating shared (co-rated) movies because unrated cells are zero.
        numer = (weighted @ weighted.T).toarray()
        
        # denominator = sqrt(sum_w x_ui^2 * sum_w x_vi^2) over co-rated items.
        var_a = (w2 @ binary.T).toarray() # user A's opinion "spread" restricted to user B's rated movies
        var_b = (binary @ w2.T).toarray() # user B's opinion "spread" restricted to user A's rated movies
        denom = np.sqrt(var_a * var_b) # normalizing term for the correlation
        np.maximum(denom, 1e-9, out=denom) # avoid divide-by-zero for pairs with no real overlap
        return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-9)
        # ^ similarity = agreement / spread, 0 where not computable

    def _enforce_min_co_ratings(self, sim: np.ndarray, ratings: sparse.csr_matrix) -> np.ndarray:
        binary = (ratings > 0).astype(np.float32) # did-they-rate-it-at-all grid
        common = (binary @ binary.T).toarray() # how many movies each pair of users both rated
        sim[common < self.min_co_ratings] = 0.0 # too few shared movies -> don't trust this similarity
        return sim

    def predict(self, user_id: int, movie_id: int) -> float:
        self._require_fit() # must be trained before predicting
        if self.user_ids is None or self.item_ids is None:
            return self.global_mean # not trained -> fallback average
        if user_id not in self.user_ids or movie_id not in self.item_ids:
            return float(self.global_mean) # unseen user or movie to fallback average
        u = int(np.where(self.user_ids == user_id)[0][0]) # translate real user ID to row slot
        i = int(np.where(self.item_ids == movie_id)[0][0]) # translate real movie ID to column slot
        return float(self._user_scores(u)[i]) # look up this user's predicted score for this movie

    def _user_scores(self, u: int) -> np.ndarray:
        """Predicted rating for every item, cached per user.

        The deviation formula is applied to all items in one pass: the
        numerator accumulates each neighbor's mean-centered ratings weighted
        by similarity, and the denominator accumulates the absolute weights so
        negative correlations still count. Items no neighbor rated fall back
        to the global mean.
        """
        cached = getattr(self, "_score_cache", None) # grab the cache (may not exist yet)
        if cached is not None and cached[0] == u:
            return cached[1] # reuse last computed scores if same user asked again

        neighbors = self._neighbor_idx[u] # this user's top-k twin IDs
        sims = self._neighbor_sim[u] # this user's top-k twin similarity scores
        valid = neighbors[sims > 0] # drop empty slots / untrusted (zeroed) twins
        if len(valid) == 0:
            scores = np.full(len(self.item_ids), self.global_mean) # no trustworthy twins to guess global average
        else:
            s = sims[sims > 0] # similarity weights for valid twins
            numer = s @ self._centered[valid].toarray()
            # ^ for each movie: sum of (twin similarity x twin's centered rating)
            denom = np.abs(s) @ self._binary[valid].toarray()
            # ^ for each movie: sum of similarity weights that actually contributed (normalizer)
            scores = np.full(len(self.item_ids), self.global_mean, dtype=np.float64) # start with fallback everywhere
            ok = denom > 0 # movies at least one twin rated
            scores[ok] = self.user_mean[u] + numer[ok] / denom[ok]
            # ^ this user's own average + twins' weighted deviation to final predicted rating
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
        if not np.any(self._neighbor_sim[u] > 0):
            return [] # no trustworthy twins to nothing meaningful to suggest
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
        if self._centered is None or self._neighbor_idx is None:
            raise RuntimeError("fit() must be called before predict() or recommend()")
