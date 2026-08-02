"""End-to-end evaluation drivers."""

from __future__ import annotations

import pandas as pd

from recommendation_lab.evaluation.metrics import mae, rmse
from recommendation_lab.evaluation.ranking import (
    average_metrics,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommendation_lab.recommenders.base import BaseRecommender


def evaluate_predictions(model: BaseRecommender, test: pd.DataFrame) -> dict[str, float]:
    """RMSE and MAE of the model's rating predictions on the test set.

    Rows the model cannot score are dropped.
    """
    preds = [model.predict(row.user_id, row.movie_id) for row in test.itertuples()]
    scored = test.assign(_pred=preds).dropna(subset=["_pred"])
    return {
        "rmse": rmse(scored["rating"], scored["_pred"]),
        "mae": mae(scored["rating"], scored["_pred"]),
        "n": len(scored),
    }


def evaluate_ranking(
    model: BaseRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Ranking metrics, macro-averaged over users that have test ratings.

    For each user the candidate set is every movie minus the ones they rated in
    train, and the relevant items are the movies they rated in test. NDCG uses
    the test rating as graded gain.
    """
    all_items = pd.Index(train["movie_id"].unique()).union(test["movie_id"].unique())
    train_rated = train.groupby("user_id")["movie_id"].agg(set)

    p_k, r_k, m_k, n_k, h_k = (
        f"precision@{k}",
        f"recall@{k}",
        f"map@{k}",
        f"ndcg@{k}",
        f"hit_rate@{k}",
    )

    per_user = []
    for user_id, group in test.groupby("user_id"):
        relevant = set(group["movie_id"])
        if not relevant:
            continue
        rated = train_rated.get(user_id, set())
        candidates = [m for m in all_items if m not in rated]
        recommended = model.recommend(user_id, k=k, candidates=candidates)
        grades = group.set_index("movie_id")["rating"].to_dict()
        per_user.append(
            {
                p_k: precision_at_k(recommended, relevant, k=k),
                r_k: recall_at_k(recommended, relevant, k=k),
                m_k: map_at_k(recommended, relevant, k=k),
                n_k: ndcg_at_k(recommended, relevant, k=k, grades=grades),
                h_k: hit_rate_at_k(recommended, relevant, k=k),
            }
        )

    agg = average_metrics(per_user)
    agg["n_users"] = len(per_user)
    return agg
