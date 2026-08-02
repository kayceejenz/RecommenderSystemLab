"""Train/test splitting utilities."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from recommendation_lab.config import DEFAULT_SEED


@dataclass
class Split:
    """A train/test split of rating data."""

    train: pd.DataFrame
    test: pd.DataFrame


def _filter_cold(split: Split) -> Split:
    """Drop test rows whose movie is not rated in the training set."""
    trained = split.train["movie_id"].unique()
    test = split.test[split.test["movie_id"].isin(trained)]
    return Split(train=split.train, test=test)


def random_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = DEFAULT_SEED,
    filter_cold: bool = True,
) -> Split:
    """Split per user at random with a fixed seed.

    Each user keeps a proportional share of their ratings in train and test,
    so every user appears in both.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, group in ratings.groupby("user_id"):
        rows = group.index.to_numpy()
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * (1 - test_ratio)))
        train_idx.extend(rows[:cut])
        test_idx.extend(rows[cut:])
    split = Split(
        train=ratings.loc[train_idx],
        test=ratings.loc[test_idx],
    )
    return _filter_cold(split) if filter_cold else split


def time_base_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    filter_cold: bool = True,
) -> Split:
    """Split per user by time: the latest fraction of each user's ratings is test.

    Ratings are sorted by timestamp with a stable tiebreak on the row index,
    so the split is deterministic. Every user contributes to train, and every
    test rating is strictly later than that user's train ratings.
    """
    ordered = ratings.assign(_row=range(len(ratings))).sort_values(
        ["user_id", "timestamp", "_row"], kind="stable"
    )
    train_parts, test_parts = [], []
    for _, group in ordered.groupby("user_id", sort=False):
        cut = max(1, int(len(group) * (1 - test_ratio)))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    train = pd.concat(train_parts).drop(columns="_row").sort_index()
    test = pd.concat(test_parts).drop(columns="_row").sort_index()
    split = Split(train=train, test=test)
    return _filter_cold(split) if filter_cold else split
