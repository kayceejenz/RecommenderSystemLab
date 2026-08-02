# RecommendationSystemLab

## Table of Contents

- [Overview](#overview)
- [Scope of the Work](#scope-of-the-work)
- [Roadmap](#roadmap)
- [Data Ingestion](#data-ingestion)
- [Popularity Model](#popularity-model)

## Overview

This is a personal experiment exploring the evolution of recommendation systems.The goal is to understand not just how these algorithms work, but why each one exists and what limitation it addresses.

Every stage answers the same question: what limitation of the previous algorithm are we solving?

All stages share one data pipeline, one evaluation framework, and one set of metrics, so results stay comparable across the lab.

## Scope of the Work

- MovieLens 1M, with 1,000,209 ratings from 6,040 users over 3,883 movies.
- Seven algorithm families implemented from scratch, from popularity baselines to hybrid systems.
- A shared evaluation framework: RMSE and MAE for rating prediction, precision, recall, MAP, NDCG, and hit rate for ranking.
- Time-based train/test splits that eliminate temporal leakage, which a random split would otherwise introduce.

## Roadmap

1. **Popularity Recommendation.** The baseline. It requires no user history and no learning, and establishes the structure every subsequent recommender builds on.

2. **Content-Based Recommendation.** Popularity ignores what a user actually likes. Content based filtering exploits item features to recommend items similar to ones a user has already engaged with, and handles the new item cold start problem.

3. **User-Based Collaborative Filtering.** Content based requires rich features and only surfaces similar items. User based collaborative filtering finds structure directly in the ratings, discovering taste by looking at users with similar preferences.

4. **Item-Based Collaborative Filtering.** User based filtering does not scale, because similarity must be computed per query. Item based filtering precomputes item to item similarities offline, making online serving fast and practical.

5. **Matrix Factorization.** Neighborhood methods struggle with sparse rating matrices and cannot generalize to unseen combinations. Matrix factorization learns compact latent factors that capture hidden taste dimensions.

6. **Implicit Feedback Recommendation.** Explicit ratings are rare, costly to obtain, and biased. Implicit signals such as views, clicks, and plays are abundant, and reframe the problem as preference ranking rather than rating prediction.

7. **Hybrid Recommendation.** Every method has blind spots. Hybrids combine complementary approaches to mitigate cold start, popularity bias, and sparsity, and are closer to what production systems actually ship.

## Data Ingestion

Ratings flow through three modules before any recommender sees them.

`downloader.py` maintains a registry of datasets (ml-1m and ml-10m) and downloads them on demand. The download is idempotent, so rerunning it never duplicates data. Run it directly with `python -m recommendation_lab.data.downloader ml-1m`.

`loader.py` parses the MovieLens `.dat` files into three tables: ratings, users, and movies, with genres split into a list. It handles the files' `::` separator and latin-1 encoding.

`split.py` provides two per-user splits, random and time-based. The time-based split keeps each user's latest ratings for test, eliminating temporal leakage entirely. On this dataset a random split leaks about half of its test rows, which would inflate every later evaluation.

Notebooks: `01_eda_ml-1m.ipynb` explores the data and surfaces the popularity bias that drives the first recommender; `02_splits_ml-1m.ipynb` quantifies the leakage difference between the two splits.

## Popularity Model

The first recommender in the lab, built as a baseline for every later stage to beat.

`base.py` defines the interface every recommender shares: fit on training ratings, predict a rating for a user and movie, and recommend a ranked list for a user.

`popularity.py` implements the recommender itself. It ranks movies by how often they were rated in train, so every user receives nearly the same list. Its rating prediction falls back to the global mean, because popularity is a ranking model, not a rating model.

`evaluation/` holds the shared scoring used by every stage. `evaluate_predictions` reports rating error, while `evaluate_ranking` scores each user's top-k against the movies they actually rated, macro-averaged across all users.

The model matches mass taste surprisingly well, but it personalizes nothing. Its precision and ranking quality are weak because the list is identical for everyone. Popularity is a useful benchmark and cold-start fallback, and the bar every later model must clear.

Notebook: `03_popularity_ml-1m.ipynb` walks through the baseline, its evaluation, and the ceiling it hits.
