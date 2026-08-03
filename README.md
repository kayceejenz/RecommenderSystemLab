# RecommendationSystemLab

## Table of Contents

- [Overview](#overview)
- [Scope of the Work](#scope-of-the-work)
- [Roadmap](#roadmap)
- [Data Ingestion](#data-ingestion)
- [Popularity Model](#popularity-model)

## Overview

This is a personal experiment exploring the evolution of recommendation systems. The goal is to understand not just how these algorithms work, but why each one exists and what limitation it addresses.

Every stage answers the same question: **what limitation of the previous approach are we solving?**

The lab covers several recommendation approaches, with **collaborative filtering treated as a family rather than a single algorithm**. User-based and item-based neighborhood methods, as well as matrix factorization, are different ways of exploiting user-item interactions. Implicit feedback is treated separately as a **feedback paradigm**, because it can be used with multiple recommendation algorithms rather than being a standalone model family.

All stages share one data pipeline, one evaluation framework, and one set of metrics, so results stay comparable across the lab.

## Scope of the Work

- MovieLens 1M, with 1,000,209 ratings from 6,040 users over 3,883 movies.
- Seven recommendation approaches/stages, from popularity baselines to hybrid systems.
- Collaborative filtering is explored through three approaches: user-based, item-based, and matrix factorization.
- Implicit feedback is explored as a different interaction setting, rather than as a separate collaborative-filtering algorithm.
- A shared evaluation framework: RMSE and MAE for rating prediction, precision, recall, MAP, NDCG, and hit rate for ranking.
- Time-based train/test splits that eliminate temporal leakage, which a random split would otherwise introduce.

## Roadmap

### 1. **Popularity Recommendation**

The baseline. It requires no user history and no learning, and establishes the structure every subsequent recommender builds on.

Popularity recommendations ignore individual preferences, so every user receives essentially the same list. This gives us a useful non-personalized baseline and a cold-start fallback.

### 2. **Content-Based Recommendation**

Popularity ignores what a user actually likes. Content-based filtering exploits **item features** to recommend items similar to ones a user has already engaged with.

This introduces personalization without requiring other users' behavior and can also handle the **new-item cold-start problem** when item features are available.

### 3. **Collaborative Filtering: User-Based**

Content-based filtering depends on useful item features and primarily recommends items similar in content.

**User-based collaborative filtering** takes a different approach: it learns structure directly from the **user-item interaction matrix**. Instead of asking which items look similar, it finds users with similar observed preferences and uses their behavior to make recommendations.

The key limitation it addresses is the reliance on item metadata, but neighborhood search can become expensive as the number of users grows.

### 4. **Collaborative Filtering: Item-Based**

User-based collaborative filtering requires finding similar users for each recommendation query. As the number of users grows, this can make online serving expensive.

**Item-based collaborative filtering** shifts the neighborhood from users to items. It precomputes item-to-item similarities from historical interactions, allowing recommendations to be generated from the items a user has already interacted with.

This makes the neighborhood computation more stable and practical for online serving, although it still suffers from sparse interactions and limited generalization.

### 5. **Collaborative Filtering: Matrix Factorization**

Both user-based and item-based collaborative filtering are **neighborhood methods**. They rely on explicit similarities and can struggle when the user-item matrix is sparse.

**Matrix factorization** takes a model-based approach. Instead of storing explicit neighborhoods, it learns compact latent representations for users and items. These latent factors capture hidden dimensions of taste and allow the model to generalize across observed interactions.

This addresses the sparsity and generalization limitations of neighborhood-based collaborative filtering.

### 6. **Implicit Feedback Recommendation**

The previous collaborative-filtering stages primarily use **explicit ratings** as preference signals. In real recommendation systems, explicit ratings are often rare, costly to obtain, and subject to selection bias.

**Implicit feedback** changes the interaction setting rather than defining one specific algorithm. Signals such as views, clicks, plays, purchases, and dwell time are abundant, but they do not directly express a numerical rating.

The problem therefore shifts from **rating prediction** toward **preference ranking**: rather than asking "what rating would this user give this item?", the system asks "which items is this user more likely to prefer or interact with?"

Implicit feedback can be used with multiple recommendation approaches, including collaborative filtering and matrix-factorization-style models.

### 7. **Hybrid Recommendation**

Every individual approach has blind spots.

Content-based methods depend on item features. Collaborative filtering suffers from sparsity and cold-start problems. Popularity models personalize nothing. Implicit-feedback systems must infer preference from behavior rather than explicit judgments.

**Hybrid recommendation** combines complementary signals or models to mitigate these weaknesses, including cold start, popularity bias, sparse interactions, and limited item metadata.

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
