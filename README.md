# RecommendationSystemLab

Recommendation systems built from first principles.

This is a personal experiment exploring the evolution of recommendation systems. Each algorithm is implemented from scratch, one milestone at a time, from popularity based baselines to hybrid systems. The goal is to understand not just how these algorithms work, but why each one exists and what limitation it addresses.

Every stage answers the same question: what limitation of the previous algorithm are we solving?

1. **Popularity Recommendation.** The baseline. It requires no user history and no learning, and establishes the structure every subsequent recommender builds on.

2. **Content-Based Recommendation.** Popularity ignores what a user actually likes. Content based filtering exploits item features to recommend items similar to ones a user has already engaged with, and handles the new item cold start problem.

3. **User-Based Collaborative Filtering.** Content based requires rich features and only surfaces similar items. User based collaborative filtering finds structure directly in the ratings, discovering taste by looking at users with similar preferences.

4. **Item-Based Collaborative Filtering.** User based filtering does not scale, because similarity must be computed per query. Item based filtering precomputes item to item similarities offline, making online serving fast and practical.

5. **Matrix Factorization.** Neighborhood methods struggle with sparse rating matrices and cannot generalize to unseen combinations. Matrix factorization learns compact latent factors that capture hidden taste dimensions.

6. **Implicit Feedback Recommendation.** Explicit ratings are rare, costly to obtain, and biased. Implicit signals such as views, clicks, and plays are abundant, and reframe the problem as preference ranking rather than rating prediction.

7. **Hybrid Recommendation.** Every method has blind spots. Hybrids combine complementary approaches to mitigate cold start, popularity bias, and sparsity, and are closer to what production systems actually ship.

