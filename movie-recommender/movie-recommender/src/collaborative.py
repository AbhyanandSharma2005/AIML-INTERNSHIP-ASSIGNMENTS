"""
collaborative.py
-----------------
Collaborative filtering via matrix factorization (Truncated SVD) on the
user-item ratings matrix. Predicts a rating for any (user, movie) pair
and can rank unseen movies for a given user.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, n_factors: int = 30):
        self.ratings = ratings
        self.n_factors = n_factors

        self.user_ids = sorted(ratings["userId"].unique())
        self.movie_ids = sorted(ratings["movieId"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}
        self.index_to_movie = {i: m for m, i in self.movie_index.items()}

        self.user_means = None
        self.predicted_matrix = None

    def fit(self):
        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)

        rows = self.ratings["userId"].map(self.user_index)
        cols = self.ratings["movieId"].map(self.movie_index)
        data = self.ratings["rating"].values

        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))

        # Mean-center each user's ratings (helps SVD model relative preference)
        dense = matrix.toarray().astype(float)
        mask = dense != 0
        counts = mask.sum(axis=1)
        sums = dense.sum(axis=1)
        self.user_means = np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts != 0
        )

        centered = dense.copy()
        for i in range(n_users):
            centered[i, mask[i]] -= self.user_means[i]

        n_components = min(self.n_factors, min(centered.shape) - 1)
        n_components = max(n_components, 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_users = svd.fit_transform(centered)
        approx = latent_users @ svd.components_

        self.predicted_matrix = approx + self.user_means.reshape(-1, 1)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicted rating for a single (user, movie) pair."""
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return float(np.nanmean(self.user_means)) if self.user_means is not None else 3.0
        u = self.user_index[user_id]
        m = self.movie_index[movie_id]
        return float(np.clip(self.predicted_matrix[u, m], 0.5, 5.0))

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True):
        """Rank all movies for a user by predicted rating."""
        if user_id not in self.user_index:
            return []

        u = self.user_index[user_id]
        preds = self.predicted_matrix[u].copy()

        if exclude_seen:
            seen = set(self.ratings[self.ratings["userId"] == user_id]["movieId"])
            for m_id in seen:
                if m_id in self.movie_index:
                    preds[self.movie_index[m_id]] = -np.inf

        top_indices = np.argsort(preds)[::-1][:top_n]
        return [
            (self.index_to_movie[i], float(np.clip(preds[i], 0.5, 5.0)))
            for i in top_indices
            if preds[i] != -np.inf
        ]


if __name__ == "__main__":
    from data_loader import DataLoader

    movies, ratings, tags = DataLoader(data_dir="data").load()
    model = CollaborativeRecommender(ratings).fit()
    sample_user = ratings["userId"].iloc[0]
    print(model.recommend(sample_user, top_n=5))
