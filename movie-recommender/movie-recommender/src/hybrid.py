"""
hybrid.py
---------
Combines content-based similarity and collaborative-filtering predicted
ratings into a single hybrid score:

    hybrid_score = alpha * normalized_cf_score + (1 - alpha) * content_score

- If a userId is given, CF supplies personalized signal.
- If no userId (cold-start user), the system falls back to pure content-based.
"""

import numpy as np
import pandas as pd

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, alpha: float = 0.5):
        """
        alpha: weight given to collaborative filtering (0 = pure content-based,
               1 = pure collaborative filtering).
        """
        self.movies = movies
        self.ratings = ratings
        self.alpha = alpha
        self.content_model = ContentBasedRecommender(movies)
        self.cf_model = CollaborativeRecommender(ratings)

    def fit(self):
        self.content_model.fit()
        self.cf_model.fit()
        return self

    def recommend_for_movie(self, title: str, user_id: int = None, top_n: int = 10):
        """
        'People who liked this movie also liked...' style recommendation,
        optionally personalized if a userId is supplied.
        """
        similar = self.content_model.recommend(title, top_n=top_n * 3)
        if similar.empty:
            return similar

        if user_id is None or user_id not in self.cf_model.user_index:
            return similar.head(top_n)

        cf_scores = similar["movieId"].apply(lambda m: self.cf_model.predict(user_id, m))
        cf_scores_norm = self._normalize(cf_scores)
        content_scores_norm = self._normalize(similar["score"])

        similar["hybrid_score"] = (
            self.alpha * cf_scores_norm + (1 - self.alpha) * content_scores_norm
        )
        return (
            similar.sort_values("hybrid_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def recommend_for_user(self, user_id: int, top_n: int = 10):
        """
        Personalized 'recommended for you' list. Falls back to popularity-based
        content recommendations if the user is unknown (cold start).
        """
        if user_id not in self.cf_model.user_index:
            return self._cold_start_recommendations(top_n)

        cf_recs = self.cf_model.recommend(user_id, top_n=top_n * 3)
        if not cf_recs:
            return self._cold_start_recommendations(top_n)

        movie_ids = [m for m, _ in cf_recs]
        cf_score_map = dict(cf_recs)

        # Blend in content similarity to the user's highest-rated movie for variety
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        content_score_map = {}
        if not user_ratings.empty:
            top_movie_id = user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"]
            for m_id, score in self.content_model.similar_movie_ids(top_movie_id, top_n=top_n * 3):
                content_score_map[m_id] = score

        rows = []
        for m_id in movie_ids:
            cf_s = cf_score_map.get(m_id, 0)
            content_s = content_score_map.get(m_id, 0)
            rows.append({"movieId": m_id, "cf_score": cf_s, "content_score": content_s})

        result = pd.DataFrame(rows)
        result["cf_norm"] = self._normalize(result["cf_score"])
        result["content_norm"] = self._normalize(result["content_score"])
        result["hybrid_score"] = (
            self.alpha * result["cf_norm"] + (1 - self.alpha) * result["content_norm"]
        )
        result = result.merge(self.movies[["movieId", "title", "genres"]], on="movieId")
        return (
            result.sort_values("hybrid_score", ascending=False)
            .head(top_n)[["movieId", "title", "genres", "hybrid_score"]]
            .reset_index(drop=True)
        )

    def _cold_start_recommendations(self, top_n: int):
        """Popularity-based fallback: highest average rating with a minimum
        number of votes (weighted rating, like IMDb's formula)."""
        stats = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        m = stats["count"].quantile(0.75)
        C = stats["mean"].mean()
        stats["weighted_score"] = (
            stats["count"] / (stats["count"] + m) * stats["mean"]
            + m / (stats["count"] + m) * C
        )
        stats = stats.reset_index()  # movieId becomes a column
        top = stats.sort_values("weighted_score", ascending=False).head(top_n)
        result = top.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
        return result[["movieId", "title", "genres", "weighted_score"]].reset_index(drop=True)

    @staticmethod
    def _normalize(series):
        series = pd.Series(series).astype(float)
        min_v, max_v = series.min(), series.max()
        if max_v - min_v == 0:
            return series * 0
        return (series - min_v) / (max_v - min_v)


if __name__ == "__main__":
    from data_loader import DataLoader

    movies, ratings, tags = DataLoader(data_dir="data").load()
    hybrid = HybridRecommender(movies, ratings, alpha=0.5).fit()

    print("== Similar to 'Toy Story' ==")
    print(hybrid.recommend_for_movie("Toy Story", top_n=5))

    sample_user = ratings["userId"].iloc[0]
    print(f"\n== Recommended for user {sample_user} ==")
    print(hybrid.recommend_for_user(sample_user, top_n=5))
