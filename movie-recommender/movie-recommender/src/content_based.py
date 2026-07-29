"""
content_based.py
-----------------
Content-based filtering: recommends movies similar to a given movie based
on genres (and tags, if available) using TF-IDF + cosine similarity.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["clean_title"].str.lower()
        )
        self._vectorizer = None
        self._similarity_matrix = None

    def fit(self):
        """Build the TF-IDF matrix and cosine similarity matrix."""
        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self._vectorizer.fit_transform(
            self.movies["content_soup"].fillna("")
        )
        self._similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return self

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n movies most similar to `title`."""
        if self._similarity_matrix is None:
            raise RuntimeError("Call .fit() before .recommend().")

        idx = self._resolve_title(title)
        if idx is None:
            return pd.DataFrame(columns=["movieId", "title", "score"])

        scores = list(enumerate(self._similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        result = self.movies.iloc[[i for i, _ in scores]][
            ["movieId", "title", "genres"]
        ].copy()
        result["score"] = [round(s, 4) for _, s in scores]
        return result.reset_index(drop=True)

    def similar_movie_ids(self, movie_id: int, top_n: int = 10):
        """Return (movieId, score) pairs similar to a given movieId."""
        row = self.movies.index[self.movies["movieId"] == movie_id]
        if len(row) == 0:
            return []
        idx = row[0]
        scores = list(enumerate(self._similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]
        return [(self.movies.iloc[i]["movieId"], score) for i, score in scores]

    def _resolve_title(self, title: str):
        key = title.strip().lower()
        if key in self.title_to_index:
            return int(self.title_to_index[key])
        # fallback: partial match
        matches = self.movies[
            self.movies["clean_title"].str.lower().str.contains(key, na=False)
        ]
        return int(matches.index[0]) if len(matches) else None


if __name__ == "__main__":
    from data_loader import DataLoader

    movies, ratings, tags = DataLoader(data_dir="data").load()
    model = ContentBasedRecommender(movies).fit()
    print(model.recommend("Toy Story", top_n=5))
