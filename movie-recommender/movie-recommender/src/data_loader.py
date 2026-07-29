"""
data_loader.py
--------------
Loads and preprocesses the MovieLens dataset (movies.csv + ratings.csv,
optionally tags.csv) into clean pandas DataFrames ready for both the
content-based and collaborative-filtering models.

Expected MovieLens files (ml-latest-small format from grouplens.org):
    data/movies.csv   -> movieId, title, genres
    data/ratings.csv  -> userId, movieId, rating, timestamp
    data/tags.csv     -> userId, movieId, tag, timestamp   (optional)
"""

import os
import re
import pandas as pd


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movies = None
        self.ratings = None
        self.tags = None

    def load(self):
        """Load raw CSVs from data_dir into DataFrames."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        tags_path = os.path.join(self.data_dir, "tags.csv")

        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Could not find movies.csv / ratings.csv in '{self.data_dir}'. "
                "Download the MovieLens ml-latest-small dataset from "
                "https://grouplens.org/datasets/movielens/ and place the CSVs there."
            )

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else None

        self._clean()
        return self.movies, self.ratings, self.tags

    def _clean(self):
        # Extract release year from the title, e.g. "Toy Story (1995)"
        self.movies["year"] = self.movies["title"].apply(self._extract_year)
        self.movies["clean_title"] = self.movies["title"].apply(self._strip_year)

        # Genres arrive as "Action|Adventure|Sci-Fi" -> normalize to space separated
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["genres_str"] = self.movies["genres"].apply(
            lambda g: " ".join(g.split("|")) if g != "(no genres listed)" else ""
        )

        # Merge tags into a single "soup" of text per movie if tags exist
        if self.tags is not None:
            tag_text = (
                self.tags.dropna(subset=["tag"])
                .groupby("movieId")["tag"]
                .apply(lambda tags: " ".join(str(t).lower() for t in tags))
                .reset_index()
                .rename(columns={"tag": "tags_str"})
            )
            self.movies = self.movies.merge(tag_text, on="movieId", how="left")
            self.movies["tags_str"] = self.movies["tags_str"].fillna("")
        else:
            self.movies["tags_str"] = ""

        # Final text field the content-based model will vectorize
        self.movies["content_soup"] = (
            self.movies["genres_str"] + " " + self.movies["tags_str"]
        ).str.strip()

        # Drop duplicate/invalid ratings
        self.ratings = self.ratings.dropna(subset=["userId", "movieId", "rating"])

    @staticmethod
    def _extract_year(title: str):
        match = re.search(r"\((\d{4})\)\s*$", str(title))
        return int(match.group(1)) if match else None

    @staticmethod
    def _strip_year(title: str):
        return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


if __name__ == "__main__":
    loader = DataLoader(data_dir="data")
    movies, ratings, tags = loader.load()
    print(movies.head())
    print(f"\n{len(movies)} movies, {len(ratings)} ratings")
