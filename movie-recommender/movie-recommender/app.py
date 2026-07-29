"""
app.py
------
Streamlit UI for the hybrid movie recommendation system.

Run with:
    streamlit run app.py
"""

import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DataLoader
from hybrid import HybridRecommender


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Training recommendation models...")
def load_and_train(data_dir: str, alpha: float):
    movies, ratings, tags = DataLoader(data_dir=data_dir).load()
    model = HybridRecommender(movies, ratings, alpha=alpha).fit()
    return movies, ratings, model


def main():
    st.title("🎬 Hybrid Movie Recommendation System")
    st.caption("Content-based + Collaborative Filtering, blended into one hybrid score.")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("Data folder", value="data")
        alpha = st.slider(
            "Collaborative filtering weight (alpha)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="0 = pure content-based, 1 = pure collaborative filtering",
        )
        mode = st.radio(
            "Recommendation mode",
            ["Similar to a movie", "Recommended for a user"],
        )

    try:
        movies, ratings, model = load_and_train(data_dir, alpha)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Download the MovieLens ml-latest-small dataset from "
            "https://grouplens.org/datasets/movielens/ and place movies.csv "
            "and ratings.csv inside the folder specified above."
        )
        return

    st.success(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")

    if mode == "Similar to a movie":
        show_similar_movie_ui(movies, model)
    else:
        show_user_recs_ui(ratings, model)


def show_similar_movie_ui(movies: pd.DataFrame, model: HybridRecommender):
    st.subheader("Find movies similar to one you like")

    titles = sorted(movies["clean_title"].unique())
    selected_title = st.selectbox("Pick a movie", titles)

    use_personalization = st.checkbox("Personalize using a user ID (optional)")
    user_id = None
    if use_personalization:
        user_id = st.number_input("User ID", min_value=1, step=1, value=1)

    top_n = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get Recommendations", type="primary"):
        results = model.recommend_for_movie(selected_title, user_id=user_id, top_n=top_n)
        render_results(results)


def show_user_recs_ui(ratings: pd.DataFrame, model: HybridRecommender):
    st.subheader("Get personalized recommendations for a user")

    known_users = sorted(ratings["userId"].unique())
    user_id = st.selectbox("Pick a user ID", known_users)
    top_n = st.slider("Number of recommendations", 5, 20, 10)

    user_history = ratings[ratings["userId"] == user_id].merge(
        model.movies[["movieId", "title"]], on="movieId"
    )
    with st.expander(f"User {user_id}'s rating history ({len(user_history)} movies)"):
        st.dataframe(
            user_history[["title", "rating"]].sort_values("rating", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Get Recommendations", type="primary"):
        results = model.recommend_for_user(user_id, top_n=top_n)
        render_results(results)


def render_results(results: pd.DataFrame):
    if results.empty:
        st.warning("No recommendations found — try a different movie or user.")
        return

    st.write("### Recommendations")
    for _, row in results.iterrows():
        score_col = "hybrid_score" if "hybrid_score" in row else (
            "score" if "score" in row else "weighted_score"
        )
        cols = st.columns([4, 2, 2])
        cols[0].markdown(f"**{row['title']}**")
        cols[1].caption(row.get("genres", ""))
        cols[2].metric("Score", f"{row[score_col]:.3f}")


if __name__ == "__main__":
    main()
