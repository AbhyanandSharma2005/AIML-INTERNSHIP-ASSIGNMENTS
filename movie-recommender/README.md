# 🎬 Hybrid Movie Recommendation System

An end-to-end movie recommender combining **content-based filtering**
(genres/tags similarity) and **collaborative filtering** (matrix
factorization on user ratings) into a single hybrid model, served through
a Streamlit web app.

## How it works

| Component | Technique | File |
|---|---|---|
| Content-based | TF-IDF over genres + tags, cosine similarity | `src/content_based.py` |
| Collaborative filtering | Mean-centered user-item matrix + Truncated SVD | `src/collaborative.py` |
| Hybrid | Weighted blend of normalized scores from both models | `src/hybrid.py` |
| Data pipeline | Loads & cleans MovieLens CSVs | `src/data_loader.py` |
| Web app | Streamlit UI: search a movie or a user, get ranked recommendations | `app.py` |

The hybrid score is:

```
hybrid_score = alpha * collaborative_score + (1 - alpha) * content_score
```

`alpha` is adjustable live in the app sidebar — set it to 0 for pure
content-based, 1 for pure collaborative filtering, or anywhere in between.

New/unknown users automatically fall back to a popularity-based
recommendation (weighted rating, same formula IMDb uses) — a simple fix
for the cold-start problem.

## Setup

```bash
pip install -r requirements.txt
```

Then download the dataset — see `data/README.md` for the two-minute setup.

## Run the app

```bash
streamlit run app.py
```

Two modes are available in the app:
- **Similar to a movie** — pick any movie, get similar ones (optionally
  personalized if you supply a user ID).
- **Recommended for a user** — pick a user ID from the dataset, get a
  personalized ranked list plus their rating history.

## Run components individually (for testing/exploring)

```bash
python src/data_loader.py       # inspect the cleaned data
python src/content_based.py     # test content-based similarity
python src/collaborative.py     # test collaborative filtering
python src/hybrid.py            # test the full hybrid pipeline
```

## Project structure

```
movie-recommender/
├── app.py                  # Streamlit UI
├── requirements.txt
├── data/
│   └── README.md           # dataset download instructions
└── src/
    ├── data_loader.py       # load + clean MovieLens CSVs
    ├── content_based.py     # TF-IDF + cosine similarity
    ├── collaborative.py     # SVD matrix factorization
    └── hybrid.py             # blends both models
```

## Possible extensions

- Swap TruncatedSVD for the `surprise` library's SVD++ for better accuracy
- Add movie posters via the TMDB API
- Add implicit feedback (watch time, clicks) alongside explicit ratings
- Deploy the Streamlit app to Streamlit Community Cloud or Render
