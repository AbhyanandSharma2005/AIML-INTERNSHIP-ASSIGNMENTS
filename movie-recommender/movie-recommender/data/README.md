# Dataset setup

This project uses the **MovieLens ml-latest-small** dataset (100,000 ratings,
~9,000 movies) from GroupLens Research.

## Steps

1. Download it from: https://grouplens.org/datasets/movielens/
   (choose "ml-latest-small.zip")
2. Unzip it.
3. Copy these three files into this `data/` folder:
   - `movies.csv`
   - `ratings.csv`
   - `tags.csv` (optional, but improves content-based recommendations)

Your folder should look like:

```
data/
├── movies.csv
├── ratings.csv
└── tags.csv
```

The CSVs are intentionally excluded from git (see `.gitignore`) since the
dataset is redistributable but large — everyone who clones this repo should
download it themselves.
