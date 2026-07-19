import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "upwork_jobs.csv")

_df = None
_tfidf = None
_tfidf_matrix = None


def _load():
    global _df, _tfidf, _tfidf_matrix
    if _df is not None:
        return

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["description"])

    def resolve_price(row):
        if row["is_hourly"] == True or row["is_hourly"] == "True":
            low, high = row.get("hourly_low"), row.get("hourly_high")
            if pd.notna(low) and pd.notna(high):
                return ((low + high) / 2) * 40
            return np.nan
        return row.get("budget")

    df["resolved_price"] = df.apply(resolve_price, axis=1)
    df = df.dropna(subset=["resolved_price"])
    df = df[(df["resolved_price"] > 20) & (df["resolved_price"] < 20000)]

    sample_df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)
    sample_df["text_for_match"] = sample_df["title"].astype(str) + " " + sample_df["description"].astype(str)

    tfidf = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(sample_df["text_for_match"])

    _df, _tfidf, _tfidf_matrix = sample_df, tfidf, tfidf_matrix


def estimate_base_price(description: str, k: int = 15, min_similarity: float = 0.1) -> dict:
    _load()

    query_vec = _tfidf.transform([description])
    sims = cosine_similarity(query_vec, _tfidf_matrix)[0]

    top_k_idx = np.argsort(sims)[-k:][::-1]
    top_matches = _df.iloc[top_k_idx].copy()
    top_matches["similarity"] = sims[top_k_idx]
    top_matches = top_matches[top_matches["similarity"] >= min_similarity]

    if len(top_matches) == 0:
        return {
            "estimated_base_price": round(float(np.median(_df["resolved_price"])), 2),
            "top_matches": [],
            "fallback_used": True,
        }

    weights = top_matches["similarity"].values
    prices = top_matches["resolved_price"].values
    estimated_price = float(np.average(prices, weights=weights))

    return {
        "estimated_base_price": round(estimated_price, 2),
        "top_matches": top_matches[["title", "resolved_price", "similarity"]].head(5).to_dict("records"),
        "fallback_used": False,
    }