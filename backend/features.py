import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_vectorizer = HashingVectorizer(n_features=16, alternate_sign=False, norm="l2")

KEYWORDS = {
    "urgent": ["urgent", "asap", "immediately", "rush"],
    "enterprise": ["enterprise", "corporation", "large company", "fortune"],
    "startup": ["startup", "seed", "early-stage", "small team"],
    "long_term": ["long-term", "ongoing", "retainer", "monthly"],
    "complex": ["dashboard", "machine learning", "ai", "integration", "scalable", "architecture"],
    "simple": ["simple", "basic", "small fix", "quick", "landing page"],
}

def extract_features(description: str) -> np.ndarray:
    text = description.lower()

    keyword_feats = []
    for _, words in KEYWORDS.items():
        keyword_feats.append(1.0 if any(w in text for w in words) else 0.0)

    length_feat = min(len(text.split()) / 100.0, 1.0)

    text_embedding = _vectorizer.transform([text]).toarray()[0]

    context = np.concatenate([
        keyword_feats,
        [length_feat],
        text_embedding,
        [1.0],
    ])
    return context

CONTEXT_DIM = 24