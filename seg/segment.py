import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency_days", "frequency", "monetary"]


def make_rfm(n_per: int = 150, seed: int = 2) -> pd.DataFrame:
    """Synthetic customers drawn from 3 known groups (loyal, new, lapsed)."""
    rng = np.random.default_rng(seed)
    groups = {"loyal": (10, 25, 900), "new": (20, 3, 120), "lapsed": (200, 5, 250)}
    parts = []
    for name, (r, f, m) in groups.items():
        parts.append(pd.DataFrame({
            "recency_days": rng.normal(r, r * 0.2, n_per).clip(1),
            "frequency": rng.poisson(f, n_per) + 1,
            "monetary": rng.normal(m, m * 0.2, n_per).clip(5),
            "true_group": name}))
    return pd.concat(parts, ignore_index=True)


def choose_k(X: np.ndarray, k_range=range(2, 8), seed: int = 0):
    scores = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=seed).fit_predict(X)) for k in k_range}
    best = max(scores, key=scores.get)
    return best, {k: round(v, 3) for k, v in scores.items()}


def segment(df: pd.DataFrame, k: int | None = None, seed: int = 0):
    X = StandardScaler().fit_transform(df[FEATURES])
    scores = None
    if k is None:
        k, scores = choose_k(X, seed=seed)
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    return k, labels, scores


def profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    return df.assign(cluster=labels).groupby("cluster")[FEATURES].mean().round(1).assign(size=pd.Series(labels).value_counts().sort_index())
