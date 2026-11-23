import numpy as np
from sklearn.metrics import adjusted_rand_score

from seg.segment import make_rfm, profile, segment


def test_recovers_planted_three_groups():
    df = make_rfm()
    k, labels, _ = segment(df)
    assert k == 3
    assert adjusted_rand_score(df["true_group"], labels) > 0.9


def test_fixed_k_and_determinism():
    df = make_rfm()
    _, a, _ = segment(df, k=4)
    _, b, _ = segment(df, k=4)
    assert (a == b).all() and len(set(a)) == 4


def test_profile_sizes_add_up():
    df = make_rfm()
    _, labels, _ = segment(df, k=3)
    assert profile(df, labels)["size"].sum() == len(df)


def test_features_positive():
    df = make_rfm()
    assert (df[["recency_days", "frequency", "monetary"]] > 0).all().all()
