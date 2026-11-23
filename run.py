from sklearn.metrics import adjusted_rand_score

from seg.segment import make_rfm, profile, segment

df = make_rfm()
k, labels, scores = segment(df)
print("silhouette by k:", scores)
print("chosen k:", k, "| adjusted Rand vs planted groups:", round(adjusted_rand_score(df["true_group"], labels), 3))
print(profile(df, labels))
