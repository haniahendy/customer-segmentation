# customer-segmentation

RFM (recency, frequency, monetary) customer segmentation with K-Means: standardize features, choose k by silhouette score, profile each cluster.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q   # 4 tests
python run.py
```

## Measured result (23 Nov 2026)
On synthetic customers drawn from 3 planted groups, silhouette picks k=3 (0.762) and the clusters match the planted groups exactly (adjusted Rand index 1.0). Planted groups are far apart, which makes this easy.

## Limits
Real customer data is rarely this well separated. K-Means assumes round clusters of similar size, and RFM features are skewed (consider log transform). Silhouette is one criterion, not proof of a meaningful segment.

## Provenance
A learning and portfolio project.
