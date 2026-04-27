# Hybrid CNN+RF — Hyperparameter Tuning

CNN feature extractor was held frozen; only the Random Forest was tuned.
Same dataset and 80/20 split as the original training run (`generate_pqd_signals(n_per_class=1000, seed=42)`, `train_test_split(..., random_state=42, stratify=y)`).

## Search space
```
{
  "n_estimators": [
    100,
    200,
    500
  ],
  "max_depth": [
    null,
    20
  ],
  "max_features": [
    "sqrt",
    "log2"
  ],
  "min_samples_leaf": [
    1,
    2
  ]
}
```
- Folds: 5 (StratifiedKFold, shuffle=True, random_state=42)
- Scoring: accuracy
- Total fits: 120
- Wall time: 285.2 s (4.8 min)

## Winner

```
{
  "max_depth": null,
  "max_features": "sqrt",
  "min_samples_leaf": 1,
  "n_estimators": 500
}
```

- Best 5-fold CV accuracy: **98.941%**
- Held-out test accuracy:  **98.882%**
- Held-out test F1 macro:  98.882%

## Comparison vs the current saved model

| Model                              | Test accuracy | Notes |
|------------------------------------|--------------:|:------|
| Current saved (n=200, log2)        | 98.85% | From the training run that produced `hybrid_rf.pkl` |
| Tuning winner                      | 98.88% | Delta +0.03 pp |

## Full grid (sorted by CV rank)

| Rank | n_estimators | max_depth | max_features | min_samples_leaf | CV accuracy            | mean fit (s) |
|-----:|-------------:|----------:|:-------------|-----------------:|:-----------------------|-------------:|
|    1 |          500 |      None | sqrt         |                1 | 98.941% +/- 0.184% |         4.10 |
|    1 |          200 |      None | sqrt         |                2 | 98.941% +/- 0.157% |         1.71 |
|    1 |          500 |      None | log2         |                2 | 98.941% +/- 0.146% |         3.43 |
|    1 |          100 |        20 | sqrt         |                1 | 98.941% +/- 0.215% |         0.85 |
|    1 |          500 |        20 | log2         |                1 | 98.941% +/- 0.151% |         4.03 |
|    1 |          500 |        20 | log2         |                2 | 98.941% +/- 0.160% |         3.51 |
|    7 |          100 |      None | sqrt         |                1 | 98.934% +/- 0.190% |         0.81 |
|    7 |          200 |      None | sqrt         |                1 | 98.934% +/- 0.193% |         1.57 |
|    7 |          100 |      None | sqrt         |                2 | 98.934% +/- 0.168% |         0.85 |
|    7 |          500 |      None | log2         |                1 | 98.934% +/- 0.154% |         3.41 |
|    7 |          200 |        20 | sqrt         |                1 | 98.934% +/- 0.193% |         1.65 |
|   12 |          200 |        20 | log2         |                2 | 98.934% +/- 0.147% |         1.44 |
|   13 |          200 |      None | log2         |                1 | 98.926% +/- 0.179% |         1.41 |
|   13 |          200 |      None | log2         |                2 | 98.926% +/- 0.153% |         1.38 |
|   13 |          500 |        20 | sqrt         |                1 | 98.926% +/- 0.194% |         4.13 |
|   13 |          200 |        20 | sqrt         |                2 | 98.926% +/- 0.173% |         1.67 |
|   13 |          200 |        20 | log2         |                1 | 98.926% +/- 0.157% |         1.56 |
|   18 |          500 |      None | sqrt         |                2 | 98.919% +/- 0.170% |         4.18 |
|   18 |          100 |      None | log2         |                2 | 98.919% +/- 0.148% |         0.70 |
|   20 |          500 |        20 | sqrt         |                2 | 98.912% +/- 0.178% |         4.27 |
|   21 |          100 |        20 | log2         |                2 | 98.912% +/- 0.164% |         0.75 |
|   22 |          100 |        20 | sqrt         |                2 | 98.904% +/- 0.170% |         0.88 |
|   23 |          100 |      None | log2         |                1 | 98.897% +/- 0.171% |         0.72 |
|   23 |          100 |        20 | log2         |                1 | 98.897% +/- 0.190% |         0.82 |

## Caveats

- CNN extractor is frozen; only RF hyperparameters were searched. A different CNN architecture / training schedule could shift the optimum.
- Dataset is synthetic (IEEE 1159 parametric equations + AWGN @ 40 dB). Real-grid data may favour different hyperparameters.
- `hybrid_rf.pkl` on disk was NOT overwritten by this script. If the tuning winner is meaningfully better and you want to deploy it, save the refitted estimator manually:
  ```python
  joblib.dump(gs.best_estimator_, 'backend/hybrid_rf.pkl')
  ```
