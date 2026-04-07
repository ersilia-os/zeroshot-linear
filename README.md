# Zero-shot Linear

`zslinear` automatically selects solvers, regularization, and feature selection for linear classification and regression — no manual tuning required.

Given a dataset `(X, y)`, it detects the regime (sample count, dimensionality, class balance, target skewness) and picks the right combination of solver, penalty, and regularization strength grounded in published theory. It handles datasets ranging from **10 million rows × 2 000 features** down to **100 rows × 3 000 features** with the same API.

## Installation

```bash
pip install git+https://github.com/ersilia-os/zeroshot-linear.git
```

Requires Python ≥ 3.10.

## Quick start

```python
from zslinear import LinearClassifier, LinearRegressor

# Binary classification
clf = LinearClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)           # binary labels
proba = clf.predict_proba(X_test)     # shape (n, 2) — [P(0), P(1)]

# Regression
reg = LinearRegressor()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)           # shape (n,)
```

Both estimators are sklearn-compatible and work inside `Pipeline`, `cross_val_score`, etc.

**Note:** scale your features before fitting. `zslinear` does not apply `StandardScaler` internally — this is intentional so you can control the preprocessing pipeline.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

clf.fit(X_train_sc, y_train)
proba = clf.predict_proba(X_test_sc)
```

## Saving and loading models

### Save

```python
clf.save("my_model/")             # default: ONNX format
clf.save("my_model/", onnx=True)  # explicit ONNX
clf.save("my_model/", onnx=False) # joblib format
```

`save()` always writes two files:
- `linear.onnx` (or `linear.joblib`) — the serialised model and full preprocessing pipeline
- `linear.json` — fit metadata (regime, n_features_in, feature_mask, classes)

### Load and run inference

Use `LinearArtifact` to load a saved model without needing to re-fit:

```python
from zslinear import LinearArtifact

artifact = LinearArtifact.load("my_model/")
out = artifact.run(X_test)
# Classification: shape (n, 2) — [P(class=0), P(class=1)]
# Regression:     shape (n,)   — predicted values
```

`load()` automatically detects whether the saved format is ONNX or joblib. The fit metadata is available at `artifact.metadata`.

## Inspecting the chosen regime and features

```python
clf = LinearClassifier(verbose=1)
clf.fit(X_train, y_train)

print(clf.regime_)           # "standard", "high_dim", or "large"
print(clf.feature_mask_)     # bool array, shape (n_features,)
print(clf.coef_)             # coefficients of the final estimator
print(clf.feature_mask_.sum(), "features selected out of", X_train.shape[1])
```

Example output for a 120-sample × 3 000-feature bioactivity dataset with 15 % positives:

```
─────────────────── LinearClassifier ───────────────────
regime        : high_dim
n             : 120
p             : 3000  →  pre-filter to 240  →  CV selects 31
l1_ratio      : 0.5
cv_folds      : 10
scoring       : balanced_accuracy
────────────────────────── Done ────────────────────────

regime_       : 'high_dim'
feature_mask_ : 31 features selected out of 3000
coef_         : shape (1, 31)
```

## How it works

### Regime detection

`zslinear` classifies every dataset into one of three regimes at fit time:

| Regime | Condition | Solver | Penalty |
|---|---|---|---|
| `standard` | *n* ≤ 50 000, *p* ≤ *n* | `liblinear` (classification) / `RidgeCV` (regression) | L1 / L2 |
| `high_dim` | *n* ≤ 50 000, *p* > *n* | `saga` / `ElasticNetCV` | ElasticNet |
| `large` | *n* > 50 000 | `SGDClassifier` / `SGDRegressor` | ElasticNet |

### Adaptive hyperparameter selection

Rather than a fixed grid, all hyperparameter grids are centered on a data-driven estimate:

| Hyperparameter | Rule |
|---|---|
| **C grid** (classification) | Centered on *C\* ≈ √n / p* — from lasso theory for random design matrices |
| **α grid** (regression/SGD) | Derived as *1 / (C\* × n)* — equivalent scale for sklearn regularization |
| **l1\_ratio** | 0.5 when *p/n > 10* (strong grouping for correlated features), 0.7 when *p/n > 2*, 0.9 otherwise |
| **CV folds** | 10 folds for *n < 200*, 5 for *n < 1 000*, 3 for *n ≥ 1 000* — always clamped to minority class count |
| **CV scoring** | `roc_auc` when class imbalance ratio < 0.1; `balanced_accuracy` otherwise (classification) |
| **CV scoring** | `neg_mean_absolute_error` when \|skewness\| > 1; `r2` otherwise (regression) |

### Feature selection

Feature selection is embedded in the regularization:

- **All regimes**: `VarianceThreshold` removes constant features before any fitting.
- **`standard`**: L1 regularization in the final estimator produces sparse coefficients directly.
- **`high_dim`**: A fast `SelectFromModel(Lasso / LogisticRegression[L1])` pre-filter reduces *p* to at most `max(10, min(p, 2×n))` features before the main ElasticNet CV — grounded in the Zou & Hastie (2005) result that pure L1 cannot select more than *n* features.
- **`large`**: ElasticNet SGD penalty naturally zeroes out irrelevant features.

### Large datasets (10 M+ rows)

For `n > 50 000`, `zslinear` automatically uses `SGDClassifier` / `SGDRegressor` with an ElasticNet penalty. Hyperparameter tuning is performed on a stratified subsample (default 10 000 rows), then the final model is refitted on the full dataset:

```python
clf = LinearClassifier(tuning_subsample=5_000)
clf.fit(X_large, y_large)   # tunes on 5 000 rows, fits on all
```

## API reference

### `LinearClassifier`

```python
LinearClassifier(
    regime=None,           # None = auto-detect; or "standard" | "high_dim" | "large"
    C_values=None,         # override adaptive C grid (standard/high_dim)
    alpha_values=None,     # override adaptive α grid (large)
    l1_ratio=None,         # ElasticNet mixing; None = auto from p/n
    cv=None,               # CV folds; None = auto from n and class balance
    class_weight="balanced",
    max_iter=10_000,
    n_jobs=-1,
    random_state=42,
    variance_threshold=0.0,
    tuning_subsample=10_000,
    verbose=0,
)
```

| Method | Description |
|---|---|
| `.fit(X, y)` | Train the model |
| `.predict(X)` | Binary labels |
| `.predict_proba(X)` | Class probabilities, shape `(n, 2)` |
| `.score(X, y)` | Balanced accuracy |
| `.get_feature_names_out()` | Indices of selected features |
| `.save(directory, onnx=True)` | Save model and metadata to a directory |

Attributes after `.fit()`: `regime_`, `feature_mask_`, `coef_`, `classes_`, `n_features_in_`.

### `LinearRegressor`

```python
LinearRegressor(
    regime=None,           # None = auto-detect
    alpha_values=None,     # override adaptive α grid
    l1_ratio=None,         # ElasticNet mixing; None = auto from p/n
    cv=None,               # CV folds; None = auto from n
    max_iter=10_000,
    n_jobs=-1,
    random_state=42,
    variance_threshold=0.0,
    tuning_subsample=10_000,
    verbose=0,
)
```

| Method | Description |
|---|---|
| `.fit(X, y)` | Train the model |
| `.predict(X)` | Predicted values, shape `(n,)` |
| `.score(X, y)` | R² score |
| `.get_feature_names_out()` | Indices of selected features |
| `.save(directory, onnx=True)` | Save model and metadata to a directory |

Loss function is chosen automatically: `r2` for symmetric targets, `neg_mean_absolute_error` for skewed targets (|skewness| > 1).

### `LinearArtifact`

```python
artifact = LinearArtifact.load(directory)  # load a saved model
out = artifact.run(X)                      # run inference
```

| Attribute | Description |
|---|---|
| `artifact.task` | `"classification"` or `"regression"` |
| `artifact.metadata` | Full contents of `linear.json` |

`run()` returns shape `(n, 2)` for classification and `(n,)` for regression.

## References

- Fan, R.-E., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9, 1871–1874.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301–320.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *JSS*, 33(1), 1–22.
- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *COMPSTAT*, 177–186.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fuelling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
