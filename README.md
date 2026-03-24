# Zero-shot Linear

Zero-shot linear (`zslinear`) automatically selects solvers, regularization, and feature selection for linear classification and regression — no manual tuning required.

Given a dataset `(X, y)`, it detects the regime (sample count, dimensionality, class balance, target skewness) and picks the right combination of solver, penalty, and regularization strength grounded in published theory. It handles datasets ranging from **10 million rows × 2 000 features** down to **100 rows × 3 000 features** with the same API.

## Why?

Linear models work well across a huge range of dataset sizes, but their defaults are not optimal for every case:

- A dataset with 100 samples and 3 000 features needs strong L1/ElasticNet regularization and a pre-filtering step — otherwise the model overfits severely. Pure L1 cannot select more than *n* features when *p > n* (Zou & Hastie, 2005); ElasticNet is required.
- A dataset with 10 million rows needs a stochastic gradient descent solver — batch solvers run out of memory or take hours.
- A standard-sized dataset with balanced features is best served by `liblinear`, the fastest L1 coordinate descent solver for small-to-medium *n* (Fan & Lin, 2008).
- Severely imbalanced bioactivity labels need `class_weight="balanced"` and a scoring metric (ROC-AUC or balanced accuracy) that reflects minority-class performance.
- A skewed regression target (e.g. raw IC50 concentrations) should use mean absolute error for cross-validation, not R².

`zslinear` encodes these decisions as data-adaptive rules derived from linear model theory, producing well-calibrated models without a single tuning job.

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

No scaling is applied internally — standardize your data before calling `fit()`.

## Installation

```bash
pip install git+https://github.com/ersilia-os/zeroshot-linear.git
```

Requires Python ≥ 3.10. Core dependencies: `scikit-learn>=1.3`, `numpy>=1.21`, `scipy`, `loguru>=0.6`, `rich>=12.0`.

For ONNX export support:

```bash
pip install "zslinear[onnx]"
```

## Quick start

```python
from zslinear import LinearClassifier, LinearRegressor

# Binary classification
clf = LinearClassifier()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]

# Regression
reg = LinearRegressor()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

Both estimators are sklearn-compatible and work inside `Pipeline`, `cross_val_score`, etc.

**Note:** scale your features before fitting. `zslinear` does not apply `StandardScaler` internally — this is intentional so you can control the preprocessing pipeline.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

clf.fit(X_train_sc, y_train)
probs = clf.predict_proba(X_test_sc)[:, 1]
```

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

For the `large` regime the banner also reports the best alpha found on the subsample and the final training size.

## ONNX export

After fitting, export to ONNX for deployment in any ONNX-compatible runtime. The full preprocessing pipeline (VarianceThreshold, optional SelectFromModel, estimator) is embedded in the ONNX graph — callers pass raw unscaled features.

```python
clf.to_onnx("classifier.onnx")
reg.to_onnx("regressor.onnx")
```

Requires `pip install zslinear[onnx]`. Run inference with `onnxruntime`:

```python
import onnxruntime as ort
import numpy as np

# Classifier — outputs: output_label (int64, n), output_probability (float32, n×2)
sess  = ort.InferenceSession("classifier.onnx")
label, proba = sess.run(None, {"float_input": X_test.astype(np.float32)})

# Regressor — outputs: variable (float32, n×1)
sess  = ort.InferenceSession("regressor.onnx")
preds = sess.run(None, {"float_input": X_test.astype(np.float32)})[0].ravel()
```

## Forcing a specific regime

You can override regime detection to benchmark or reproduce results:

```python
clf = LinearClassifier(regime="high_dim")   # force high_dim path on any data
clf = LinearClassifier(regime="large")      # force SGD path
```

## Handling large datasets (10 M+ rows)

For `n > 50 000`, `zslinear` automatically uses `SGDClassifier` / `SGDRegressor` with an ElasticNet penalty. Hyperparameter tuning is performed on a stratified subsample (default 10 000 rows), then the final model is refitted on the full dataset. You can tune the subsample size:

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

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `fit(X, y)` | `self` | Train the model |
| `predict(X)` | `ndarray (n,)` | Class labels |
| `predict_proba(X)` | `ndarray (n, 2)` | Class probabilities |
| `score(X, y)` | `float` | Balanced accuracy |
| `get_feature_names_out()` | `ndarray` | Indices of selected features |
| `to_onnx(path)` | — | Export full pipeline to ONNX |

**Attributes after `.fit()`:**

| Attribute | Description |
|---|---|
| `regime_` | Detected or forced regime |
| `feature_mask_` | Boolean mask, shape `(n_features_in_,)` |
| `coef_` | Estimator coefficients |
| `classes_` | Unique class labels |
| `n_features_in_` | Number of input features |

---

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

Same methods as `LinearClassifier`, minus `predict_proba`. `score()` returns R². CV scoring is auto-selected: `r2` for symmetric targets, `neg_mean_absolute_error` for skewed targets (|skewness| > 1).

## References

- Fan, R.-E., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9, 1871–1874.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301–320.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *JSS*, 33(1), 1–22.
- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *COMPSTAT*, 177–186.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fuelling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
