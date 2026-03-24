"""
LinearClassifier and LinearRegressor: linear models with auto feature selection.

Both handle three dataset regimes automatically:
  - standard  (n <= 50K, p <= n): fast closed-form / coordinate descent solver
  - high_dim  (n <= 50K, p >  n): ElasticNet + SelectFromModel pre-filter
  - large     (n >  50K)        : SGD estimator, alpha tuned on subsample

Note: feature scaling is NOT applied internally — callers should standardize their
data before passing it to fit() / predict().

Literature basis:
  Fan & Lin (2008) - LIBLINEAR coordinate descent for L1
  Zou & Hastie (2005) - ElasticNet removes L1 feature-count ceiling at p>n
  Friedman et al. (2010) - ElasticNet grouping effect for correlated features
  Bottou (2010) - SGD for large-scale linear models
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted

from zslinear.utils.logging import logger


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _detect_regime(n: int, p: int) -> str:
    if n > 50_000:
        return "large"
    if p > n:
        return "high_dim"
    return "standard"


# ---------------------------------------------------------------------------
# Hyperparameter heuristics
# ---------------------------------------------------------------------------

def _default_C_grid(n: int, p: int, n_grid: int = 20) -> np.ndarray:
    """Adaptive C grid centered on lasso-theory optimal C* ~ sqrt(n)/p."""
    C_center = max(1e-4, (n ** 0.5) / p)
    return np.logspace(np.log10(C_center) - 2, np.log10(C_center) + 2, n_grid)


def _default_alpha_grid(n: int, p: int, n_grid: int = 20) -> np.ndarray:
    """SGD alpha = 1/(C*n); derived from C grid."""
    C_grid = _default_C_grid(n, p, n_grid)
    return np.clip(1.0 / (C_grid * n), 1e-7, 1.0)


def _default_cv(n: int, y: np.ndarray) -> int:
    """Adaptive fold count: more folds when n is small."""
    min_class = int(np.bincount(y).min())
    if n < 200:
        return min(min_class, 10)
    if n < 1000:
        return min(min_class, 5)
    return min(min_class, 3)


def _imbalance_ratio(y: np.ndarray) -> float:
    counts = np.bincount(y)
    return float(counts.min() / counts.max())


def _default_l1_ratio(n: int, p: int) -> float:
    """ElasticNet mixing: more grouping (lower l1_ratio) when p/n is large."""
    ratio = p / max(n, 1)
    if ratio > 10:
        return 0.5
    if ratio > 2:
        return 0.7
    return 0.9


def _sfm_max_features(n: int, p: int) -> int:
    """SelectFromModel cap: at most 2n features (Zou & Hastie 2005 bound), at least 10."""
    return max(10, min(p, 2 * n))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class LinearClassifier:
    """
    Binary logistic regression with embedded L1/ElasticNet feature selection.

    Automatically selects solver and regularization regime based on (n, p):
      - standard  (n<=50K, p<=n): liblinear + L1,        LogisticRegressionCV
      - high_dim  (n<=50K, p>n):  saga + ElasticNet,     SelectFromModel + LogisticRegressionCV
      - large     (n>50K):        SGDClassifier + ElasticNet, subsample alpha tuning

    Feature scaling is NOT applied internally. Standardize X before calling fit().

    Parameters
    ----------
    regime : str or None
        Force a specific regime ("standard", "high_dim", "large").
        None = auto-detect from data shape.
    C_values : array-like or None
        Override the adaptive C grid for standard/high_dim regimes.
    alpha_values : array-like or None
        Override the adaptive alpha grid for the large regime.
    l1_ratio : float or None
        ElasticNet mixing (0=L2, 1=L1). None = auto from p/n heuristic.
    cv : int or None
        Number of CV folds. None = auto from n and class balance.
    class_weight : str or dict
        Passed to all estimators. "balanced" is strongly recommended for
        imbalanced bioactivity data.
    max_iter : int
        Max iterations for liblinear/saga solvers.
    n_jobs : int
        Parallelism for CV and some solvers.
    random_state : int or None
        Reproducibility seed.
    variance_threshold : float
        VarianceThreshold cutoff. 0.0 removes only constant features.
    tuning_subsample : int
        Max rows used for alpha-grid tuning in the large regime.
    verbose : int
        0 = silent, 1 = rule banners via Rich/loguru.
    """

    def __init__(
        self,
        *,
        regime: str | None = None,
        C_values: list | np.ndarray | None = None,
        alpha_values: list | np.ndarray | None = None,
        l1_ratio: float | None = None,
        cv: int | None = None,
        class_weight: str | dict = "balanced",
        max_iter: int = 10_000,
        n_jobs: int = -1,
        random_state: int | None = 42,
        variance_threshold: float = 0.0,
        tuning_subsample: int = 10_000,
        verbose: int = 0,
    ):
        self.regime = regime
        self.C_values = C_values
        self.alpha_values = alpha_values
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.tuning_subsample = tuning_subsample
        self.verbose = verbose

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "LinearClassifier":
        logger.set_verbosity(bool(self.verbose))
        logger.rule("LinearClassifier")

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y)

        # --- Label encoding ---
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)

        if len(np.unique(y_enc)) < 2:
            raise ValueError(
                "Training data contains only one class. "
                "Binary classification requires both classes."
            )

        n, p = X.shape
        self.n_features_in_ = p

        # --- Regime ---
        self.regime_ = self.regime if self.regime is not None else _detect_regime(n, p)

        # --- Heuristics ---
        effective_l1_ratio = self.l1_ratio if self.l1_ratio is not None else _default_l1_ratio(n, p)
        effective_cv = self.cv if self.cv is not None else _default_cv(n, y_enc)
        imbalance = _imbalance_ratio(y_enc)
        scoring = "roc_auc" if imbalance < 0.1 else "balanced_accuracy"

        skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)

        # --- Preprocessing: VarianceThreshold only (scaling is caller's responsibility) ---
        self._vt = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_vt = self._vt.fit_transform(X)
        except ValueError:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        if X_vt.shape[1] == 0:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        # --- Dispatch ---
        if self.regime_ == "standard":
            self._fit_standard(X_vt, y_enc, n, p, skf, scoring)
        elif self.regime_ == "high_dim":
            self._fit_high_dim(X_vt, y_enc, n, p, skf, scoring, effective_l1_ratio)
        elif self.regime_ == "large":
            self._fit_large(X_vt, y_enc, n, p, skf, scoring, effective_l1_ratio)
        else:
            raise ValueError(f"Unknown regime: {self.regime_!r}. Expected 'standard', 'high_dim', or 'large'.")

        # --- Consolidated feature mask ---
        vt_support = self._vt.get_support()
        if self._sfm is not None:
            sfm_support = self._sfm.get_support()
            combined = np.zeros(p, dtype=bool)
            idx = np.where(vt_support)[0]
            combined[idx[sfm_support]] = True
            self.feature_mask_ = combined
        else:
            self.feature_mask_ = vt_support

        # --- Coef extraction ---
        if hasattr(self._estimator, "coef_"):
            self.coef_ = self._estimator.coef_
        else:
            self.coef_ = None

        self.classes_ = self._label_encoder.classes_

        logger.rule("Done")
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        X_t = self._transform(X)
        y_enc = self._estimator.predict(X_t)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        X_t = self._transform(X)
        return self._estimator.predict_proba(X_t)

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_mask_"])
        return np.where(self.feature_mask_)[0].astype(str)

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Requires the optional ``onnx`` extras:
            pip install zslinear[onnx]

        Accepts a float32 input named ``"float_input"`` with shape
        ``(n_samples, n_features_in_)`` (raw, unprocessed features).
        Produces:
          - ``"output_label"``       int64  (n_samples,)   — predicted class
          - ``"output_probability"`` float32 (n_samples, 2) — [P(class_0), P(class_1)]

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, attributes=["_estimator"])
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "ONNX export requires optional dependencies. "
                "Install them with:  pip install zslinear[onnx]"
            )
        from sklearn.pipeline import Pipeline

        steps = [("vt", self._vt)]
        if self._sfm is not None:
            steps.append(("sfm", self._sfm))
        steps.append(("clf", self._estimator))

        pipeline = Pipeline(steps)
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    # ------------------------------------------------------------------
    # Internal: transform helpers
    # ------------------------------------------------------------------

    def _transform(self, X) -> np.ndarray:
        X_vt = self._vt.transform(X)
        if self._sfm is not None:
            return self._sfm.transform(X_vt)
        return X_vt

    # ------------------------------------------------------------------
    # Internal: regime-specific fit methods
    # ------------------------------------------------------------------

    def _fit_standard(self, X, y, n, p, skf, scoring):
        self._sfm = None
        C_grid = np.asarray(self.C_values) if self.C_values is not None else _default_C_grid(n, p)

        self._estimator = LogisticRegressionCV(
            Cs=C_grid,
            cv=skf,
            penalty="l1",
            solver="liblinear",
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=True,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y)

    def _fit_high_dim(self, X, y, n, p, skf, scoring, l1_ratio):
        # Pre-filter with fast liblinear L1 to reduce dimensionality
        max_feat = _sfm_max_features(n, p)
        pre_lr = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._sfm = SelectFromModel(estimator=pre_lr, max_features=max_feat, threshold=-np.inf)
        self._sfm.fit(X, y)
        X_sfm = self._sfm.transform(X)

        if X_sfm.shape[1] == 0:
            # Pathological fallback: keep at least 1 feature
            self._sfm = SelectFromModel(estimator=pre_lr, max_features=1, threshold=-np.inf)
            self._sfm.fit(X, y)
            X_sfm = self._sfm.transform(X)

        C_grid = np.asarray(self.C_values) if self.C_values is not None else _default_C_grid(n, X_sfm.shape[1])

        self._estimator = LogisticRegressionCV(
            Cs=C_grid,
            cv=skf,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[l1_ratio],
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=True,
            random_state=self.random_state,
        )
        self._estimator.fit(X_sfm, y)

    def _fit_large(self, X, y, n, p, skf, scoring, l1_ratio):
        self._sfm = None
        alpha_grid = np.asarray(self.alpha_values) if self.alpha_values is not None else _default_alpha_grid(n, p)

        # Tune alpha on a stratified subsample
        sub_size = min(self.tuning_subsample, n)
        if sub_size < n:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=sub_size, random_state=self.random_state)
            sub_idx, _ = next(sss.split(X, y))
            X_sub, y_sub = X[sub_idx], y[sub_idx]
        else:
            X_sub, y_sub = X, y

        base_sgd = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state,
        )

        # Adjust CV folds for subsample size
        sub_min_class = int(np.bincount(y_sub).min())
        sub_cv = min(skf.n_splits, sub_min_class)
        if sub_cv < 2:
            sub_cv = 2
        sub_skf = StratifiedKFold(n_splits=sub_cv, shuffle=True, random_state=self.random_state)

        gs = GridSearchCV(
            estimator=base_sgd,
            param_grid={"alpha": alpha_grid},
            cv=sub_skf,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=False,
        )
        gs.fit(X_sub, y_sub)
        best_alpha = gs.best_params_["alpha"]

        # Refit on full data with best alpha
        self._estimator = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=best_alpha,
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y)


# ---------------------------------------------------------------------------
# Regression helper
# ---------------------------------------------------------------------------

def _cv_from_n(n: int) -> int:
    """Adaptive CV folds for regression (no class constraint)."""
    if n < 200:
        return min(n // 10, 10)
    if n < 1000:
        return 5
    return 3


def _regression_scoring(y: np.ndarray) -> str:
    """Auto-select CV metric based on target skewness.

    Skewness > 1.0 (moderately skewed) → neg_mean_absolute_error, which is
    more robust to outliers than R². Otherwise R².
    """
    from scipy.stats import skew as scipy_skew
    sk = abs(float(scipy_skew(y)))
    return "neg_mean_absolute_error" if sk > 1.0 else "r2"


# ---------------------------------------------------------------------------
# Linear regressor
# ---------------------------------------------------------------------------

class LinearRegressor:
    """
    Linear regression with embedded feature selection.

    Automatically selects solver and regularization regime based on (n, p):
      - standard  (n<=50K, p<=n): RidgeCV (L2),               k-fold CV
      - high_dim  (n<=50K, p>n):  ElasticNetCV,               SelectFromModel(Lasso) pre-filter
      - large     (n>50K):        SGDRegressor + ElasticNet,  subsample alpha tuning

    Feature scaling is NOT applied internally. Standardize X before calling fit().
    CV scoring is auto-selected based on target skewness (R² or neg_MAE).

    Parameters
    ----------
    regime : str or None
        Force a specific regime. None = auto-detect from data shape.
    alpha_values : array-like or None
        Override the adaptive regularization strength grid.
    l1_ratio : float or None
        ElasticNet mixing (0=L2, 1=L1). None = auto from p/n heuristic.
        Only used in high_dim and large regimes.
    cv : int or None
        Number of CV folds. None = auto from n.
    max_iter : int
        Max iterations for iterative solvers.
    n_jobs : int
        Parallelism for CV.
    random_state : int or None
        Reproducibility seed.
    variance_threshold : float
        VarianceThreshold cutoff. 0.0 removes only constant features.
    tuning_subsample : int
        Max rows used for alpha-grid tuning in the large regime.
    verbose : int
        0 = silent, 1 = rule banners via Rich/loguru.
    """

    def __init__(
        self,
        *,
        regime: str | None = None,
        alpha_values: list | np.ndarray | None = None,
        l1_ratio: float | None = None,
        cv: int | None = None,
        max_iter: int = 10_000,
        n_jobs: int = -1,
        random_state: int | None = 42,
        variance_threshold: float = 0.0,
        tuning_subsample: int = 10_000,
        verbose: int = 0,
    ):
        self.regime = regime
        self.alpha_values = alpha_values
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.tuning_subsample = tuning_subsample
        self.verbose = verbose

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "LinearRegressor":
        logger.set_verbosity(bool(self.verbose))
        logger.rule("LinearRegressor")

        X = check_array(X, dtype="numeric", accept_sparse="csr")
        y = np.asarray(y, dtype=float)

        n, p = X.shape
        self.n_features_in_ = p

        # --- Regime ---
        self.regime_ = self.regime if self.regime is not None else _detect_regime(n, p)

        # --- Heuristics ---
        effective_l1_ratio = self.l1_ratio if self.l1_ratio is not None else _default_l1_ratio(n, p)
        effective_cv = self.cv if self.cv is not None else _cv_from_n(n)
        scoring = _regression_scoring(y)

        kf = KFold(n_splits=max(2, effective_cv), shuffle=True, random_state=self.random_state)

        # --- Preprocessing: VarianceThreshold only ---
        self._vt = VarianceThreshold(threshold=self.variance_threshold)
        try:
            X_vt = self._vt.fit_transform(X)
        except ValueError:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        if X_vt.shape[1] == 0:
            raise ValueError(
                "All features have zero variance after VarianceThreshold. "
                "Check your input data."
            )

        # --- Dispatch ---
        if self.regime_ == "standard":
            self._fit_standard(X_vt, y, n, p, kf, scoring)
        elif self.regime_ == "high_dim":
            self._fit_high_dim(X_vt, y, n, p, kf, scoring, effective_l1_ratio)
        elif self.regime_ == "large":
            self._fit_large(X_vt, y, n, p, kf, scoring, effective_l1_ratio)
        else:
            raise ValueError(f"Unknown regime: {self.regime_!r}.")

        # --- Consolidated feature mask ---
        vt_support = self._vt.get_support()
        if self._sfm is not None:
            sfm_support = self._sfm.get_support()
            combined = np.zeros(p, dtype=bool)
            idx = np.where(vt_support)[0]
            combined[idx[sfm_support]] = True
            self.feature_mask_ = combined
        else:
            self.feature_mask_ = vt_support

        # --- Coef extraction ---
        if hasattr(self._estimator, "coef_"):
            self.coef_ = self._estimator.coef_
        else:
            self.coef_ = None

        logger.rule("Done")
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["_estimator"])
        X = check_array(X, dtype="numeric", accept_sparse="csr")
        return self._estimator.predict(self._transform(X))

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_mask_"])
        return np.where(self.feature_mask_)[0].astype(str)

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Requires the optional ``onnx`` extras:
            pip install zslinear[onnx]

        Accepts a float32 input named ``"float_input"`` with shape
        ``(n_samples, n_features_in_)`` (raw, unprocessed features).
        Produces a single output ``"variable"`` (float32, shape [n_samples, 1]).

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, attributes=["_estimator"])
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "ONNX export requires optional dependencies. "
                "Install them with:  pip install zslinear[onnx]"
            )
        from sklearn.pipeline import Pipeline

        steps = [("vt", self._vt)]
        if self._sfm is not None:
            steps.append(("sfm", self._sfm))
        steps.append(("reg", self._estimator))

        pipeline = Pipeline(steps)
        initial_types = [("float_input", FloatTensorType([None, self.n_features_in_]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transform(self, X) -> np.ndarray:
        X_vt = self._vt.transform(X)
        if self._sfm is not None:
            return self._sfm.transform(X_vt)
        return X_vt

    def _fit_standard(self, X, y, n, p, kf, scoring):
        self._sfm = None
        alpha_grid = np.asarray(self.alpha_values) if self.alpha_values is not None else _default_alpha_grid(n, p)

        self._estimator = RidgeCV(
            alphas=alpha_grid,
            cv=kf,
            scoring=scoring,
        )
        self._estimator.fit(X, y)

    def _fit_high_dim(self, X, y, n, p, kf, scoring, l1_ratio):
        # Pre-filter with Lasso to reduce dimensionality
        max_feat = _sfm_max_features(n, p)
        pre_lasso = Lasso(alpha=1.0, max_iter=self.max_iter)
        self._sfm = SelectFromModel(estimator=pre_lasso, max_features=max_feat, threshold=-np.inf)
        self._sfm.fit(X, y)
        X_sfm = self._sfm.transform(X)

        if X_sfm.shape[1] == 0:
            self._sfm = SelectFromModel(estimator=pre_lasso, max_features=1, threshold=-np.inf)
            self._sfm.fit(X, y)
            X_sfm = self._sfm.transform(X)

        alpha_grid = np.asarray(self.alpha_values) if self.alpha_values is not None else _default_alpha_grid(n, X_sfm.shape[1])

        self._estimator = ElasticNetCV(
            alphas=alpha_grid,
            l1_ratio=[l1_ratio],
            cv=kf,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
        )
        self._estimator.fit(X_sfm, y)

    def _fit_large(self, X, y, n, p, kf, scoring, l1_ratio):
        self._sfm = None
        alpha_grid = np.asarray(self.alpha_values) if self.alpha_values is not None else _default_alpha_grid(n, p)

        # Tune alpha on a random subsample
        sub_size = min(self.tuning_subsample, n)
        if sub_size < n:
            rng = np.random.default_rng(self.random_state)
            sub_idx = rng.choice(n, size=sub_size, replace=False)
            X_sub, y_sub = X[sub_idx], y[sub_idx]
        else:
            X_sub, y_sub = X, y

        base_sgd = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=self.random_state,
        )

        sub_cv = min(kf.n_splits, max(2, sub_size // 50))
        sub_kf = KFold(n_splits=sub_cv, shuffle=True, random_state=self.random_state)

        gs = GridSearchCV(
            estimator=base_sgd,
            param_grid={"alpha": alpha_grid},
            cv=sub_kf,
            scoring=scoring,
            n_jobs=self.n_jobs,
            refit=False,
        )
        gs.fit(X_sub, y_sub)
        best_alpha = gs.best_params_["alpha"]

        self._estimator = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=best_alpha,
            max_iter=1000,
            random_state=self.random_state,
        )
        self._estimator.fit(X, y)
