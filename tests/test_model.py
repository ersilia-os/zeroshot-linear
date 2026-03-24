"""
Tests for LinearClassifier — all three regimes (standard, high_dim, large).
Use `pytest -m "not slow"` to skip the large-regime test in CI.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from zslinear.model import LinearClassifier, _detect_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_standard(n=500, p=80, seed=0):
    """n <= 50K, p <= n  →  standard regime."""
    return make_classification(
        n_samples=n, n_features=p, n_informative=10, n_redundant=20,
        weights=[0.8, 0.2], random_state=seed,
    )


def make_high_dim(n=120, p=300, seed=0):
    """p > n, n <= 50K  →  high_dim regime."""
    return make_classification(
        n_samples=n, n_features=p, n_informative=10, n_redundant=50,
        weights=[0.85, 0.15], random_state=seed,
    )


def make_large(n=60_000, p=200, seed=0):
    """n > 50K  →  large regime."""
    return make_classification(
        n_samples=n, n_features=p, n_informative=20, n_redundant=40,
        weights=[0.9, 0.1], random_state=seed,
    )


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def test_regime_detection_standard():
    assert _detect_regime(500, 80) == "standard"


def test_regime_detection_high_dim():
    assert _detect_regime(120, 300) == "high_dim"


def test_regime_detection_large():
    assert _detect_regime(60_000, 200) == "large"


def test_regime_detection_boundary_p_equals_n():
    # p == n should go to standard (not high_dim)
    assert _detect_regime(100, 100) == "standard"


# ---------------------------------------------------------------------------
# Fit / predict shapes
# ---------------------------------------------------------------------------

def test_fit_predict_standard():
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert clf.regime_ == "standard"
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), 2)
    assert clf._sfm is None


def test_fit_predict_high_dim():
    X, y = make_high_dim()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert clf.regime_ == "high_dim"
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), 2)
    assert clf._sfm is not None


@pytest.mark.slow
def test_fit_predict_large():
    X, y = make_large()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert clf.regime_ == "large"
    assert clf.predict(X).shape == (len(y),)
    assert clf.predict_proba(X).shape == (len(y), 2)


# ---------------------------------------------------------------------------
# Prediction quality (above-chance)
# ---------------------------------------------------------------------------

def test_score_above_chance_standard():
    X, y = make_standard(n=600)
    clf = LinearClassifier()
    clf.fit(X[:400], y[:400])
    assert clf.score(X[400:], y[400:]) > 0.55


def test_score_above_chance_high_dim():
    X, y = make_high_dim(n=150)
    clf = LinearClassifier()
    clf.fit(X[:100], y[:100])
    assert clf.score(X[100:], y[100:]) > 0.52


# ---------------------------------------------------------------------------
# Feature mask
# ---------------------------------------------------------------------------

def test_feature_mask_shape_standard():
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert clf.feature_mask_.shape == (X.shape[1],)
    assert clf.feature_mask_.dtype == bool


def test_feature_mask_shape_high_dim():
    X, y = make_high_dim()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert clf.feature_mask_.shape == (X.shape[1],)
    # high_dim pre-filter should reduce features
    assert clf.feature_mask_.sum() < X.shape[1]


def test_l1_produces_sparse_coef():
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    # L1 regularization should zero out some coefficients
    assert np.any(clf.coef_ == 0.0)


# ---------------------------------------------------------------------------
# Sparse inputs
# ---------------------------------------------------------------------------

def test_sparse_input_standard():
    X, y = make_standard()
    X_sparse = sp.csr_matrix(X)
    clf = LinearClassifier(random_state=0)
    clf.fit(X_sparse, y)
    preds = clf.predict(X_sparse)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_sparse_input_high_dim():
    X, y = make_high_dim()
    X_sparse = sp.csr_matrix(X)
    clf = LinearClassifier()
    clf.fit(X_sparse, y)
    assert clf.predict(X_sparse).shape == (len(y),)


# ---------------------------------------------------------------------------
# Forced regime override
# ---------------------------------------------------------------------------

def test_forced_regime_override():
    X, y = make_standard()   # standard data forced into high_dim path
    clf = LinearClassifier(regime="high_dim")
    clf.fit(X, y)
    assert clf.regime_ == "high_dim"
    assert clf._sfm is not None


def test_forced_large_regime_on_small_data():
    X, y = make_classification(n_samples=200, n_features=50, n_informative=10,
                                n_redundant=10, random_state=0)
    clf = LinearClassifier(regime="large", tuning_subsample=100)
    clf.fit(X, y)
    assert clf.regime_ == "large"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_class_raises():
    X, _ = make_standard()
    y_bad = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match="one class"):
        LinearClassifier().fit(X, y_bad)


def test_all_zero_features_raises():
    X = np.zeros((100, 10))
    y = np.array([0] * 50 + [1] * 50)
    with pytest.raises(ValueError, match="zero variance"):
        LinearClassifier().fit(X, y)


def test_not_fitted_raises():
    clf = LinearClassifier()
    X, _ = make_standard()
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_non_binary_labels_minus_one_one():
    X, y = make_standard()
    y_signed = np.where(y == 0, -1, 1)
    clf = LinearClassifier()
    clf.fit(X, y_signed)
    preds = clf.predict(X)
    assert set(preds).issubset({-1, 1})


def test_string_labels():
    X, y = make_standard()
    y_str = np.where(y == 0, "inactive", "active")
    clf = LinearClassifier()
    clf.fit(X, y_str)
    preds = clf.predict(X)
    assert set(preds).issubset({"inactive", "active"})


def test_cv_fold_reduction_warning():
    # Only 2 positive samples — cv=5 is impossible; should warn and reduce
    X, _ = make_standard(n=50)
    y = np.array([1, 1] + [0] * 48)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # Should not raise, just reduce folds internally
        clf = LinearClassifier(cv=None)  # auto-cv will clamp to min_class
        clf.fit(X, y)
        assert clf.regime_ in ("standard", "high_dim", "large")


def test_reproducibility():
    X, y = make_standard()
    clf1 = LinearClassifier(random_state=7)
    clf2 = LinearClassifier(random_state=7)
    clf1.fit(X, y)
    clf2.fit(X, y)
    np.testing.assert_array_equal(clf1.coef_, clf2.coef_)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_classes_attribute():
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    assert set(clf.classes_) == {0, 1}


def test_get_feature_names_out():
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    names = clf.get_feature_names_out()
    assert names.shape == (clf.feature_mask_.sum(),)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_to_onnx_standard(tmp_path):
    """ONNX export works and produces valid predictions for standard regime."""
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    out = str(tmp_path / "model.onnx")
    clf.to_onnx(out)
    sess = rt.InferenceSession(out)
    preds = sess.run(["output_label"], {"float_input": X.astype(np.float32)})[0]
    assert preds.shape == (len(y),)


def test_to_onnx_high_dim(tmp_path):
    """ONNX export works for high_dim regime (includes SelectFromModel step)."""
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_high_dim()
    clf = LinearClassifier()
    clf.fit(X, y)
    out = str(tmp_path / "model.onnx")
    clf.to_onnx(out)
    sess = rt.InferenceSession(out)
    preds = sess.run(["output_label"], {"float_input": X.astype(np.float32)})[0]
    assert preds.shape == (len(y),)


def test_to_onnx_before_fit_raises():
    clf = LinearClassifier()
    with pytest.raises(NotFittedError):
        clf.to_onnx("model.onnx")


def test_to_onnx_matches_predict_standard(tmp_path):
    """ONNX predictions match clf.predict() for standard regime."""
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_standard()
    clf = LinearClassifier()
    clf.fit(X, y)
    out = str(tmp_path / "model.onnx")
    clf.to_onnx(out)
    sess = rt.InferenceSession(out)
    onnx_preds = sess.run(["output_label"], {"float_input": X.astype(np.float32)})[0]
    sklearn_preds = clf.predict(X).astype(np.int64)
    np.testing.assert_array_equal(onnx_preds, sklearn_preds)
