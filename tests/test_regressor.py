"""
Tests for LinearRegressor — all three regimes (standard, high_dim, large).
Use `pytest -m "not slow"` to skip the large-regime test in CI.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError

from zslinear.model import LinearRegressor, _detect_regime, _regression_scoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reg_standard(n=500, p=80, seed=0):
    """n <= 50K, p <= n  →  standard regime."""
    return make_regression(n_samples=n, n_features=p, n_informative=15,
                           noise=10.0, random_state=seed)


def make_reg_high_dim(n=120, p=300, seed=0):
    """p > n, n <= 50K  →  high_dim regime."""
    return make_regression(n_samples=n, n_features=p, n_informative=10,
                           noise=5.0, random_state=seed)


def make_reg_large(n=60_000, p=200, seed=0):
    """n > 50K  →  large regime."""
    return make_regression(n_samples=n, n_features=p, n_informative=30,
                           noise=10.0, random_state=seed)


# ---------------------------------------------------------------------------
# Scoring heuristic
# ---------------------------------------------------------------------------

def test_scoring_symmetric():
    y = np.random.default_rng(0).normal(0, 1, 500)
    assert _regression_scoring(y) == "r2"


def test_scoring_skewed():
    y = np.random.default_rng(0).exponential(1.0, 500)  # highly right-skewed
    assert _regression_scoring(y) == "neg_mean_absolute_error"


# ---------------------------------------------------------------------------
# Fit / predict shapes
# ---------------------------------------------------------------------------

def test_fit_predict_standard():
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.regime_ == "standard"
    assert reg.predict(X).shape == (len(y),)
    assert reg._sfm is None


def test_fit_predict_high_dim():
    X, y = make_reg_high_dim()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.regime_ == "high_dim"
    assert reg.predict(X).shape == (len(y),)
    assert reg._sfm is not None


@pytest.mark.slow
def test_fit_predict_large():
    X, y = make_reg_large()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.regime_ == "large"
    assert reg.predict(X).shape == (len(y),)


# ---------------------------------------------------------------------------
# Prediction quality
# ---------------------------------------------------------------------------

def test_score_above_baseline_standard():
    X, y = make_reg_standard(n=600)
    reg = LinearRegressor()
    reg.fit(X[:400], y[:400])
    assert reg.score(X[400:], y[400:]) > 0.1


def test_score_above_baseline_high_dim():
    X, y = make_reg_high_dim(n=150)
    reg = LinearRegressor()
    reg.fit(X[:100], y[:100])
    assert reg.score(X[100:], y[100:]) > 0.1


# ---------------------------------------------------------------------------
# Feature mask
# ---------------------------------------------------------------------------

def test_feature_mask_shape_standard():
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.feature_mask_.shape == (X.shape[1],)
    assert reg.feature_mask_.dtype == bool


def test_feature_mask_shape_high_dim():
    X, y = make_reg_high_dim()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.feature_mask_.shape == (X.shape[1],)
    assert reg.feature_mask_.sum() < X.shape[1]


def test_coef_not_none():
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    assert reg.coef_ is not None


# ---------------------------------------------------------------------------
# Sparse inputs
# ---------------------------------------------------------------------------

def test_sparse_input_standard():
    X, y = make_reg_standard()
    X_sparse = sp.csr_matrix(X)
    reg = LinearRegressor()
    reg.fit(X_sparse, y)
    preds = reg.predict(X_sparse)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_sparse_input_high_dim():
    X, y = make_reg_high_dim()
    X_sparse = sp.csr_matrix(X)
    reg = LinearRegressor()
    reg.fit(X_sparse, y)
    assert reg.predict(X_sparse).shape == (len(y),)


# ---------------------------------------------------------------------------
# Forced regime override
# ---------------------------------------------------------------------------

def test_forced_regime_override():
    X, y = make_reg_standard()
    reg = LinearRegressor(regime="high_dim")
    reg.fit(X, y)
    assert reg.regime_ == "high_dim"
    assert reg._sfm is not None


def test_forced_large_regime_on_small_data():
    X, y = make_regression(n_samples=200, n_features=50, n_informative=10,
                            noise=5.0, random_state=0)
    reg = LinearRegressor(regime="large", tuning_subsample=100)
    reg.fit(X, y)
    assert reg.regime_ == "large"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_zero_features_raises():
    X = np.zeros((100, 10))
    y = np.random.default_rng(0).normal(0, 1, 100)
    with pytest.raises(ValueError, match="zero variance"):
        LinearRegressor().fit(X, y)


def test_not_fitted_raises():
    reg = LinearRegressor()
    X, _ = make_reg_standard()
    with pytest.raises(NotFittedError):
        reg.predict(X)


def test_to_onnx_before_fit_raises():
    reg = LinearRegressor()
    with pytest.raises(NotFittedError):
        reg.to_onnx("model.onnx")


def test_reproducibility():
    X, y = make_reg_standard()
    reg1 = LinearRegressor(random_state=7)
    reg2 = LinearRegressor(random_state=7)
    reg1.fit(X, y)
    reg2.fit(X, y)
    np.testing.assert_array_equal(reg1.coef_, reg2.coef_)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_get_feature_names_out():
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    names = reg.get_feature_names_out()
    assert names.shape == (reg.feature_mask_.sum(),)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_to_onnx_standard(tmp_path):
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    out = str(tmp_path / "reg.onnx")
    reg.to_onnx(out)
    sess = rt.InferenceSession(out)
    output_names = [o.name for o in sess.get_outputs()]
    preds = sess.run(output_names[:1], {"float_input": X.astype(np.float32)})[0]
    assert preds.shape[0] == len(y)


def test_to_onnx_high_dim(tmp_path):
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_reg_high_dim()
    reg = LinearRegressor()
    reg.fit(X, y)
    out = str(tmp_path / "reg_hd.onnx")
    reg.to_onnx(out)
    sess = rt.InferenceSession(out)
    output_names = [o.name for o in sess.get_outputs()]
    preds = sess.run(output_names[:1], {"float_input": X.astype(np.float32)})[0]
    assert preds.shape[0] == len(y)


def test_to_onnx_matches_predict_standard(tmp_path):
    """ONNX predictions match reg.predict() for standard regime."""
    pytest.importorskip("skl2onnx")
    import onnxruntime as rt
    X, y = make_reg_standard()
    reg = LinearRegressor()
    reg.fit(X, y)
    out = str(tmp_path / "reg_match.onnx")
    reg.to_onnx(out)
    sess = rt.InferenceSession(out)
    output_names = [o.name for o in sess.get_outputs()]
    onnx_preds = sess.run(output_names[:1], {"float_input": X.astype(np.float32)})[0].ravel()
    sklearn_preds = reg.predict(X)
    np.testing.assert_allclose(onnx_preds, sklearn_preds, rtol=1e-4, atol=1e-4)
