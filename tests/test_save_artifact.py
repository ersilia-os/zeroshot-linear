"""
Tests for LinearClassifier.save(), LinearRegressor.save(), and LinearArtifact.

Covers:
  - Files written to disk (onnx and joblib paths)
  - Metadata correctness (linear.json)
  - LinearArtifact.load() + artifact.run() output shapes and value ranges
  - artifact.run() agrees with predict_proba / predict within tolerance
  - Error cases (missing directory, missing files, not-fitted)
"""

import json
import os

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError

from zslinear.model import LinearArtifact, LinearClassifier, LinearRegressor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_standard():
    X, y = make_classification(
        n_samples=400, n_features=60, n_informative=10, n_redundant=15,
        weights=[0.7, 0.3], random_state=0,
    )
    clf = LinearClassifier(random_state=0)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def clf_high_dim():
    X, y = make_classification(
        n_samples=100, n_features=250, n_informative=10, n_redundant=50,
        weights=[0.8, 0.2], random_state=1,
    )
    clf = LinearClassifier(random_state=1)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def reg_standard():
    X, y = make_regression(
        n_samples=400, n_features=60, n_informative=15, noise=10.0, random_state=0,
    )
    reg = LinearRegressor(random_state=0)
    reg.fit(X, y)
    return reg, X, y


@pytest.fixture
def reg_high_dim():
    X, y = make_regression(
        n_samples=100, n_features=250, n_informative=10, noise=5.0, random_state=2,
    )
    reg = LinearRegressor(random_state=2)
    reg.fit(X, y)
    return reg, X, y


# ---------------------------------------------------------------------------
# LinearClassifier.save — files on disk
# ---------------------------------------------------------------------------

class TestClassifierSave:
    def test_onnx_creates_expected_files(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "model"), onnx=True)
        assert (tmp_path / "model" / "linear.onnx").exists()
        assert (tmp_path / "model" / "linear.json").exists()
        assert not (tmp_path / "model" / "linear.joblib").exists()

    def test_joblib_creates_expected_files(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "model"), onnx=False)
        assert (tmp_path / "model" / "linear.joblib").exists()
        assert (tmp_path / "model" / "linear.json").exists()
        assert not (tmp_path / "model" / "linear.onnx").exists()

    def test_creates_directory_if_absent(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        dest = str(tmp_path / "new_dir" / "nested")
        clf.save(dest, onnx=True)
        assert os.path.isdir(dest)

    def test_metadata_fields_onnx(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        with open(tmp_path / "m" / "linear.json") as f:
            meta = json.load(f)
        assert meta["task"] == "classification"
        assert meta["format"] == "onnx"
        assert meta["regime"] == "standard"
        assert meta["n_features_in"] == X.shape[1]
        assert set(meta["classes"]) == {0, 1}
        assert len(meta["feature_mask"]) == X.shape[1]

    def test_metadata_fields_joblib(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=False)
        with open(tmp_path / "m" / "linear.json") as f:
            meta = json.load(f)
        assert meta["format"] == "joblib"

    def test_metadata_high_dim(self, tmp_path, clf_high_dim):
        clf, X, y = clf_high_dim
        clf.save(str(tmp_path / "m"), onnx=True)
        with open(tmp_path / "m" / "linear.json") as f:
            meta = json.load(f)
        assert meta["regime"] == "high_dim"

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(NotFittedError):
            LinearClassifier().save(str(tmp_path / "m"))


# ---------------------------------------------------------------------------
# LinearRegressor.save — files on disk
# ---------------------------------------------------------------------------

class TestRegressorSave:
    def test_onnx_creates_expected_files(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "model"), onnx=True)
        assert (tmp_path / "model" / "linear.onnx").exists()
        assert (tmp_path / "model" / "linear.json").exists()

    def test_joblib_creates_expected_files(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "model"), onnx=False)
        assert (tmp_path / "model" / "linear.joblib").exists()
        assert (tmp_path / "model" / "linear.json").exists()

    def test_metadata_fields_onnx(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=True)
        with open(tmp_path / "m" / "linear.json") as f:
            meta = json.load(f)
        assert meta["task"] == "regression"
        assert meta["format"] == "onnx"
        assert meta["n_features_in"] == X.shape[1]
        assert "classes" not in meta

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(NotFittedError):
            LinearRegressor().save(str(tmp_path / "m"))


# ---------------------------------------------------------------------------
# LinearArtifact — classifier, onnx format
# ---------------------------------------------------------------------------

class TestArtifactClassifierOnnx:
    def test_run_shape(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y), 2)

    def test_run_probabilities_sum_to_one(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(len(y)), atol=1e-5)

    def test_run_values_in_unit_interval(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert (out >= 0).all() and (out <= 1).all()

    def test_run_matches_predict_proba(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        art_out = art.run(X)
        sk_out = clf.predict_proba(X)
        np.testing.assert_allclose(art_out, sk_out, rtol=1e-4, atol=1e-4)

    def test_run_high_dim(self, tmp_path, clf_high_dim):
        clf, X, y = clf_high_dim
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y), 2)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(len(y)), atol=1e-5)

    def test_task_attribute(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        assert art.task == "classification"


# ---------------------------------------------------------------------------
# LinearArtifact — classifier, joblib format
# ---------------------------------------------------------------------------

class TestArtifactClassifierJoblib:
    def test_run_shape(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y), 2)

    def test_run_matches_predict_proba(self, tmp_path, clf_standard):
        clf, X, y = clf_standard
        clf.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        np.testing.assert_allclose(art.run(X), clf.predict_proba(X), rtol=1e-6)

    def test_run_high_dim(self, tmp_path, clf_high_dim):
        clf, X, y = clf_high_dim
        clf.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y), 2)


# ---------------------------------------------------------------------------
# LinearArtifact — regressor, onnx format
# ---------------------------------------------------------------------------

class TestArtifactRegressorOnnx:
    def test_run_shape(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y),)

    def test_run_finite(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        assert np.isfinite(art.run(X)).all()

    def test_run_matches_predict(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        np.testing.assert_allclose(art.run(X), reg.predict(X), rtol=1e-4, atol=1e-4)

    def test_run_high_dim(self, tmp_path, reg_high_dim):
        reg, X, y = reg_high_dim
        reg.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y),)
        assert np.isfinite(out).all()

    def test_task_attribute(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=True)
        art = LinearArtifact.load(str(tmp_path / "m"))
        assert art.task == "regression"


# ---------------------------------------------------------------------------
# LinearArtifact — regressor, joblib format
# ---------------------------------------------------------------------------

class TestArtifactRegressorJoblib:
    def test_run_shape(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y),)

    def test_run_matches_predict(self, tmp_path, reg_standard):
        reg, X, y = reg_standard
        reg.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        np.testing.assert_allclose(art.run(X), reg.predict(X), rtol=1e-4, atol=1e-4)

    def test_run_high_dim(self, tmp_path, reg_high_dim):
        reg, X, y = reg_high_dim
        reg.save(str(tmp_path / "m"), onnx=False)
        art = LinearArtifact.load(str(tmp_path / "m"))
        out = art.run(X)
        assert out.shape == (len(y),)


# ---------------------------------------------------------------------------
# LinearArtifact — error cases
# ---------------------------------------------------------------------------

class TestArtifactErrors:
    def test_load_missing_metadata_raises(self, tmp_path):
        os.makedirs(str(tmp_path / "empty"), exist_ok=True)
        with pytest.raises(FileNotFoundError, match="linear.json"):
            LinearArtifact.load(str(tmp_path / "empty"))

    def test_load_missing_onnx_file_raises(self, tmp_path):
        # Write metadata claiming onnx but no model file
        d = tmp_path / "bad"
        d.mkdir()
        with open(d / "linear.json", "w") as f:
            json.dump({"task": "classification", "format": "onnx"}, f)
        with pytest.raises(FileNotFoundError, match="linear.onnx"):
            LinearArtifact.load(str(d))

    def test_load_missing_joblib_file_raises(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        with open(d / "linear.json", "w") as f:
            json.dump({"task": "regression", "format": "joblib"}, f)
        with pytest.raises(FileNotFoundError, match="linear.joblib"):
            LinearArtifact.load(str(d))

    def test_run_before_load_raises(self):
        art = LinearArtifact()
        X = np.random.default_rng(0).normal(size=(10, 5)).astype(np.float32)
        with pytest.raises(RuntimeError, match="No model loaded"):
            art.run(X)
