"""Head invariants: no test-set leakage, and honest behaviour at tiny N.

The README claims "every head sits behind a StandardScaler fitted inside the
pipeline, so nothing from the test set reaches the fit".  That is testable
without inspecting the pipeline: if no test-set statistic influences the fit or
the transform, then a given test row's prediction cannot depend on which *other*
test rows were passed alongside it.  A scaler accidentally fitted on the
combined data -- the classic low-N leak -- fails that immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from lown.heads import PRIMARY_METRIC, _inner_folds, build_head, fit_predict, score

HEADS = ["linear", "mlp", "gbm"]


def toy_classification(n=120, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def toy_regression(n=120, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.3 * rng.standard_normal(n)
    return X, y


# --------------------------------------------------- the no-leakage claim ----
@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_prediction_for_a_row_is_independent_of_the_rest_of_the_test_set(head, task):
    """The leakage test that matters.

    Predict on the whole test set, then on a single row alone.  If the scaler
    (or anything else) had been fitted on test data, the two predictions for
    that row would differ.
    """
    X, y = toy_classification() if task == "classification" else toy_regression()
    X_tr, y_tr, X_te = X[:90], y[:90], X[90:]

    full = fit_predict(head, task, X_tr, y_tr, X_te, seed=0)
    alone = fit_predict(head, task, X_tr, y_tr, X_te[:1], seed=0)
    assert full[0] == pytest.approx(alone[0], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("head", HEADS)
def test_test_set_order_only_permutes_predictions(head):
    """A fit that peeked at test-set statistics would not be permutation-equivariant."""
    X, y = toy_classification()
    X_tr, y_tr, X_te = X[:90], y[:90], X[90:]

    order = np.random.default_rng(1).permutation(len(X_te))
    straight = fit_predict(head, "classification", X_tr, y_tr, X_te, seed=0)
    shuffled = fit_predict(head, "classification", X_tr, y_tr, X_te[order], seed=0)
    assert straight[order] == pytest.approx(shuffled, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("head", HEADS)
def test_duplicating_the_test_set_does_not_change_predictions(head):
    """Doubling the test rows changes test-set statistics but must not move the fit."""
    X, y = toy_classification()
    X_tr, y_tr, X_te = X[:90], y[:90], X[90:]

    once = fit_predict(head, "classification", X_tr, y_tr, X_te, seed=0)
    twice = fit_predict(head, "classification", X_tr, y_tr, np.vstack([X_te, X_te]), seed=0)
    assert once == pytest.approx(twice[: len(once)], rel=1e-9, abs=1e-12)
    assert once == pytest.approx(twice[len(once) :], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("head", HEADS)
def test_every_head_standardises_inside_the_pipeline(head):
    """Structural check to complement the behavioural ones."""
    from sklearn.preprocessing import StandardScaler

    pipe = build_head(head, "classification", n_train=100, seed=0, y_train=np.arange(100) % 2)
    assert isinstance(pipe[0], StandardScaler)


@pytest.mark.parametrize("head", HEADS)
def test_shifting_the_test_features_wildly_leaves_the_model_alone(head):
    """A leaked scaler would absorb an absurd test-set shift; a clean one will not.

    Predictions on shifted rows should change (the model sees different inputs),
    but predictions on the *unshifted* rows must be untouched.
    """
    X, y = toy_regression()
    X_tr, y_tr, X_te = X[:90], y[:90], X[90:]
    contaminated = np.vstack([X_te, X_te + 1000.0])

    clean = fit_predict(head, "regression", X_tr, y_tr, X_te, seed=0)
    mixed = fit_predict(head, "regression", X_tr, y_tr, contaminated, seed=0)
    assert clean == pytest.approx(mixed[: len(clean)], rel=1e-9, abs=1e-9)


# ------------------------------------------------ tiny-N / degenerate cases --
def test_inner_folds_refuses_cv_when_a_class_has_one_member():
    """At N=25 and a 20% positive rate this is a real, frequent situation."""
    y = np.array([0] * 24 + [1])
    assert _inner_folds(25, y) == 0


def test_inner_folds_shrinks_to_the_minority_class_size():
    y = np.array([0] * 40 + [1] * 2)
    assert _inner_folds(42, y) == 2
    y = np.array([0] * 100 + [1] * 50)
    assert _inner_folds(150, y) == 3


def test_inner_folds_without_labels_depends_only_on_n():
    assert _inner_folds(10) == 2
    assert _inner_folds(29) == 2
    assert _inner_folds(30) == 3


def test_linear_head_falls_back_to_fixed_regularisation_when_cv_is_impossible():
    """No honest inner CV is available, so the head must not attempt one."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    y = np.array([0] * 24 + [1])
    pipe = build_head("linear", "classification", 25, seed=0, y_train=y)
    assert isinstance(pipe[-1], LogisticRegression)
    assert not isinstance(pipe[-1], LogisticRegressionCV)

    y_ok = np.array([0] * 20 + [1] * 5)
    pipe_ok = build_head("linear", "classification", 25, seed=0, y_train=y_ok)
    assert isinstance(pipe_ok[-1], LogisticRegressionCV)


def test_fit_predict_survives_a_barely_viable_subsample():
    """Two positives out of 25, every head, no exception and a usable score."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((25, 6))
    y_tr = np.array([0] * 23 + [1] * 2)
    X_te = rng.standard_normal((40, 6))
    y_te = rng.integers(0, 2, 40)

    for head in HEADS:
        pred = fit_predict(head, "classification", X_tr, y_tr, X_te, seed=0)
        assert pred.shape == (40,)
        assert np.isfinite(pred).all()
        assert np.isfinite(score("classification", y_te, pred)["auc"])


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fits_are_reproducible_for_a_seed(head, task):
    """20 seeds only means something if a seed pins the fit."""
    X, y = toy_classification() if task == "classification" else toy_regression()
    a = fit_predict(head, task, X[:90], y[:90], X[90:], seed=11)
    b = fit_predict(head, task, X[:90], y[:90], X[90:], seed=11)
    assert a == pytest.approx(b)


def test_unknown_head_is_rejected():
    with pytest.raises(KeyError):
        build_head("transformer", "classification", 100, 0)


# --------------------------------------------------------------- metrics -----
def test_classification_score_is_nan_when_the_test_set_is_single_class():
    y_true = np.zeros(20, dtype=int)
    got = score("classification", y_true, np.linspace(0, 1, 20))
    assert np.isnan(got["auc"]) and np.isnan(got["ap"])


def test_perfect_and_inverted_rankings_score_as_expected():
    y_true = np.array([0] * 10 + [1] * 10)
    perfect = score("classification", y_true, np.arange(20, dtype=float))
    inverted = score("classification", y_true, np.arange(20, 0, -1, dtype=float))
    assert perfect["auc"] == pytest.approx(1.0)
    assert inverted["auc"] == pytest.approx(0.0)


def test_constant_predictions_score_zero_correlation_not_nan():
    """A degenerate regression fit must yield 0.0, not NaN.

    NaNs here would silently drop cells from the aggregated curves and quietly
    bias the means upward.
    """
    y_true = np.linspace(0, 1, 30)
    got = score("regression", y_true, np.full(30, 0.5))
    assert got["spearman"] == 0.0
    assert got["pearson"] == 0.0
    assert np.isfinite(got["r2"]) and np.isfinite(got["rmse"])


def test_non_finite_regression_predictions_are_repaired():
    """A diverging MLP can emit NaN/inf; the sweep must still record a number."""
    y_true = np.linspace(0, 1, 20)
    pred = np.full(20, np.nan)
    pred[0], pred[1] = np.inf, -np.inf
    got = score("regression", y_true, pred)
    assert all(np.isfinite(v) for v in got.values())


def test_regression_metrics_are_sane_on_a_good_fit():
    y_true = np.linspace(0, 1, 50)
    got = score("regression", y_true, y_true + 0.01)
    assert got["spearman"] == pytest.approx(1.0)
    assert got["rmse"] == pytest.approx(0.01, abs=1e-9)
    assert got["r2"] > 0.99


def test_primary_metric_names_exist_in_the_score_dicts():
    """Aggregation and plotting index scores by PRIMARY_METRIC."""
    y_bin = np.array([0] * 10 + [1] * 10)
    assert PRIMARY_METRIC["classification"] in score(
        "classification", y_bin, np.arange(20, dtype=float)
    )
    assert PRIMARY_METRIC["regression"] in score(
        "regression", np.linspace(0, 1, 20), np.linspace(0, 1, 20)
    )
