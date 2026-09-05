"""Prediction heads and metrics.

Three heads, all cheap enough to refit thousands of times:

``linear``  L2 logistic regression (classification) or ridge (regression),
            with the regularisation strength chosen by inner cross-validation
            on the training subsample only.
``mlp``     one hidden layer of 64 units.  At N=25 with 2,560 ESM features
            this is wildly over-parameterised; that is the point.
``gbm``     histogram gradient boosting, with leaf sizes shrunk so it can
            still split at N=25.

Every head is wrapped in a StandardScaler fitted inside the pipeline, so no
information from the test set reaches the fit.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 5, 15)
CS = np.logspace(-4, 2, 7)


def _inner_folds(n: int, y=None) -> int:
    """Largest inner-CV fold count that still leaves both classes in every fold.

    Returns 0 when no honest inner CV is possible -- at N=25 with a 20%
    positive rate a subsample can land with a single positive antibody, and
    every stratified fold then has a single-class training half.  Callers fall
    back to a fixed regularisation strength in that case, which is what a real
    practitioner with 25 measurements would have to do anyway.
    """
    k = 3 if n >= 30 else 2
    if y is not None:
        minority = int(np.bincount(np.asarray(y, dtype=int)).min())
        if minority < 2:
            return 0
        k = max(2, min(k, minority))
    return k


def build_head(head: str, task: str, n_train: int, seed: int, y_train=None):
    if head == "linear":
        if task == "classification":
            folds = _inner_folds(n_train, y_train)
            if folds == 0:
                est = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
            else:
                est = LogisticRegressionCV(
                    Cs=CS,
                    cv=folds,
                    scoring="roc_auc",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                    n_jobs=1,
                )
        else:
            est = RidgeCV(alphas=ALPHAS)
    elif head == "mlp":
        kw = dict(
            hidden_layer_sizes=(64,),
            alpha=1e-2,
            max_iter=800,
            learning_rate_init=1e-3,
            early_stopping=False,
            random_state=seed,
        )
        est = MLPClassifier(**kw) if task == "classification" else MLPRegressor(**kw)
    elif head == "gbm":
        kw = dict(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=15,
            min_samples_leaf=max(2, min(10, n_train // 5)),
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        est = (
            HistGradientBoostingClassifier(**kw)
            if task == "classification"
            else HistGradientBoostingRegressor(**kw)
        )
    else:
        raise KeyError(head)
    return make_pipeline(StandardScaler(), est)


def fit_predict(head, task, X_tr, y_tr, X_te, seed):
    model = build_head(head, task, len(y_tr), seed, y_train=y_tr if task == "classification" else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(X_tr, y_tr)
        if task == "classification":
            est = model[-1]
            if hasattr(est, "predict_proba"):
                return model.predict_proba(X_te)[:, 1]
            return model.decision_function(X_te)
        return model.predict(X_te)


def score(task: str, y_true, y_pred) -> dict[str, float]:
    if task == "classification":
        if len(np.unique(y_true)) < 2:
            return {"auc": np.nan, "ap": np.nan}
        return {
            "auc": float(roc_auc_score(y_true, y_pred)),
            "ap": float(average_precision_score(y_true, y_pred)),
        }
    y_pred = np.nan_to_num(y_pred, nan=float(np.mean(y_true)), posinf=0.0, neginf=0.0)
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        rho = pear = 0.0
    else:
        rho = float(spearmanr(y_true, y_pred).statistic)
        pear = float(pearsonr(y_true, y_pred).statistic)
    return {
        "spearman": rho,
        "pearson": pear,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2))),
    }


PRIMARY_METRIC = {"classification": "auc", "regression": "spearman"}
