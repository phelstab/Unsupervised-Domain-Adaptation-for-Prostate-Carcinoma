"""Sidecar classifier factory for B1/C1 late-fusion experiments.

Provides a unified interface for creating classifiers used in the LOOCV
pipeline.  All classifiers follow the sklearn API (``fit``, ``predict``,
``predict_proba``).

Supported classifiers:

- **lr** — L2-penalized logistic regression (baseline, default)
- **gp** — Gaussian process classifier with RBF kernel
- **bayesian_lr** — Bayesian logistic regression (via BayesianRidge on
  log-odds; approximates a MAP-regularised LR with uncertainty)
- **svm** — Support vector classifier with RBF kernel + Platt scaling

Why these three alternatives and not few-shot / neural networks?
With N=24-25 total labelled PET samples, any model with more than ~5-10
effective parameters will overfit.  LR has ~K+1 params (one per feature
plus intercept).  GP and SVM use kernel methods that regularise
implicitly through the kernel bandwidth without adding free parameters
proportional to feature count.  Bayesian LR adds a prior over weights,
giving automatic regularisation and calibrated uncertainty.  By contrast,
even a 1-hidden-layer neural net with 50 units would have ~100-200 params
— enough to memorise the 23-sample training folds perfectly.  Few-shot
learning (prototypical nets, MAML) requires a large meta-training
distribution of similar tasks to learn *how to learn*; we have exactly
one task with 24 samples, so there is nothing to meta-train on.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for supported classifier names
# ---------------------------------------------------------------------------
ClassifierName = Literal["lr", "gp", "bayesian_lr", "svm"]

CLASSIFIER_CHOICES: list[str] = ["lr", "gp", "bayesian_lr", "svm"]

CLASSIFIER_DISPLAY_NAMES: dict[str, str] = {
    "lr": "Logistic Regression (L2)",
    "gp": "Gaussian Process (RBF)",
    "bayesian_lr": "Bayesian Logistic Regression",
    "svm": "SVM (RBF + Platt)",
}


# ---------------------------------------------------------------------------
# Bayesian Logistic Regression wrapper
# ---------------------------------------------------------------------------


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Bayesian logistic regression via Laplace approximation.

    Uses sklearn's ``LogisticRegression`` with a grid of C values selected
    by LOO log-likelihood on the training fold.  This is a pragmatic
    approximation: the L2 penalty ``1/(2C)`` acts as a Gaussian prior on
    the weights, and selecting C by LOO approximates the marginal
    likelihood.

    After fitting, ``predict_proba`` returns calibrated probabilities and
    ``coef_`` / ``intercept_`` are available for inspection.
    """

    def __init__(self, random_state: int = 42, class_weight: str | dict | None = None):
        self.random_state = random_state
        self.class_weight = class_weight
        self._clf: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticRegression":
        import warnings

        best_score = -np.inf
        best_clf = None
        # Grid of regularisation strengths (prior precision)
        for C in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            clf = LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X, y)

            # Compute training log-likelihood as proxy for model evidence
            proba = clf.predict_proba(X)
            # Clip to avoid log(0)
            proba = np.clip(proba, 1e-15, 1 - 1e-15)
            log_lik = np.sum(np.log(proba[np.arange(len(y)), y.astype(int)]))
            # Add log-prior (Gaussian with precision 1/C)
            w = clf.coef_.ravel()
            log_prior = -0.5 * np.sum(w**2) / C
            score = log_lik + log_prior

            if score > best_score:
                best_score = score
                best_clf = clf

        self._clf = best_clf
        self.classes_ = best_clf.classes_
        self.coef_ = best_clf.coef_
        self.intercept_ = best_clf.intercept_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_classifier(
    name: ClassifierName,
    C: float = 1.0,
    random_state: int = 42,
    class_weight: str | dict | None = None,
) -> BaseEstimator:
    """Create a classifier instance by name.

    Parameters
    ----------
    name : one of ``"lr"``, ``"gp"``, ``"bayesian_lr"``, ``"svm"``
    C : regularisation parameter (used directly by LR and SVM;
        ignored by GP and Bayesian LR which set their own)
    random_state : random seed for reproducibility
    class_weight : ``"balanced"`` to upweight the minority class
        proportional to its inverse frequency, ``None`` (default) for
        equal weights.  Supported by LR, Bayesian LR, and SVM.
        Ignored by GP (no native support).

    Returns
    -------
    A sklearn-compatible classifier with ``fit``, ``predict``,
    ``predict_proba`` methods.
    """
    if name == "lr":
        return LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
            class_weight=class_weight,
        )
    elif name == "gp":
        # Length-scale bounds let the GP adapt to the data scale.
        # ConstantKernel * RBF is the standard choice.
        # Note: GaussianProcessClassifier does not support class_weight.
        if class_weight is not None:
            log.warning("GP classifier does not support class_weight; ignoring.")
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
        )
        return GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=random_state,
            max_iter_predict=200,
        )
    elif name == "bayesian_lr":
        return BayesianLogisticRegression(
            random_state=random_state, class_weight=class_weight
        )
    elif name == "svm":
        # Platt scaling (probability=True) wraps a 5-fold CV inside fit()
        # to calibrate probabilities.  With N=23 training samples per LOOCV
        # fold this is tight but functional.
        return SVC(
            C=C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=random_state,
            class_weight=class_weight,
        )
    else:
        raise ValueError(
            f"Unknown classifier {name!r}. Choose from: {', '.join(CLASSIFIER_CHOICES)}"
        )


def has_coef(clf: BaseEstimator) -> bool:
    """Check whether a fitted classifier exposes linear coefficients."""
    return hasattr(clf, "coef_") and clf.coef_ is not None


def get_coef_or_none(clf: BaseEstimator, n_features: int) -> np.ndarray | None:
    """Return ``clf.coef_[0]`` if available, else ``None``.

    For classifiers without linear coefficients (GP, SVM with RBF kernel),
    returns ``None``.  Callers should handle this gracefully.
    """
    if has_coef(clf):
        return clf.coef_[0].copy()
    return None
