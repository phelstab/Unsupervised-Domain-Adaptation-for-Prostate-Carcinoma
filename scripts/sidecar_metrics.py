"""Shared evaluation utilities for B1/C1 PET sidecar experiments.

Turns out-of-fold patient-level probabilities into a consistent set of:
  - discrimination metrics (AUROC, AUPRC)
  - threshold metrics at a fixed decision threshold
  - proper scoring rules (Brier, log loss)
  - calibration summaries
  - stratified bootstrap confidence intervals
  - paired model-comparison deltas for MRI-only vs fused models

The sidecar runs use LOOCV, so all metrics here are computed on the final
out-of-fold predictions only. Confidence intervals are descriptive internal
uncertainty estimates from stratified patient-level bootstrap resampling.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_N_BOOTSTRAPS = 1000
DEFAULT_RANDOM_SEED = 42
CALIBRATION_N_BINS = 5
THRESHOLD_ANALYSIS_POINTS = (0.3, 0.5, 0.7)
DECISION_CURVE_THRESHOLDS = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2).tolist())

PRIMARY_METRIC_SPECS = [
    ("auc", "AUC"),
    ("auprc", "AUPRC"),
    ("balanced_accuracy", "BalAcc"),
    ("brier_score", "Brier"),
]
THRESHOLD_METRIC_SPECS = [
    ("sensitivity", "Sens"),
    ("specificity", "Spec"),
    ("ppv", "PPV"),
    ("npv", "NPV"),
]
RESULT_METRIC_KEYS = [
    "auc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "brier_score",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
]
DELTA_METRIC_KEYS = [
    "auc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "brier_score",
]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite ratio; 0.0 if the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def _clip_probabilities(y_proba: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for log-based calculations."""
    return np.clip(np.asarray(y_proba, dtype=float), eps, 1.0 - eps)


def _predict_from_threshold(
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Convert probabilities to binary predictions at a fixed threshold."""
    return (np.asarray(y_proba, dtype=float) >= threshold).astype(int)


def _build_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
) -> list[dict[str, float]]:
    """Build a small quantile-style calibration table.

    With n~25, a simple equal-count binning is more stable than uniform bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    order = np.argsort(y_proba)
    sorted_true = y_true[order]
    sorted_proba = y_proba[order]

    curve: list[dict[str, float]] = []
    for bin_idx, indices in enumerate(
        np.array_split(np.arange(len(y_true)), n_bins), start=1
    ):
        if len(indices) == 0:
            continue
        curve.append(
            {
                "bin": int(bin_idx),
                "count": int(len(indices)),
                "mean_pred": float(sorted_proba[indices].mean()),
                "frac_positive": float(sorted_true[indices].mean()),
                "pred_min": float(sorted_proba[indices].min()),
                "pred_max": float(sorted_proba[indices].max()),
            }
        )
    return curve


def _fit_calibration_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope.

    This fits a logistic recalibration model:
        y ~ intercept + slope * logit(p)

    The estimates are descriptive only for these very small cohorts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = _clip_probabilities(y_proba)
    logits = np.log(y_proba / (1.0 - y_proba)).reshape(-1, 1)

    try:
        clf = LogisticRegression(
            C=1e6,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=DEFAULT_RANDOM_SEED,
        )
        clf.fit(logits, y_true)
        intercept = float(clf.intercept_[0])
        slope = float(clf.coef_[0, 0])
        return intercept, slope
    except Exception:
        return float("nan"), float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    include_curves: bool = True,
) -> dict:
    """Compute binary classification metrics from patient-level probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = _predict_from_threshold(y_proba, threshold=threshold)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "ppv": _safe_divide(tp, tp + fp),
        "npv": _safe_divide(tn, tn + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack(
                    [1.0 - _clip_probabilities(y_proba), _clip_probabilities(y_proba)]
                ),
                labels=[0, 1],
            )
        ),
        "prevalence": float(y_true.mean()),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
    }

    intercept, slope = _fit_calibration_model(y_true, y_proba)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope
    metrics["calibration_curve"] = _build_calibration_curve(y_true, y_proba)

    if include_curves:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        metrics["roc_curve"] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in roc_thresholds],
        }
        metrics["pr_curve"] = {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "thresholds": [float(x) for x in pr_thresholds],
        }

    return metrics


def _stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Draw a stratified bootstrap sample preserving class counts."""
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    sample_idx = np.concatenate([boot_pos, boot_neg])
    rng.shuffle(sample_idx)
    return sample_idx


def compute_metric_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    metrics: Iterable[str] = RESULT_METRIC_KEYS,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """Compute percentile bootstrap CIs for a fixed metric set."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    metric_names = list(metrics)
    rng = np.random.RandomState(seed)
    draws = {name: np.zeros(n_bootstraps, dtype=float) for name in metric_names}

    for i in range(n_bootstraps):
        idx = _stratified_bootstrap_indices(y_true, rng)
        boot_metrics = compute_binary_metrics(
            y_true[idx],
            y_proba[idx],
            threshold=threshold,
            include_curves=False,
        )
        for name in metric_names:
            draws[name][i] = boot_metrics[name]

    cis: dict[str, dict[str, float]] = {}
    for name in metric_names:
        cis[name] = {
            "low": float(np.percentile(draws[name], 2.5)),
            "high": float(np.percentile(draws[name], 97.5)),
        }
    return cis


def build_binary_result(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    compute_cis: bool = True,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_RANDOM_SEED,
    include_curves: bool = True,
    include_auxiliary: bool = True,
) -> dict:
    """Build the common result payload used by B1 and C1."""
    result = compute_binary_metrics(
        y_true,
        y_proba,
        threshold=threshold,
        include_curves=include_curves,
    )
    if compute_cis:
        result["confidence_intervals"] = compute_metric_confidence_intervals(
            y_true,
            y_proba,
            threshold=threshold,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
    if include_auxiliary:
        result["threshold_analysis"] = compute_threshold_analysis(y_true, y_proba)
        result["decision_curve"] = compute_decision_curve(y_true, y_proba)
    return result


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    thresholds: Iterable[float] = THRESHOLD_ANALYSIS_POINTS,
) -> list[dict[str, float]]:
    """Compute threshold metrics at a small fixed set of decision thresholds."""
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        metrics = compute_binary_metrics(
            y_true,
            y_proba,
            threshold=float(thr),
            include_curves=False,
        )
        cm = metrics["confusion_matrix"]
        rows.append(
            {
                "threshold": float(thr),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
            }
        )
    return rows


def compute_decision_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    thresholds: Iterable[float] = DECISION_CURVE_THRESHOLDS,
) -> list[dict[str, float]]:
    """Compute simple decision-curve net benefit values."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    prevalence = float(y_true.mean())

    rows: list[dict[str, float]] = []
    for thr in thresholds:
        thr = float(thr)
        if thr <= 0.0 or thr >= 1.0:
            continue
        pred = _predict_from_threshold(y_proba, threshold=thr)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        weight = thr / (1.0 - thr)
        net_benefit = (tp / n) - (fp / n) * weight
        treat_all = prevalence - (1.0 - prevalence) * weight
        rows.append(
            {
                "threshold": thr,
                "net_benefit_model": float(net_benefit),
                "net_benefit_treat_all": float(treat_all),
                "net_benefit_treat_none": 0.0,
                "tp": int(tp),
                "fp": int(fp),
            }
        )
    return rows


def compare_binary_models(
    y_true: np.ndarray,
    baseline_proba: np.ndarray,
    compare_proba: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    metrics: Iterable[str] = DELTA_METRIC_KEYS,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """Compare two models on the same patients using paired bootstrap deltas.

    The most clinically interpretable outputs for the PET sidecar study are the
    threshold-level transitions:
      - rescued false negatives
      - new false positives introduced
    """
    y_true = np.asarray(y_true, dtype=int)
    baseline_proba = np.asarray(baseline_proba, dtype=float)
    compare_proba = np.asarray(compare_proba, dtype=float)
    metric_names = list(metrics)

    base_metrics = compute_binary_metrics(
        y_true, baseline_proba, threshold=threshold, include_curves=False
    )
    comp_metrics = compute_binary_metrics(
        y_true, compare_proba, threshold=threshold, include_curves=False
    )

    observed_delta = {
        name: float(comp_metrics[name] - base_metrics[name]) for name in metric_names
    }

    base_pred = _predict_from_threshold(baseline_proba, threshold=threshold)
    comp_pred = _predict_from_threshold(compare_proba, threshold=threshold)
    base_cm = np.array(base_metrics["confusion_matrix"], dtype=int)
    comp_cm = np.array(comp_metrics["confusion_matrix"], dtype=int)

    transition_counts = {
        "rescued_false_negatives": int(
            ((y_true == 1) & (base_pred == 0) & (comp_pred == 1)).sum()
        ),
        "new_false_positives": int(
            ((y_true == 0) & (base_pred == 0) & (comp_pred == 1)).sum()
        ),
        "corrected_false_positives": int(
            ((y_true == 0) & (base_pred == 1) & (comp_pred == 0)).sum()
        ),
        "new_false_negatives": int(
            ((y_true == 1) & (base_pred == 1) & (comp_pred == 0)).sum()
        ),
        "changed_predictions": int((base_pred != comp_pred).sum()),
    }

    rng = np.random.RandomState(seed)
    delta_draws = {name: np.zeros(n_bootstraps, dtype=float) for name in metric_names}
    for i in range(n_bootstraps):
        idx = _stratified_bootstrap_indices(y_true, rng)
        boot_base = compute_binary_metrics(
            y_true[idx], baseline_proba[idx], threshold=threshold, include_curves=False
        )
        boot_comp = compute_binary_metrics(
            y_true[idx], compare_proba[idx], threshold=threshold, include_curves=False
        )
        for name in metric_names:
            delta_draws[name][i] = boot_comp[name] - boot_base[name]

    delta_cis = {
        name: {
            "low": float(np.percentile(delta_draws[name], 2.5)),
            "high": float(np.percentile(delta_draws[name], 97.5)),
        }
        for name in metric_names
    }

    return {
        "baseline": base_metrics,
        "compare": comp_metrics,
        "delta": observed_delta,
        "delta_confidence_intervals": delta_cis,
        "threshold": float(threshold),
        "confusion_matrix_delta": (comp_cm - base_cm).tolist(),
        "transition_counts": transition_counts,
    }


def flatten_result_for_csv(model_name: str, result: dict) -> dict[str, float | str]:
    """Flatten one model result into a single CSV row."""
    row: dict[str, float | str] = {
        "model": model_name,
        "n_samples": result.get("n_samples"),
        "n_positive": result.get("n_positive"),
        "n_negative": result.get("n_negative"),
        "threshold": result.get("threshold"),
    }
    for key in RESULT_METRIC_KEYS:
        row[key] = result.get(key)
        ci = result.get("confidence_intervals", {}).get(key)
        if ci:
            row[f"{key}_ci_low"] = ci["low"]
            row[f"{key}_ci_high"] = ci["high"]
    cm = result.get("confusion_matrix", [[0, 0], [0, 0]])
    row["tn"] = cm[0][0]
    row["fp"] = cm[0][1]
    row["fn"] = cm[1][0]
    row["tp"] = cm[1][1]
    return row


def flatten_comparison_for_csv(
    comparison_name: str, comparison: dict
) -> dict[str, float | str]:
    """Flatten one model-comparison payload into a single CSV row."""
    row: dict[str, float | str] = {"comparison": comparison_name}
    for key, value in comparison.get("delta", {}).items():
        row[f"delta_{key}"] = value
        ci = comparison.get("delta_confidence_intervals", {}).get(key)
        if ci:
            row[f"delta_{key}_ci_low"] = ci["low"]
            row[f"delta_{key}_ci_high"] = ci["high"]
    for key, value in comparison.get("transition_counts", {}).items():
        row[key] = value
    cm_delta = comparison.get("confusion_matrix_delta", [[0, 0], [0, 0]])
    row["delta_tn"] = cm_delta[0][0]
    row["delta_fp"] = cm_delta[0][1]
    row["delta_fn"] = cm_delta[1][0]
    row["delta_tp"] = cm_delta[1][1]
    return row


def flatten_threshold_analysis_for_csv(
    model_name: str, result: dict
) -> list[dict[str, float | str]]:
    """Flatten threshold-analysis rows for CSV export."""
    rows: list[dict[str, float | str]] = []
    for item in result.get("threshold_analysis", []):
        row = {"model": model_name}
        row.update(item)
        rows.append(row)
    return rows


def flatten_decision_curve_for_csv(
    model_name: str, result: dict
) -> list[dict[str, float | str]]:
    """Flatten decision-curve rows for CSV export."""
    rows: list[dict[str, float | str]] = []
    for item in result.get("decision_curve", []):
        row = {"model": model_name}
        row.update(item)
        rows.append(row)
    return rows


def format_metric_block(
    result: dict,
    metric_specs: Iterable[tuple[str, str]],
    *,
    include_ci: bool = False,
    precision: int = 3,
) -> str:
    """Format a compact metric string for console/summary output."""
    parts: list[str] = []
    for key, label in metric_specs:
        value = result.get(key)
        if value is None:
            continue
        if include_ci:
            ci = result.get("confidence_intervals", {}).get(key)
            if ci:
                parts.append(
                    f"{label}={value:.{precision}f} "
                    f"[{ci['low']:.{precision}f}, {ci['high']:.{precision}f}]"
                )
                continue
        parts.append(f"{label}={value:.{precision}f}")
    return "  ".join(parts)


def format_delta_block(
    comparison: dict,
    metric_keys: Iterable[str] = ("auc", "auprc", "balanced_accuracy", "brier_score"),
    *,
    include_ci: bool = False,
    precision: int = 3,
) -> str:
    """Format a compact delta string for paired model comparisons."""
    label_map = {
        "auc": "dAUC",
        "auprc": "dAUPRC",
        "balanced_accuracy": "dBalAcc",
        "brier_score": "dBrier",
        "sensitivity": "dSens",
        "specificity": "dSpec",
        "ppv": "dPPV",
        "npv": "dNPV",
    }
    parts: list[str] = []
    for key in metric_keys:
        value = comparison.get("delta", {}).get(key)
        if value is None:
            continue
        label = label_map.get(key, key)
        if include_ci:
            ci = comparison.get("delta_confidence_intervals", {}).get(key)
            if ci:
                parts.append(
                    f"{label}={value:+.{precision}f} "
                    f"[{ci['low']:+.{precision}f}, {ci['high']:+.{precision}f}]"
                )
                continue
        parts.append(f"{label}={value:+.{precision}f}")
    return "  ".join(parts)
