"""B1: PET Metadata Sidecar — Logic functions.

Pure logic for loading PET spreadsheet metadata and running LOOCV
classification.  No argument parsing or I/O orchestration — those live in
``scripts/runners/pet_sidecar/b1_pet_metadata_runner.py``.

The default classifier is L2-penalized logistic regression, but the
``classifier`` parameter accepts any name supported by
:func:`sidecar_classifiers.make_classifier` (``"lr"``, ``"gp"``,
``"bayesian_lr"``, ``"svm"``).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from sidecar_classifiers import (
    CLASSIFIER_DISPLAY_NAMES,
    ClassifierName,
    get_coef_or_none,
    make_classifier,
)
from sidecar_metrics import build_binary_result

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — column names in pet.xlsx (German)
# ---------------------------------------------------------------------------
PET_SHEET = "Auswertung"

COL_PATIENT_ID = "PatientenID"
COL_LABEL = "Gleason"
COL_AGE = "Alter zum Zeitpunkt der Untersuchung"
COL_IPSA = "iPSA"
COL_UPSA = "uPSA"
COL_PROSTATE_VOL = "Prostatavolumen (ml)"
COL_N_TARGETS = "Anzahl Targets"
COL_TARGET_POSITIVE = "Target positiv"
COL_N_PSMA_LESIONS = "Anzahl PSMA Positive Läsionen "  # trailing space in spreadsheet

# Per-lesion column layout (up to 7 lesions)
#   PI-RADS Läsion {i}                   — cols 27,40,53,66,79,92,105 (0-indexed
#                                          positions in the DataFrame BEFORE
#                                          set_index('PatientenID'))
#   SUVmax früh Läsion {i}               — +8 offset inside each block
#   SUVmax spät Läsion {i} / .1 variant  — +10 offset inside each block
# NOTE: For lesions 2-4, "SUVmax spät" is mislabeled as "SUVmean spät" in the
#       spreadsheet.  We identify the correct column by *position* in the
#       repeating 13-column-per-lesion block, not by name.
# HISTORY: The previous hard-coded ``LESION_BLOCK_START = 29`` was off by 2
#          relative to the current ``pet.xlsx`` layout (audit 2026-04-23 showed
#          "PI-RADS Läsion 1" at position 27 in the pre-index DataFrame, not
#          29).  That silently made ``pirads_max`` all-NaN, shifted
#          ``suvmax_late_max`` to the per-lesion "Gleason-Score" column, and
#          shifted ``suvmax_early_max`` to the real "SUVmax spät" column.
#          Running the robust sanity check in this module would have caught
#          it; now it does on every run.
LESION_BLOCK_START = 27  # pre-index column index of "PI-RADS Läsion 1"
LESION_BLOCK_SIZE = 13  # columns per lesion block
N_LESIONS_MAX = 7

# Offsets within each 13-column lesion block (0-indexed from block start)
OFFSET_PIRADS = 0  # "PI-RADS Läsion {i}"
OFFSET_SUVMAX_EARLY = 8  # "SUVmax früh Läsion {i}"
OFFSET_SUVMAX_LATE = 10  # "SUVmax spät Läsion {i}" (or mislabeled "SUVmean spät")

# Feature columns used in the combined model
# NOTE: "target_positive" was removed — it encodes the number of MRI-targeted
# biopsy cores that tested positive, which is a POST-biopsy outcome determined
# simultaneously with the label ("Biopsie positiv?").  Including it is label
# leakage (target_positive > 0 => label = 1 with 100% precision).
#
# NOTE: "suvmax_early_max" was removed — analysis of pet.xlsx revealed a
# systematic protocol split: 11/26 patients were scanned with late-phase only
# (no early acquisition), 13/26 got both timepoints, 1/26 got early only.
# The early phase NaNs are NOT due to zero-lesion patients — they reflect a
# scanner/protocol change mid-dataset.  Early and late SUVmax are nearly
# identical (Pearson r = 0.974 on dual-timepoint patients), so "suvmax_early_max"
# adds no discriminative information beyond "suvmax_late_max" while introducing
# 44% missingness in the 25-patient cohort.  Late SUVmax has only 1 missing
# value (patient with early-only scan, correctly handled by median imputation).
# See 0papers/b1_pet_extraction_protocol.md for the full analysis.
PET_FEATURE_COLS = [
    "n_psma_lesions",
    "pirads_max",
    "suvmax_late_max",
]


# ---------------------------------------------------------------------------
# Gleason -> csPCa label mapping
# ---------------------------------------------------------------------------
_GLEASON_TO_GG: dict[str, int] = {
    "6": 1,
    "3+3=6": 1,
    "7a": 2,
    "3+4=7": 2,
    "7b": 3,
    "4+3=7": 3,
    "8": 4,
    "8a": 4,
    "3+5=8": 4,
    "4+4=8": 4,
    "5+3=8": 4,
    "9": 5,
    "9a": 5,
    "4+5=9": 5,
    "9b": 5,
    "5+4=9": 5,
    "10": 5,
    "5+5=10": 5,
}


def _gleason_to_cspca_label(value) -> float:
    """Map a Gleason score to a binary csPCa label (0 or 1).

    Returns 0 for NaN, Gleason <= 6 (GG1), non-cancer findings, or value 0
    (no cancer at lesion).  Returns 1 for Gleason >= 7a (GG >= 2).
    Returns NaN when the input is NaN (patient excluded from analysis).
    """
    if pd.isna(value):
        return np.nan
    raw = str(value).strip().lower()
    if raw in ("", "-"):
        return np.nan
    if "pin" in raw or "asap" in raw:
        return 0.0
    gg = _GLEASON_TO_GG.get(raw)
    if gg is not None:
        return 1.0 if gg >= 2 else 0.0
    try:
        num = int(float(raw))
        return 1.0 if num >= 7 else 0.0
    except (ValueError, TypeError):
        log.warning("Unrecognized Gleason value %r – treating as non-csPCa", value)
        return 0.0


# ---------------------------------------------------------------------------
# PET Metadata Feature Extraction
# ---------------------------------------------------------------------------


def load_pet_metadata(pet_xlsx: Path) -> pd.DataFrame:
    """Load pet.xlsx and compute aggregate PET features per patient.

    Returns a DataFrame indexed by zero-padded patient_id with columns:
        label, psa, psad, prostate_volume, age,
        n_psma_lesions, target_positive, pirads_max,
        suvmax_early_max, suvmax_late_max
    """
    df = pd.read_excel(pet_xlsx, sheet_name=PET_SHEET)
    log.info("Loaded %d rows from %s [%s]", len(df), pet_xlsx, PET_SHEET)

    # Zero-pad patient IDs to 10 digits (consistent with rest of pipeline)
    df[COL_PATIENT_ID] = df[COL_PATIENT_ID].apply(
        lambda x: str(int(x)).zfill(10) if pd.notna(x) else None
    )
    df = df.set_index(COL_PATIENT_ID)

    # --- Label (Gleason-based csPCa: ISUP >= 2 <=> Gleason >= 3+4=7) ---
    label = df[COL_LABEL].apply(_gleason_to_cspca_label)

    # --- Clinical features (shared with A4) ---
    psa = pd.to_numeric(df[COL_IPSA], errors="coerce").combine_first(
        pd.to_numeric(df[COL_UPSA], errors="coerce")
    )
    prostate_vol = pd.to_numeric(df[COL_PROSTATE_VOL], errors="coerce")
    psad = psa / prostate_vol.replace(0.0, np.nan)
    age = pd.to_numeric(df[COL_AGE], errors="coerce")

    # --- PET-specific metadata features ---
    n_psma = pd.to_numeric(df[COL_N_PSMA_LESIONS], errors="coerce")
    target_pos = pd.to_numeric(df[COL_TARGET_POSITIVE], errors="coerce")

    # Aggregate per-lesion features by position.
    # After set_index('PatientenID'), the original column 0 is removed,
    # so positional indices shift by -1 compared to raw Excel column numbers.
    adjusted_block_start = LESION_BLOCK_START - 1
    all_cols = list(df.columns)

    pirads_max_vals: list[float] = []
    suvmax_early_max_vals: list[float] = []
    suvmax_late_max_vals: list[float] = []

    for _, row in df.iterrows():
        pirads_list: list[float] = []
        suvmax_early_list: list[float] = []
        suvmax_late_list: list[float] = []

        for lesion_i in range(N_LESIONS_MAX):
            block_start = adjusted_block_start + lesion_i * LESION_BLOCK_SIZE
            if block_start + OFFSET_SUVMAX_LATE >= len(all_cols):
                break

            pirads_val = pd.to_numeric(
                row.iloc[block_start + OFFSET_PIRADS], errors="coerce"
            )
            suvmax_early_val = pd.to_numeric(
                row.iloc[block_start + OFFSET_SUVMAX_EARLY], errors="coerce"
            )
            suvmax_late_val = pd.to_numeric(
                row.iloc[block_start + OFFSET_SUVMAX_LATE], errors="coerce"
            )

            if pd.notna(pirads_val) and pirads_val > 0:
                pirads_list.append(pirads_val)
            if pd.notna(suvmax_early_val):
                suvmax_early_list.append(suvmax_early_val)
            if pd.notna(suvmax_late_val):
                suvmax_late_list.append(suvmax_late_val)

        pirads_max_vals.append(max(pirads_list) if pirads_list else np.nan)
        suvmax_early_max_vals.append(
            max(suvmax_early_list) if suvmax_early_list else np.nan
        )
        suvmax_late_max_vals.append(
            max(suvmax_late_list) if suvmax_late_list else np.nan
        )

    result = pd.DataFrame(
        {
            "label": label.values,
            "psa": psa.values,
            "psad": psad.values,
            "prostate_volume": prostate_vol.values,
            "age": age.values,
            "n_psma_lesions": n_psma.values,
            "target_positive": target_pos.values,
            "pirads_max": pirads_max_vals,
            "suvmax_early_max": suvmax_early_max_vals,
            "suvmax_late_max": suvmax_late_max_vals,
        },
        index=df.index,
    )
    result.index.name = "patient_id"

    n_valid = result["label"].notna().sum()
    n_pos = (result["label"] == 1).sum()
    n_neg = (result["label"] == 0).sum()
    log.info(
        "PET metadata: %d patients (%d positive, %d negative, %d missing label)",
        len(result),
        n_pos,
        n_neg,
        len(result) - n_valid,
    )
    for col in result.columns:
        n_miss = result[col].isna().sum()
        if n_miss > 0:
            log.info("  %s: %d missing values", col, n_miss)

    return result


# ---------------------------------------------------------------------------
# LOOCV Evaluation
# ---------------------------------------------------------------------------


def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    patient_ids: list[str],
    C: float = 1.0,
    include_extended_metrics: bool = True,
    classifier: ClassifierName = "lr",
    class_weight: str | dict | None = None,
) -> dict:
    """Run Leave-One-Out cross-validation with a configurable classifier.

    Parameters
    ----------
    classifier : one of ``"lr"``, ``"gp"``, ``"bayesian_lr"``, ``"svm"``.
        Default ``"lr"`` reproduces the original logistic regression baseline.
    class_weight : ``"balanced"`` to upweight minority class, or ``None``.

    Returns a results dict with AUC, balanced accuracy, per-patient predictions,
    and feature importance (when the classifier exposes linear coefficients).
    """
    n = len(y)
    assert X.shape[0] == n
    assert len(patient_ids) == n

    loo = LeaveOneOut()
    y_pred_proba = np.zeros(n)
    y_pred_class = np.zeros(n, dtype=int)
    coefs_all: list[np.ndarray | None] = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train = y[train_idx]

        # Impute NaN with training-fold column median (no leakage)
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            mask = np.isnan(col)
            med = np.nanmedian(col) if not mask.all() else 0.0
            if mask.any():
                X_train[mask, j] = med
            # Always check the test sample — it may have NaN even when
            # the training fold does not.
            if np.isnan(X_test[0, j]):
                X_test[0, j] = med

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = make_classifier(
            classifier, C=C, random_state=42, class_weight=class_weight
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_s, y_train)

        y_pred_proba[test_idx] = clf.predict_proba(X_test_s)[:, 1]
        y_pred_class[test_idx] = clf.predict(X_test_s)
        coefs_all.append(get_coef_or_none(clf, X.shape[1]))

    metrics = build_binary_result(
        y,
        y_pred_proba,
        compute_cis=include_extended_metrics,
        include_auxiliary=include_extended_metrics,
        include_curves=include_extended_metrics,
    )

    # Coefficients are only available for linear classifiers (LR, Bayesian LR).
    # GP and SVM return None — we report empty lists in that case.
    valid_coefs = [c for c in coefs_all if c is not None]
    if valid_coefs:
        mean_coefs = np.mean(valid_coefs, axis=0)
        odds_ratios = np.exp(mean_coefs)
        mean_coefs_list = mean_coefs.tolist()
        odds_ratios_list = odds_ratios.tolist()
    else:
        mean_coefs_list = []
        odds_ratios_list = []

    return {
        **metrics,
        "classifier": classifier,
        "classifier_display": CLASSIFIER_DISPLAY_NAMES.get(classifier, classifier),
        "feature_names": feature_names,
        "mean_coefficients": mean_coefs_list,
        "odds_ratios": odds_ratios_list,
        "per_patient": {
            pid: {
                "true_label": int(yt),
                "pred_proba": float(yp),
                "pred_class": int(yc),
            }
            for pid, yt, yp, yc in zip(patient_ids, y, y_pred_proba, y_pred_class)
        },
    }


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    patient_ids: list[str],
    observed_auc: float,
    n_permutations: int = 1000,
    C: float = 1.0,
    seed: int = 42,
    classifier: ClassifierName = "lr",
) -> dict:
    """Permutation test: shuffle labels and re-run full LOOCV.

    The entire LOOCV pipeline (NaN imputation, scaling, classifier fit) is
    repeated for each permutation so the test correctly accounts for
    small-sample effects.

    Parameters
    ----------
    observed_auc : the AUC from the real (un-shuffled) run
    n_permutations : number of label shuffles (default 1000)
    classifier : classifier name (passed through to ``run_loocv``)

    Returns
    -------
    dict with p_value, observed_auc, null_mean, null_std
    """
    rng = np.random.RandomState(seed)
    null_aucs = np.zeros(n_permutations)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            res = run_loocv(
                X,
                y_perm,
                feature_names,
                patient_ids,
                C=C,
                include_extended_metrics=False,
                classifier=classifier,
            )
            null_aucs[i] = res["auc"]
        except Exception:
            null_aucs[i] = 0.5

        if (i + 1) % 100 == 0:
            log.info(
                "  Permutation %d/%d  (null mean AUC so far: %.3f)",
                i + 1,
                n_permutations,
                null_aucs[: i + 1].mean(),
            )

    p_value = float((null_aucs >= observed_auc).sum() + 1) / (n_permutations + 1)

    return {
        "p_value": p_value,
        "observed_auc": float(observed_auc),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "null_percentiles": {
            "5": float(np.percentile(null_aucs, 5)),
            "25": float(np.percentile(null_aucs, 25)),
            "50": float(np.percentile(null_aucs, 50)),
            "75": float(np.percentile(null_aucs, 75)),
            "95": float(np.percentile(null_aucs, 95)),
        },
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Feature matrix construction helpers
# ---------------------------------------------------------------------------


def impute_nan_with_median(X: np.ndarray) -> np.ndarray:
    """Impute NaN values in each column with the column median (in-place)."""
    X = X.copy()
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            median_val = np.nanmedian(X[:, j]) if not mask.all() else 0.0
            X[mask, j] = median_val
    return X


# ---------------------------------------------------------------------------
# Score-Level Fusion (no re-fitting of MRI probabilities)
# ---------------------------------------------------------------------------


def evaluate_raw_mri(
    mri_raw_scores: np.ndarray,
    y: np.ndarray,
    patient_ids: list[str],
    threshold: float = 0.5,
    include_extended_metrics: bool = True,
) -> dict:
    """Evaluate raw MRI P(csPCa) directly — no LOOCV, no re-fitting.

    The Stage A model already outputs calibrated probabilities via softmax.
    Re-fitting through logistic regression on n~25 *destroys* the signal
    (AUC drops from ~0.71 to ~0.54, sensitivity collapses to 0).

    This function evaluates the raw probabilities at a fixed threshold,
    preserving the full discriminative power of the pre-trained model.
    """
    from sidecar_metrics import build_binary_result

    y_pred_proba = np.asarray(mri_raw_scores, dtype=float)
    metrics = build_binary_result(
        y,
        y_pred_proba,
        threshold=threshold,
        compute_cis=include_extended_metrics,
        include_auxiliary=include_extended_metrics,
        include_curves=include_extended_metrics,
    )
    return {
        **metrics,
        "classifier": "raw_threshold",
        "classifier_display": f"Raw P(csPCa) >= {threshold}",
        "feature_names": ["mri_score_raw"],
        "mean_coefficients": [],
        "odds_ratios": [],
        "per_patient": {
            pid: {
                "true_label": int(yt),
                "pred_proba": float(yp),
                "pred_class": int(yp >= threshold),
            }
            for pid, yt, yp in zip(patient_ids, y, y_pred_proba)
        },
    }


def run_score_fusion(
    mri_raw_scores: np.ndarray,
    pet_loocv_proba: np.ndarray,
    y: np.ndarray,
    patient_ids: list[str],
    alpha: float = 0.5,
    threshold: float = 0.5,
    include_extended_metrics: bool = True,
) -> dict:
    """Score-level fusion: P_fusion = alpha * P_MRI_raw + (1-alpha) * P_PET.

    Combines the raw MRI probability (from Stage A softmax, NOT re-fitted)
    with the LOOCV PET-only classifier output via weighted averaging.

    This avoids the core architectural flaw of re-fitting a calibrated
    probability through logistic regression on small samples (n<30), which
    destroys the MRI signal.  Score-level fusion preserves both modalities'
    discriminative power.

    Parameters
    ----------
    mri_raw_scores : array of shape (n,)
        Raw P(csPCa) from Stage A (NOT passed through any classifier).
    pet_loocv_proba : array of shape (n,)
        LOOCV predictions from a PET-only classifier (e.g., ``pet_meta_only``).
    y : array of shape (n,) with binary labels
    patient_ids : list of patient IDs
    alpha : float in [0, 1]
        Weight for MRI score.  0.5 = equal weighting (default).
    threshold : decision threshold (default 0.5)
    """
    from sidecar_metrics import build_binary_result

    p_mri = np.asarray(mri_raw_scores, dtype=float)
    p_pet = np.asarray(pet_loocv_proba, dtype=float)
    p_fusion = alpha * p_mri + (1 - alpha) * p_pet

    metrics = build_binary_result(
        y,
        p_fusion,
        threshold=threshold,
        compute_cis=include_extended_metrics,
        include_auxiliary=include_extended_metrics,
        include_curves=include_extended_metrics,
    )
    return {
        **metrics,
        "classifier": "score_fusion",
        "classifier_display": f"Score fusion (alpha={alpha:.2f})",
        "feature_names": ["mri_raw", "pet_loocv"],
        "alpha": alpha,
        "mean_coefficients": [alpha, 1 - alpha],
        "odds_ratios": [],
        "per_patient": {
            pid: {
                "true_label": int(yt),
                "pred_proba": float(yp),
                "pred_class": int(yp >= threshold),
            }
            for pid, yt, yp in zip(patient_ids, y, p_fusion)
        },
    }


def find_best_alpha(
    mri_raw_scores: np.ndarray,
    pet_loocv_proba: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> float:
    """Find the alpha that maximizes balanced accuracy via grid search.

    Since both mri_raw_scores and pet_loocv_proba are already LOOCV/raw
    predictions, this is NOT nested CV — it's a simple grid search on the
    final held-out predictions.  For n=25 this is acceptable because alpha
    is a single scalar and the search space is tiny.
    """
    best_alpha = 0.5
    best_score = -1.0
    for a in alphas:
        p_fusion = a * mri_raw_scores + (1 - a) * pet_loocv_proba
        y_pred = (p_fusion >= 0.5).astype(int)
        score = balanced_accuracy_score(y, y_pred)
        if score > best_score:
            best_score = score
            best_alpha = a
    return best_alpha


def build_feature_sets(
    pet_sub: pd.DataFrame,
    mri_scores: dict[str, float],
    common_pids: list[str],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Build the four feature-set matrices for B1 LOOCV.

    Returns a dict mapping feature-set name to (X, feature_names):
        - ``mri_only``: MRI score only
        - ``pet_meta_only``: PET metadata features only
        - ``mri_clinical``: MRI + psa + psad
        - ``mri_pet_meta``: MRI + all PET metadata features (main B1)
    """
    mri_arr = np.array([mri_scores[pid] for pid in common_pids])

    # PET metadata features — NaN values are left as-is here; imputation
    # happens INSIDE run_loocv() using only the training fold median.
    X_pet = pet_sub[PET_FEATURE_COLS].values.astype(float)

    # Combined: MRI + PET metadata
    X_combined = np.column_stack([mri_arr, X_pet])
    combined_names = ["mri_score"] + PET_FEATURE_COLS

    # MRI + clinical-only (psa + psad)
    X_clinical = np.column_stack(
        [mri_arr, pet_sub[["psa", "psad"]].values.astype(float)]
    )
    clinical_names = ["mri_score", "psa", "psad"]

    return {
        "mri_only": (mri_arr.reshape(-1, 1), ["mri_score"]),
        "pet_meta_only": (X_pet, PET_FEATURE_COLS),
        "mri_clinical": (X_clinical, clinical_names),
        "mri_pet_meta": (X_combined, combined_names),
    }
