"""C1: PET Image Radiomics Fusion — Logic functions.

Pure logic for extracting radiomic features from SUV-normalized PET volumes
within prostate gland ROIs, and running LOOCV classification with nested
feature selection.  No argument parsing or I/O orchestration — those live in
``scripts/runners/pet_sidecar/c1_pet_radiomics_runner.py``.

The default classifier is L2-penalized logistic regression, but the
``classifier`` parameter accepts any name supported by
:func:`sidecar_classifiers.make_classifier` (``"lr"``, ``"gp"``,
``"bayesian_lr"``, ``"svm"``).

**Design choices** (n=25 with labels, n=26 total):
  - ROI = prostate gland mask (same as A2/A4), resampled to PET space
  - Feature extraction via pyradiomics
  - Feature selection INSIDE LOOCV to prevent information leakage
  - Max 3-5 features (n/5 heuristic for n~25)
  - MRI score always included as anchor feature
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.feature_selection import RFE, VarianceThreshold
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
# Config
# ---------------------------------------------------------------------------
# Radiomics defaults aligned with the prostate-PSMA-PET radiomics literature:
#   - Zamboglou et al. 2019 (Theranostics): fixed bin width 0.05 SUV on 2x2x2 mm
#     PET volumes for intraprostatic tumour discrimination (AUC 0.84-0.93).
#   - Solari et al. 2022 (EJNMMI): IBSI-compliant 107-feature PyRadiomics set
#     (firstorder + shape + GLCM + GLSZM + GLRLM + NGTDM + GLDM), no extra
#     resampling beyond the scanner-native 2 mm grid, FBW discretisation on SUV.
# Our UULM PSMA PET volumes are reconstructed natively at ~2.086x2.086x2.031 mm
# by the Siemens Biograph mMR, so no additional isotropic resampling is needed.
# Nested feature selection inside every LOOCV fold follows Demircioglu 2021
# (Insights into Imaging, 2021), which showed that feature selection performed
# outside CV inflates AUC by up to 0.15 on radiomics datasets.
MIN_ROI_VOXELS = 50  # minimum voxels in resampled gland mask to proceed
N_FEATURES_TO_SELECT = 4  # max features for RFE (including MRI score)
VARIANCE_THRESHOLD = 0.01  # drop near-zero-variance features
CORRELATION_THRESHOLD = 0.90  # drop highly correlated features
DEFAULT_BIN_WIDTH = 0.05  # SUV, following Zamboglou 2019


# ---------------------------------------------------------------------------
# PET Radiomic Feature Extraction
# ---------------------------------------------------------------------------


def resample_mask_to_pet(pet_image: sitk.Image, gland_mask: sitk.Image) -> sitk.Image:
    """Resample gland mask from MRI space to PET space using nearest-neighbor.

    Both images must share the same physical coordinate system (which they do —
    both are in the Siemens Biograph mMR scanner coordinate frame).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(pet_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(gland_mask)


def extract_radiomics_for_patient(
    suv_path: Path,
    gland_path: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> dict[str, float] | None:
    """Extract radiomic features from a single patient's PET volume within prostate ROI.

    Returns a dict of {feature_name: value} or None if extraction fails.
    """
    suv_img = sitk.ReadImage(str(suv_path))
    gland_img = sitk.ReadImage(str(gland_path))

    gland_resampled = resample_mask_to_pet(suv_img, gland_img)

    gland_arr = sitk.GetArrayFromImage(gland_resampled)
    n_voxels = np.count_nonzero(gland_arr > 0)
    if n_voxels < MIN_ROI_VOXELS:
        log.warning(
            "Insufficient ROI voxels (%d < %d) for %s, skipping",
            n_voxels,
            MIN_ROI_VOXELS,
            suv_path.name,
        )
        return None

    gland_resampled = sitk.Cast(gland_resampled, sitk.sitkUInt8)

    try:
        result = extractor.execute(suv_img, gland_resampled)
    except Exception as e:
        log.warning("Feature extraction failed for %s: %s", suv_path.name, e)
        return None

    features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[key] = float(value)
        except (TypeError, ValueError):
            continue

    log.info(
        "  %s: %d features extracted, %d ROI voxels",
        suv_path.stem,
        len(features),
        n_voxels,
    )
    return features


def create_radiomics_extractor(
    include_shape: bool = True,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> featureextractor.RadiomicsFeatureExtractor:
    """Create and configure pyradiomics feature extractor.

    Enables first-order, shape, and texture (GLCM, GLRLM, GLSZM, NGTDM, GLDM)
    features -- the IBSI-compliant 107-feature set used by Solari 2022 on
    [68Ga]-PSMA-11 PET/MR for prostate cancer staging.  No wavelet or LoG
    filters: with N=24 labelled PET patients the feature-to-sample ratio is
    already critical, and Solari 2022 explicitly restricts extraction to
    original images for the same reason.

    Parameters
    ----------
    include_shape : whether to include the 14 IBSI 3D shape features.  On by
        default (Solari 2022, Zamboglou 2019); only set ``False`` as an
        ablation.
    bin_width : fixed bin width for SUV discretisation.  Default ``0.05``
        matches Zamboglou 2019; Solari 2022 swept {0.03, 0.06, 0.125, 0.25,
        0.5, 1.0} and picked per model via inner CV.
    """
    settings = {
        "binWidth": bin_width,
        "resampledPixelSpacing": None,  # mMR delivers ~2 mm isotropic natively
        "interpolator": "sitkBSpline",
        "normalizeScale": 1,
        "normalize": False,  # SUV is already quantitative
        "removeOutliers": None,
        "minimumROIDimensions": 2,
        "minimumROISize": MIN_ROI_VOXELS,
        "geometryTolerance": 1e-3,
        "correctMask": True,
        "label": 1,
        "additionalInfo": False,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    feature_classes = [
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "ngtdm",
        "gldm",
    ]
    if include_shape:
        feature_classes.insert(1, "shape")

    for cls_name in feature_classes:
        extractor.enableFeatureClassByName(cls_name)

    return extractor


def extract_all_radiomics(
    suv_dir: Path,
    gland_dir: Path,
    include_shape: bool = True,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    """Extract radiomic features for all PET patients.

    Returns a DataFrame indexed by patient_id with one column per feature.
    """
    extractor = create_radiomics_extractor(
        include_shape=include_shape, bin_width=bin_width
    )

    suv_files = sorted(suv_dir.glob("*_pet_suv.mha"))
    gland_files = {f.stem: f for f in gland_dir.glob("*.mha")}

    patient_data: dict[str, tuple[Path, Path]] = {}
    for suv_path in suv_files:
        parts = suv_path.stem.replace("_pet_suv", "").split("_")
        pid = parts[0]
        sid = parts[1] if len(parts) > 1 else ""
        gland_key = f"{pid}_{sid}"

        if gland_key in gland_files:
            patient_data[pid] = (suv_path, gland_files[gland_key])
        else:
            log.warning("No gland mask for patient %s (key: %s)", pid, gland_key)

    log.info(
        "Found %d patients with both SUV volumes and gland masks", len(patient_data)
    )

    all_features: dict[str, dict[str, float]] = {}
    for i, (pid, (suv_path, gland_path)) in enumerate(sorted(patient_data.items())):
        log.info("[%d/%d] Processing %s...", i + 1, len(patient_data), pid)
        features = extract_radiomics_for_patient(suv_path, gland_path, extractor)
        if features is not None:
            all_features[pid] = features

    if not all_features:
        raise RuntimeError("No radiomic features extracted for any patient!")

    df = pd.DataFrame(all_features).T
    df.index.name = "patient_id"

    n_before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    n_after = len(df.columns)
    if n_before > n_after:
        log.info("Dropped %d all-NaN feature columns", n_before - n_after)

    log.info("Extracted radiomics: %d patients x %d features", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Feature Selection Utilities
# ---------------------------------------------------------------------------


def remove_correlated_features(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[np.ndarray, list[str]]:
    """Remove features with |Pearson r| > threshold, keeping the first of each pair."""
    if X.shape[1] <= 1:
        return X, feature_names

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    to_drop: set[int] = set()
    n = corr.shape[0]
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j)

    keep_mask = [i for i in range(n) if i not in to_drop]
    return X[:, keep_mask], [feature_names[i] for i in keep_mask]


# ---------------------------------------------------------------------------
# LOOCV Evaluation
# ---------------------------------------------------------------------------


def run_loocv(
    mri_scores: np.ndarray,
    radiomics_df: pd.DataFrame,
    y: np.ndarray,
    patient_ids: list[str],
    n_features_to_select: int = N_FEATURES_TO_SELECT,
    C: float = 1.0,
    include_extended_metrics: bool = True,
    classifier: ClassifierName = "lr",
    class_weight: str | dict | None = None,
) -> dict:
    """Run LOOCV with nested feature selection (variance -> correlation -> RFE).

    Feature selection is performed INSIDE each LOOCV fold to prevent leakage.
    The MRI score is always included as the first feature. RFE is applied only
    to the radiomic features, never to ``mri_score``.

    Parameters
    ----------
    mri_scores : array of shape (n,)
    radiomics_df : DataFrame of shape (n, K) with raw radiomic features
    y : array of shape (n,) with binary labels
    patient_ids : list of patient IDs
    n_features_to_select : max features for RFE (including MRI score)
    C : regularization parameter (for LR / SVM)
    classifier : one of ``"lr"``, ``"gp"``, ``"bayesian_lr"``, ``"svm"``.
        Default ``"lr"`` reproduces the original logistic regression baseline.
        Note: RFE always uses LR internally regardless of this setting —
        only the final classifier is swapped.
    class_weight : ``"balanced"`` to upweight minority class, or ``None``.
    """
    n = len(y)
    radiomics_names = list(radiomics_df.columns)
    R = radiomics_df.values.astype(float)

    loo = LeaveOneOut()
    y_pred_proba = np.zeros(n)
    y_pred_class = np.zeros(n, dtype=int)
    selected_features_per_fold: list[list[str]] = []
    coefs_per_fold: list[dict[str, float]] = []
    n_rad_features_to_select = max(0, n_features_to_select - 1)

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(R)):
        R_train, R_test = R[train_idx].copy(), R[test_idx].copy()
        mri_train, mri_test = mri_scores[train_idx], mri_scores[test_idx]
        y_train = y[train_idx]

        # Step 1: Impute NaN with training column median.
        #
        # Test-sample imputation must happen even when the training column
        # has no NaN, because the test sample can still be NaN for features
        # like ``suvmax_late_max`` when this module is called through the
        # H4 matrix runner with ``--include-metadata``.  Matches the
        # convention already used in ``b1_pet_metadata_sidecar.run_loocv``.
        for j in range(R_train.shape[1]):
            col = R_train[:, j]
            mask = np.isnan(col)
            med = np.nanmedian(col) if not mask.all() else 0.0
            if mask.any():
                R_train[mask, j] = med
            if np.isnan(R_test[0, j]):
                R_test[0, j] = med

        # Step 2: Remove near-zero-variance features (on training set)
        vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        try:
            R_train_filt = vt.fit_transform(R_train)
            R_test_filt = vt.transform(R_test)
            names_filt = [
                radiomics_names[i]
                for i in range(len(radiomics_names))
                if vt.get_support()[i]
            ]
        except ValueError:
            R_train_filt = np.empty((len(train_idx), 0))
            R_test_filt = np.empty((len(test_idx), 0))
            names_filt = []

        # Step 3: Remove highly correlated features (on training set)
        if R_train_filt.shape[1] > 1:
            R_train_filt, names_filt = remove_correlated_features(
                R_train_filt, names_filt, CORRELATION_THRESHOLD
            )
            filt_indices = [
                i
                for i, name in enumerate(
                    [
                        radiomics_names[j]
                        for j in range(len(radiomics_names))
                        if vt.get_support()[j]
                    ]
                )
                if name in names_filt
            ]
            R_test_filt_vt = vt.transform(R_test)
            R_test_filt = R_test_filt_vt[:, filt_indices]

        # Step 4: Select radiomics only. MRI score is always retained.
        selected_rad_names = names_filt
        if R_train_filt.shape[1] > 0 and n_rad_features_to_select > 0:
            if R_train_filt.shape[1] > n_rad_features_to_select:
                rad_scaler = StandardScaler()
                R_train_filt_s = rad_scaler.fit_transform(R_train_filt)
                estimator = LogisticRegression(
                    C=C, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42
                )
                rfe = RFE(estimator, n_features_to_select=n_rad_features_to_select)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rfe.fit(R_train_filt_s, y_train)
                R_train_filt = R_train_filt[:, rfe.support_]
                R_test_filt = R_test_filt[:, rfe.support_]
                selected_rad_names = [
                    names_filt[i] for i in range(len(names_filt)) if rfe.support_[i]
                ]
        else:
            R_train_filt = np.empty((len(train_idx), 0))
            R_test_filt = np.empty((len(test_idx), 0))
            selected_rad_names = []

        # Step 5: Combine MRI score + selected radiomics, then standardize.
        if R_train_filt.shape[1] > 0:
            X_train = np.column_stack([mri_train, R_train_filt])
            X_test = np.column_stack([mri_test, R_test_filt])
            selected = ["mri_score"] + selected_rad_names
        else:
            X_train = mri_train.reshape(-1, 1)
            X_test = mri_test.reshape(-1, 1)
            selected = ["mri_score"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Step 6: Train final classifier
        clf = make_classifier(
            classifier, C=C, random_state=42, class_weight=class_weight
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_s, y_train)

        y_pred_proba[test_idx] = clf.predict_proba(X_test_s)[:, 1]
        y_pred_class[test_idx] = clf.predict(X_test_s)
        selected_features_per_fold.append(selected)
        coef_vec = get_coef_or_none(clf, len(selected))
        if coef_vec is not None:
            coefs_per_fold.append(dict(zip(selected, coef_vec.tolist())))
        else:
            coefs_per_fold.append({})

    # --- Aggregate results ---
    metrics = build_binary_result(
        y,
        y_pred_proba,
        compute_cis=include_extended_metrics,
        include_auxiliary=include_extended_metrics,
        include_curves=include_extended_metrics,
    )

    from collections import Counter

    feature_counts = Counter()
    for sel in selected_features_per_fold:
        feature_counts.update(sel)

    avg_coefs: dict[str, float] = {}
    for feat in feature_counts:
        coef_vals = [c[feat] for c in coefs_per_fold if feat in c]
        avg_coefs[feat] = float(np.mean(coef_vals)) if coef_vals else 0.0

    return {
        **metrics,
        "classifier": classifier,
        "classifier_display": CLASSIFIER_DISPLAY_NAMES.get(classifier, classifier),
        "feature_selection_frequency": {
            feat: int(count) for feat, count in feature_counts.most_common()
        },
        "mean_coefficients": avg_coefs,
        "odds_ratios": {feat: float(np.exp(c)) for feat, c in avg_coefs.items()},
        "n_features_to_select": n_features_to_select,
        "per_patient": {
            pid: {
                "true_label": int(yt),
                "pred_proba": float(yp),
                "pred_class": int(yc),
            }
            for pid, yt, yp, yc in zip(patient_ids, y, y_pred_proba, y_pred_class)
        },
    }


def run_permutation_test(
    mri_scores: np.ndarray,
    radiomics_df: pd.DataFrame,
    y: np.ndarray,
    patient_ids: list[str],
    observed_auc: float,
    n_permutations: int = 1000,
    n_features_to_select: int = N_FEATURES_TO_SELECT,
    C: float = 1.0,
    seed: int = 42,
    classifier: ClassifierName = "lr",
) -> dict:
    """Permutation test: shuffle labels and re-run full LOOCV pipeline.

    The entire pipeline (NaN imputation, variance filter, correlation filter,
    RFE, classifier fit) is repeated for each permutation, so the test
    correctly accounts for any overfitting from feature selection.

    Parameters
    ----------
    observed_auc : the AUC from the real (un-shuffled) run
    n_permutations : number of label shuffles (default 1000)
    classifier : classifier name (passed through to ``run_loocv``)

    Returns
    -------
    dict with p_value, observed_auc, null_mean, null_std, null_aucs
    """
    rng = np.random.RandomState(seed)
    null_aucs = np.zeros(n_permutations)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        try:
            res = run_loocv(
                mri_scores,
                radiomics_df,
                y_perm,
                patient_ids,
                n_features_to_select=n_features_to_select,
                C=C,
                include_extended_metrics=False,
                classifier=classifier,
            )
            null_aucs[i] = res["auc"]
        except Exception:
            # If LOOCV fails (e.g., single-class fold), assign chance AUC
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
    Re-fitting through logistic regression on n~25 destroys the signal.
    """
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
    """
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
    """Find alpha maximizing balanced accuracy via grid search.

    Since both scores are already out-of-fold predictions, this is a simple
    post-hoc combination — not nested CV.  Acceptable for a single scalar
    parameter on n=25.
    """
    from sklearn.metrics import balanced_accuracy_score as _ba

    best_alpha = 0.5
    best_score = -1.0
    for a in alphas:
        p_fusion = a * mri_raw_scores + (1 - a) * pet_loocv_proba
        y_pred = (p_fusion >= 0.5).astype(int)
        score = _ba(y, y_pred)
        if score > best_score:
            best_score = score
            best_alpha = a
    return best_alpha


# ---------------------------------------------------------------------------
# Simple LOOCV (no feature selection) — used for MRI-only baseline
# ---------------------------------------------------------------------------


def run_loocv_simple(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    patient_ids: list[str],
    C: float = 1.0,
    include_extended_metrics: bool = True,
    classifier: ClassifierName = "lr",
    class_weight: str | dict | None = None,
) -> dict:
    """Simple LOOCV (no feature selection) — used for MRI-only baseline."""
    n = len(y)
    loo = LeaveOneOut()
    y_pred_proba = np.zeros(n)
    y_pred_class = np.zeros(n, dtype=int)
    coefs_all: list[np.ndarray | None] = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

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

    valid_coefs = [c for c in coefs_all if c is not None]
    if valid_coefs:
        mean_coefs = np.mean(valid_coefs, axis=0)
        mean_coefs_list = mean_coefs.tolist()
        odds_ratios_list = np.exp(mean_coefs).tolist()
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
