"""C1 Runner: PET Image Radiomics Fusion — Argument handling and entry point.

Parses CLI arguments, extracts or loads radiomic features, loads MRI scores,
calls C1 logic functions, prints results, and saves output files.  Each run
creates a timestamped directory under ``workdir/pet/`` following the same
conventions as Stage A (``1_cnn_uda_runner.py``).

Prerequisites::

    # 1. PET DICOM data must exist under 0ii/files/pet/.
    #    SUV volumes are generated automatically on first run
    #    (catalog + DICOM→SUV conversion) if not already present.

    # 2. Gland masks must exist:
    0ii/files/gland_masks/{pid}_{sid}.mha

Usage::

    # Mode 1: Load MRI scores from a run directory (preferred)
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/c1_pet_radiomics_runner.py \\
        --run-dir C:\\runs\\UULM_binary_class_3CV_RESNET10_GLAND_META\\<run_name> \\
        --da-weight 0.9

    # Mode 2: Dry-run (dummy MRI scores, test pipeline)
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/c1_pet_radiomics_runner.py --dry-run

Output (timestamped run directory)::

    workdir/pet/YYYYMMDD_HHMMSS_c1_pet_radiomics_<validator>_<n>feat/
        c1_results.json                    — AUC, accuracy, confusion matrix
        c1_predictions.csv                 — per-patient predictions
        c1_feature_importance.csv          — selected features and importance
        summary.txt                        — human-readable experiment summary
        c1_pet_radiomics_runner_*.txt      — full log file

    workdir/pet/c1_radiomics_features.csv  — cached radiomics (stable, shared)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "MRI" / "baseline" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from b1_pet_metadata_sidecar import PET_FEATURE_COLS, load_pet_metadata
from c1_pet_radiomics_fusion import (
    DEFAULT_BIN_WIDTH,
    N_FEATURES_TO_SELECT,
    evaluate_raw_mri,
    extract_all_radiomics,
    find_best_alpha,
    run_loocv,
    run_loocv_simple,
    run_permutation_test,
    run_score_fusion,
)
from mri_score_loader import (
    generate_dummy_mri_scores,
    load_mri_scores_from_run_dir,
)
from pet_data_explorer import CATALOG_PATH, build_catalog
from pet_dicom_to_suv import load_pet_series_as_suv
from sidecar_classifiers import CLASSIFIER_CHOICES, CLASSIFIER_DISPLAY_NAMES
from sidecar_metrics import (
    PRIMARY_METRIC_SPECS,
    THRESHOLD_METRIC_SPECS,
    compare_binary_models,
    flatten_decision_curve_for_csv,
    flatten_comparison_for_csv,
    flatten_result_for_csv,
    flatten_threshold_analysis_for_csv,
    format_delta_block,
    format_metric_block,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PET_XLSX = PROJECT_ROOT / "0ii" / "pet.xlsx"
PET_SHEET = "Auswertung"
SUV_DIR = PROJECT_ROOT / "workdir" / "pet" / "suv_volumes"
GLAND_DIR = PROJECT_ROOT / "0ii" / "files" / "gland_masks"
OUTPUT_DIR = PROJECT_ROOT / "workdir" / "pet"

COL_PATIENT_ID = "PatientenID"
COL_LABEL = "Gleason"

# Logging is configured in main() after the run directory is created.
log = logging.getLogger(__name__)

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

    Returns 0 for Gleason <= 6 (GG1), non-cancer findings, or value 0
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
# Adaptive prerequisite helpers
# ---------------------------------------------------------------------------


def _ensure_suv_volumes(suv_dir: Path) -> None:
    """Run prerequisite steps if SUV volumes are missing.

    1. Build the PET series catalog (``pet_data_explorer``) if it does not exist.
    2. Convert PET DICOM → SUV ``.mha`` (``pet_dicom_to_suv``) for every
       cataloged patient whose output file is not yet present.
    """
    import SimpleITK as sitk

    existing = set(suv_dir.glob("*_pet_suv.mha")) if suv_dir.is_dir() else set()
    if existing:
        log.info(
            "SUV volumes directory already contains %d files, skipping conversion.",
            len(existing),
        )
        return

    # --- Step A: build catalog if needed ---
    if not CATALOG_PATH.exists():
        log.info("PET series catalog not found – running pet_data_explorer ...")
        catalog = build_catalog()
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CATALOG_PATH, "w") as f:
            json.dump(catalog, f, indent=2)
        log.info("Catalog written to %s (%d patients)", CATALOG_PATH, len(catalog))
    else:
        log.info("Loading existing PET series catalog from %s", CATALOG_PATH)
        with open(CATALOG_PATH) as f:
            catalog = json.load(f)

    # --- Step B: convert DICOM → SUV for missing patients ---
    suv_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = sorted(catalog.keys())
    log.info("Converting PET DICOM → SUV for %d patients ...", len(patient_ids))

    success, failed = 0, []
    for pid in patient_ids:
        entry = catalog[pid]
        study_id = entry["study_id"]
        output_path = suv_dir / f"{pid}_{study_id}_pet_suv.mha"
        if output_path.exists():
            success += 1
            continue
        series_path = Path(entry["series_path"])
        log.info("  Converting %s (study %s) ...", pid, study_id)
        try:
            suv_image = load_pet_series_as_suv(series_path)
            sitk.WriteImage(suv_image, str(output_path))
            success += 1
        except Exception as exc:
            log.error("  FAILED for %s: %s", pid, exc)
            failed.append(pid)

    log.info("SUV conversion done: %d/%d succeeded", success, len(patient_ids))
    if failed:
        log.warning("Failed patients: %s", failed)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C1: PET Image Radiomics Fusion — post-hoc LR combining MRI score + PET radiomic features"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--run-dir",
        type=Path,
        help="Path to experiment run directory (average MRI scores across folds)",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Use dummy MRI scores (test pipeline without model)",
    )
    p.add_argument(
        "--da-weight",
        default="0.9",
        help="DA loss weight for --run-dir mode (default: 0.9)",
    )
    p.add_argument(
        "--epoch",
        default=None,
        help="Epoch to use (e.g., '0099' or 'last'). Default: validator selection.",
    )
    p.add_argument(
        "--validator",
        default="source_val",
        choices=["source_val", "last"],
        help="Validator for epoch selection (default: source_val)",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression regularization (default: 1.0)",
    )
    p.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="lr",
        help=(
            "Classifier for the LOOCV fusion stage.  "
            "Choices: lr (default), gp, bayesian_lr, svm.  "
            "Use 'all' via the batch script to run every classifier."
        ),
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=N_FEATURES_TO_SELECT,
        help=f"Max features for RFE including MRI score (default: {N_FEATURES_TO_SELECT})",
    )
    p.add_argument(
        "--suv-dir",
        type=Path,
        default=SUV_DIR,
        help="Directory with SUV .mha files",
    )
    p.add_argument(
        "--gland-dir",
        type=Path,
        default=GLAND_DIR,
        help="Directory with gland mask .mha files",
    )
    p.add_argument(
        "--pet-xlsx",
        type=Path,
        default=PET_XLSX,
        help="Path to pet.xlsx",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: workdir/pet)",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip radiomics extraction (load from c1_radiomics_features.csv)",
    )
    p.add_argument(
        "--include-shape-features",
        action="store_true",
        default=True,
        help=(
            "Include ROI shape features derived from the resampled gland mask.  "
            "ON by default, matching Solari 2022 and Zamboglou 2019 radiomics "
            "pipelines.  Use ``--no-shape-features`` to turn them off as an "
            "ablation."
        ),
    )
    p.add_argument(
        "--no-shape-features",
        dest="include_shape_features",
        action="store_false",
        help="Disable shape features (ablation; default ON).",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help=(
            "Fixed bin width for SUV discretisation (FBW).  Default 0.05 "
            "follows Zamboglou 2019 on PSMA PET; Solari 2022 swept "
            "{0.03, 0.06, 0.125, 0.25, 0.5, 1.0}."
        ),
    )
    p.add_argument(
        "--include-metadata",
        action="store_true",
        help=(
            "Include PET metadata features from pet.xlsx (n_psma_lesions, "
            "pirads_max, suvmax_late_max) in the radiomics feature pool.  "
            "RFE selects from both radiomics AND metadata columns."
        ),
    )
    p.add_argument(
        "--class-weight",
        choices=["balanced"],
        default=None,
        help=(
            "Class weighting strategy for the LOOCV classifier.  "
            "'balanced' upweights the minority class proportional to "
            "its inverse frequency.  Default: None (equal weights)."
        ),
    )
    p.add_argument(
        "--permutation-test",
        type=int,
        default=0,
        metavar="N",
        help="Run permutation test with N shuffles (e.g., 1000). 0 = skip.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run-directory & logging setup (matching Stage A conventions)
# ---------------------------------------------------------------------------


def _build_run_dir(base_dir: Path, args: argparse.Namespace) -> Path:
    """Create a timestamped run directory under *base_dir*.

    Naming: ``YYYYMMDD_HHMMSS_c1_pet_radiomics_<validator>_<n>feat[_C<val>]``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_dry = args.dry_run or not args.run_dir
    parts = [timestamp, "c1_pet_radiomics"]
    parts.append("dryrun" if is_dry else args.validator)
    parts.append(f"{args.n_features}feat")
    if args.classifier != "lr":
        parts.append(args.classifier)
    if args.C != 1.0:
        parts.append(f"C{args.C}")
    if args.class_weight:
        parts.append("cw_balanced")
    if args.include_metadata:
        parts.append("with_meta")
    if args.permutation_test > 0:
        parts.append(f"perm{args.permutation_test}")
    folder_name = "_".join(parts)
    run_dir = base_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logging(run_dir: Path) -> None:
    """Configure root logger with file + console handlers (Stage A style)."""
    log_file = (
        run_dir
        / f"c1_pet_radiomics_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _generate_summary(
    run_dir: Path,
    args: argparse.Namespace,
    all_results: dict,
    comparisons: dict[str, dict],
    common_pids: list[str],
    y: np.ndarray,
    n_radiomics: int,
) -> None:
    """Write a human-readable ``summary.txt`` into the run directory."""
    summary_file = run_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        meta_tag = " + METADATA" if args.include_metadata else ""
        f.write(f"C1: PET IMAGE RADIOMICS{meta_tag} FUSION — EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(
            f"Patients:  {len(common_pids)} "
            f"({int(y.sum())} positive, {int(len(y) - y.sum())} negative)\n"
        )
        f.write(f"Feature pool: {n_radiomics} columns{meta_tag.lower()}\n")
        f.write(
            f"Classifier: {CLASSIFIER_DISPLAY_NAMES.get(args.classifier, args.classifier)}\n"
        )
        f.write(f"RFE target: {args.n_features} features (including MRI score)\n")
        f.write(f"LR regularization: C={args.C}\n")
        if args.run_dir:
            f.write(
                f"MRI source: {args.run_dir}\n"
                f"  da_weight={args.da_weight}, validator={args.validator}\n"
            )
        else:
            f.write("MRI source: dry-run (dummy scores)\n")
        f.write(f"Run directory: {run_dir}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        for name, res in all_results.items():
            if name == "permutation_test":
                continue
            f.write(
                f"  {name:<24s} {format_metric_block(res, PRIMARY_METRIC_SPECS, include_ci=True)}\n"
            )
            f.write(
                f"{'':28s}{format_metric_block(res, THRESHOLD_METRIC_SPECS, include_ci=True)}\n"
            )

        if comparisons:
            f.write("\nPAIRED DELTAS VS MRI-ONLY:\n")
            f.write("-" * 70 + "\n")
            for name, comp in comparisons.items():
                f.write(f"  {name:<24s} {format_delta_block(comp, include_ci=True)}\n")
                transitions = comp.get("transition_counts", {})
                f.write(
                    f"{'':28s}rescued FN={transitions.get('rescued_false_negatives', 0)}  "
                    f"new FP={transitions.get('new_false_positives', 0)}  "
                    f"changed={transitions.get('changed_predictions', 0)}\n"
                )

        if "permutation_test" in all_results:
            f.write("\nPERMUTATION TEST:\n")
            f.write("-" * 70 + "\n")
            for name, perm in all_results["permutation_test"].items():
                f.write(
                    f"  {name:<24s} p = {perm['p_value']:.4f}  "
                    f"(null AUC = {perm['null_mean']:.3f} "
                    f"+/- {perm['null_std']:.3f})\n"
                )

        # Feature selection frequency
        res_c1 = all_results.get("mri_pet_radiomics")
        if res_c1 and "feature_selection_frequency" in res_c1:
            f.write("\nFEATURE SELECTION FREQUENCY (MRI+PET-radiomics):\n")
            f.write("-" * 70 + "\n")
            freq = res_c1["feature_selection_frequency"]
            for feat, count in sorted(freq.items(), key=lambda x: -x[1])[:15]:
                pct = count / len(common_pids) * 100
                coef = res_c1["mean_coefficients"].get(feat, 0.0)
                f.write(
                    f"  {feat:<50s} {count:2d}/{len(common_pids)} "
                    f"({pct:5.1f}%)  coef={coef:+.4f}\n"
                )

        f.write("\n" + "=" * 70 + "\n")
    log.info("Summary saved to %s", summary_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Create timestamped run directory & configure logging ---
    run_dir = _build_run_dir(args.output_dir, args)
    _setup_logging(run_dir)
    log.info("C1 run directory: %s", run_dir)
    log.info("Arguments: %s", vars(args))

    # --- Step 1: Load labels from pet.xlsx ---
    log.info("Loading labels from %s", args.pet_xlsx)
    pet_df = pd.read_excel(args.pet_xlsx, sheet_name=PET_SHEET)
    pet_df[COL_PATIENT_ID] = pet_df[COL_PATIENT_ID].apply(
        lambda x: str(int(x)).zfill(10) if pd.notna(x) else None
    )
    pet_df = pet_df.set_index(COL_PATIENT_ID)
    labels = pet_df[COL_LABEL].apply(_gleason_to_cspca_label)

    # --- Step 2: Extract or load radiomic features ---
    # The cache lives in the *base* output dir (not the timestamped run dir)
    # so it persists across runs. Cache files are keyed by the radiomics
    # configuration (shape flag, bin width) so the runner cannot silently
    # reuse incompatible features.
    shape_tag = "with_shape" if args.include_shape_features else "no_shape"
    bw_tag = f"bw{str(args.bin_width).replace('.', 'p')}"
    features_csv = args.output_dir / f"c1_radiomics_features_{shape_tag}_{bw_tag}.csv"

    if features_csv.exists():
        log.info("Loading cached radiomics features from %s", features_csv)
        radiomics_df = pd.read_csv(features_csv, index_col="patient_id")
        radiomics_df.index = radiomics_df.index.astype(str).str.zfill(10)
    else:
        if args.skip_extraction:
            log.warning(
                "--skip-extraction was specified but cache not found: %s  "
                "Falling through to extraction.",
                features_csv,
            )
        # Ensure SUV volumes exist (runs catalog + DICOM->SUV if needed)
        _ensure_suv_volumes(args.suv_dir)
        log.info(
            "Extracting radiomic features from PET volumes "
            "(shape=%s, bin_width=%.3f SUV)...",
            args.include_shape_features,
            args.bin_width,
        )
        radiomics_df = extract_all_radiomics(
            suv_dir=args.suv_dir,
            gland_dir=args.gland_dir,
            include_shape=args.include_shape_features,
            bin_width=args.bin_width,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        radiomics_df.to_csv(features_csv)
        log.info("Saved radiomics features to %s", features_csv)

    # --- Step 3: Load MRI scores ---
    if args.run_dir:
        mri_scores = load_mri_scores_from_run_dir(
            args.run_dir,
            da_weight=args.da_weight,
            epoch=args.epoch,
            validator=args.validator,
        )
    else:
        if not args.dry_run:
            log.warning("No --run-dir specified. Defaulting to --dry-run mode.")
        mri_scores = generate_dummy_mri_scores(list(radiomics_df.index))

    # --- Step 4: Align patients (common set with labels + radiomics + MRI scores) ---
    common_pids = sorted(
        set(radiomics_df.index) & set(mri_scores.keys()) & set(labels.index)
    )
    common_pids = [pid for pid in common_pids if pd.notna(labels.get(pid))]
    log.info(
        "Common patients with valid labels: %d (radiomics=%d, mri=%d, labels=%d)",
        len(common_pids),
        len(radiomics_df),
        len(mri_scores),
        labels.notna().sum(),
    )

    if len(common_pids) < 5:
        log.error(
            "Too few patients (%d) for meaningful LOOCV. Aborting.", len(common_pids)
        )
        sys.exit(1)

    # Build aligned arrays
    y = labels.loc[common_pids].values.astype(int)
    mri_arr = np.array([mri_scores[pid] for pid in common_pids])
    rad_df = radiomics_df.loc[common_pids]

    # --- Optional: merge B1 metadata features into the radiomics pool ---
    if args.include_metadata:
        log.info("Loading PET metadata features to merge with radiomics...")
        pet_meta = load_pet_metadata(args.pet_xlsx)
        meta_cols = [c for c in PET_FEATURE_COLS if c in pet_meta.columns]
        meta_sub = pet_meta.loc[common_pids, meta_cols].copy()
        n_before = len(rad_df.columns)
        rad_df = pd.concat([rad_df, meta_sub], axis=1)
        log.info(
            "Merged %d metadata columns into radiomics pool: %d → %d features",
            len(meta_cols),
            n_before,
            len(rad_df.columns),
        )

    # --- Step 5: Run LOOCV for multiple feature sets ---
    clf_display = CLASSIFIER_DISPLAY_NAMES.get(args.classifier, args.classifier)
    meta_tag = " + metadata" if args.include_metadata else ""
    print("\n" + "=" * 70)
    print(f"C1: PET Image Radiomics{meta_tag} Fusion — LOOCV Results  [{clf_display}]")
    print("=" * 70)
    print(
        f"Patients: {len(common_pids)} ({int(y.sum())} positive, "
        f"{int(len(y) - y.sum())} negative)"
    )
    print(f"Feature pool: {len(rad_df.columns)} columns{meta_tag}")
    print(f"RFE target: {args.n_features} features (including MRI score)")
    print()

    all_results = {}

    # 5a: MRI-only baseline
    res_mri = run_loocv_simple(
        mri_arr.reshape(-1, 1),
        y,
        ["mri_score"],
        common_pids,
        C=args.C,
        classifier=args.classifier,
        class_weight=args.class_weight,
    )
    all_results["mri_only"] = res_mri
    print(
        f"  MRI-only:              {format_metric_block(res_mri, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':26s}{format_metric_block(res_mri, THRESHOLD_METRIC_SPECS)}")

    # 5b: MRI + PET radiomics (main C1 result)
    res_c1 = run_loocv(
        mri_arr,
        rad_df,
        y,
        common_pids,
        n_features_to_select=args.n_features,
        C=args.C,
        classifier=args.classifier,
        class_weight=args.class_weight,
    )
    all_results["mri_pet_radiomics"] = res_c1
    print(
        f"  MRI+PET-radiomics:     {format_metric_block(res_c1, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':26s}{format_metric_block(res_c1, THRESHOLD_METRIC_SPECS)}")

    # 5c: PET radiomics only (ablation — zero MRI scores)
    res_rad_only = run_loocv(
        np.zeros(len(y)),
        rad_df,
        y,
        common_pids,
        n_features_to_select=args.n_features,
        C=args.C,
        classifier=args.classifier,
        class_weight=args.class_weight,
    )
    all_results["pet_radiomics_only"] = res_rad_only
    print(
        f"  PET-radiomics only:    {format_metric_block(res_rad_only, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':26s}{format_metric_block(res_rad_only, THRESHOLD_METRIC_SPECS)}")

    # 5d: Score-level fusion (MRI raw + PET LOOCV) — no MRI re-fitting
    print()
    print("--- Score-Level Fusion (no MRI re-fitting) ---")
    res_mri_raw = evaluate_raw_mri(mri_arr, y, common_pids)
    all_results["mri_raw"] = res_mri_raw
    print(
        f"  MRI raw P(csPCa):      {format_metric_block(res_mri_raw, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':26s}{format_metric_block(res_mri_raw, THRESHOLD_METRIC_SPECS)}")

    # Get PET-only LOOCV predictions for score fusion
    pet_loocv_proba = np.array(
        [res_rad_only["per_patient"][pid]["pred_proba"] for pid in common_pids]
    )

    # Fixed alpha = 0.5
    res_sf_50 = run_score_fusion(mri_arr, pet_loocv_proba, y, common_pids, alpha=0.5)
    all_results["score_fusion_a50"] = res_sf_50
    print(
        f"  Score fusion (a=0.5):  {format_metric_block(res_sf_50, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':26s}{format_metric_block(res_sf_50, THRESHOLD_METRIC_SPECS)}")

    # Optimized alpha
    best_alpha = find_best_alpha(mri_arr, pet_loocv_proba, y)
    res_sf_opt = run_score_fusion(
        mri_arr, pet_loocv_proba, y, common_pids, alpha=best_alpha
    )
    all_results["score_fusion_opt"] = res_sf_opt
    label_sf = f"Score fusion (a={best_alpha:.1f}):"
    print(f"  {label_sf:<24s} {format_metric_block(res_sf_opt, PRIMARY_METRIC_SPECS)}")
    print(f"{'':26s}{format_metric_block(res_sf_opt, THRESHOLD_METRIC_SPECS)}")

    comparisons = {
        "mri_pet_radiomics": compare_binary_models(
            y,
            np.array(
                [res_mri["per_patient"][pid]["pred_proba"] for pid in common_pids]
            ),
            np.array([res_c1["per_patient"][pid]["pred_proba"] for pid in common_pids]),
        ),
        "pet_radiomics_only": compare_binary_models(
            y,
            np.array(
                [res_mri["per_patient"][pid]["pred_proba"] for pid in common_pids]
            ),
            np.array(
                [res_rad_only["per_patient"][pid]["pred_proba"] for pid in common_pids]
            ),
        ),
    }

    # Detail block
    print()
    print("Feature selection frequency (MRI+PET-radiomics, across LOOCV folds):")
    freq = res_c1["feature_selection_frequency"]
    for feat, count in sorted(freq.items(), key=lambda x: -x[1])[:15]:
        pct = count / len(common_pids) * 100
        coef = res_c1["mean_coefficients"].get(feat, 0.0)
        odds = res_c1["odds_ratios"].get(feat, 1.0)
        print(
            f"  {feat:<50s}  selected {count:2d}/{len(common_pids)} ({pct:5.1f}%)  "
            f"coef={coef:+.4f}  OR={odds:.4f}"
        )

    print()
    print("Paired deltas vs MRI-only:")
    for name, comp in comparisons.items():
        print(f"  {name:<24s}{format_delta_block(comp)}")
        transitions = comp.get("transition_counts", {})
        print(
            f"{'':26s}rescued FN={transitions.get('rescued_false_negatives', 0)}  "
            f"new FP={transitions.get('new_false_positives', 0)}  "
            f"changed={transitions.get('changed_predictions', 0)}"
        )

    print()
    print("Confusion matrix (MRI+PET-radiomics):")
    cm = np.array(res_c1["confusion_matrix"])
    print(f"  TN={cm[0, 0]:2d}  FP={cm[0, 1]:2d}")
    print(f"  FN={cm[1, 0]:2d}  TP={cm[1, 1]:2d}")
    print("=" * 70)

    # --- Step 5d: Permutation test (optional) ---
    if args.permutation_test > 0:
        print()
        print(f"Running permutation test ({args.permutation_test} shuffles)...")
        print("  This tests whether the MRI+PET-radiomics AUC is significantly")
        print("  better than chance given the full feature-selection pipeline.")
        print()

        perm_results = {}

        # Permutation test for MRI+PET radiomics
        log.info("Permutation test for MRI+PET-radiomics (AUC=%.3f)...", res_c1["auc"])
        perm_c1 = run_permutation_test(
            mri_arr,
            rad_df,
            y,
            common_pids,
            observed_auc=res_c1["auc"],
            n_permutations=args.permutation_test,
            n_features_to_select=args.n_features,
            C=args.C,
            classifier=args.classifier,
        )
        perm_results["mri_pet_radiomics"] = perm_c1
        print(
            f"  MRI+PET-radiomics:  observed AUC = {perm_c1['observed_auc']:.3f}  "
            f"null mean = {perm_c1['null_mean']:.3f} +/- {perm_c1['null_std']:.3f}  "
            f"p = {perm_c1['p_value']:.4f}"
        )

        # Permutation test for PET radiomics only
        log.info(
            "Permutation test for PET-radiomics only (AUC=%.3f)...",
            res_rad_only["auc"],
        )
        perm_rad = run_permutation_test(
            np.zeros(len(y)),
            rad_df,
            y,
            common_pids,
            observed_auc=res_rad_only["auc"],
            n_permutations=args.permutation_test,
            n_features_to_select=args.n_features,
            C=args.C,
            seed=123,
            classifier=args.classifier,
        )
        perm_results["pet_radiomics_only"] = perm_rad
        print(
            f"  PET-radiomics only: observed AUC = {perm_rad['observed_auc']:.3f}  "
            f"null mean = {perm_rad['null_mean']:.3f} +/- {perm_rad['null_std']:.3f}  "
            f"p = {perm_rad['p_value']:.4f}"
        )

        all_results["permutation_test"] = perm_results
        print()

    # --- Step 6: Save results ---
    results_path = run_dir / "c1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # Per-patient predictions CSV
    pred_rows = []
    for pid in common_pids:
        pred_rows.append(
            {
                "patient_id": pid,
                "true_label": int(labels.loc[pid]),
                "mri_score": mri_scores[pid],
                "pred_mri_only": res_mri["per_patient"][pid]["pred_proba"],
                "pred_mri_raw": res_mri_raw["per_patient"][pid]["pred_proba"],
                "pred_mri_pet_radiomics": res_c1["per_patient"][pid]["pred_proba"],
                "pred_pet_radiomics_only": res_rad_only["per_patient"][pid][
                    "pred_proba"
                ],
                "pred_score_fusion": res_sf_opt["per_patient"][pid]["pred_proba"],
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_path = run_dir / "c1_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info("Predictions saved to %s", pred_path)

    metric_rows = [
        flatten_result_for_csv(name, res)
        for name, res in all_results.items()
        if name != "permutation_test"
    ]
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = run_dir / "c1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Metrics saved to %s", metrics_path)

    comp_rows = [
        flatten_comparison_for_csv(name, comp) for name, comp in comparisons.items()
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_path = run_dir / "c1_model_deltas.csv"
    comp_df.to_csv(comp_path, index=False)
    log.info("Model deltas saved to %s", comp_path)

    threshold_rows = []
    decision_rows = []
    for name, res in all_results.items():
        if name == "permutation_test":
            continue
        threshold_rows.extend(flatten_threshold_analysis_for_csv(name, res))
        decision_rows.extend(flatten_decision_curve_for_csv(name, res))

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_path = run_dir / "c1_threshold_analysis.csv"
    threshold_df.to_csv(threshold_path, index=False)
    log.info("Threshold analysis saved to %s", threshold_path)

    decision_df = pd.DataFrame(decision_rows)
    decision_path = run_dir / "c1_decision_curve.csv"
    decision_df.to_csv(decision_path, index=False)
    log.info("Decision curve saved to %s", decision_path)

    # Feature importance CSV
    fi_rows = []
    for feat, count in res_c1["feature_selection_frequency"].items():
        fi_rows.append(
            {
                "feature": feat,
                "selection_count": count,
                "selection_pct": count / len(common_pids) * 100,
                "mean_coefficient": res_c1["mean_coefficients"].get(feat, 0.0),
                "odds_ratio": res_c1["odds_ratios"].get(feat, 1.0),
            }
        )
    fi_df = pd.DataFrame(fi_rows).sort_values("selection_count", ascending=False)
    fi_path = run_dir / "c1_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    log.info("Feature importance saved to %s", fi_path)

    # --- Step 7: Generate summary ---
    _generate_summary(
        run_dir, args, all_results, comparisons, common_pids, y, len(rad_df.columns)
    )
    log.info("All experiments completed! Results saved to: %s", run_dir)


if __name__ == "__main__":
    main()
