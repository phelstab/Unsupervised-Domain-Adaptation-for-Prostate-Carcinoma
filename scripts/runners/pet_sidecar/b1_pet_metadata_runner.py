"""B1 Runner: PET Metadata Sidecar — Argument handling and entry point.

Parses CLI arguments, loads MRI scores, calls B1 logic functions, prints
results, and saves output files.  Each run creates a timestamped directory
under ``workdir/pet/`` following the same conventions as Stage A
(``1_cnn_uda_runner.py``).

Usage::

    # Mode 1: Load MRI scores from a run directory (preferred — averages across folds)
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/b1_pet_metadata_runner.py \\
        --run-dir C:\\runs\\UULM_binary_class_3CV_RESNET10_GLAND_META\\<run_name> \\
        --da-weight 0.9

    # Mode 2: Load MRI scores from a single epoch outputs file
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/b1_pet_metadata_runner.py \\
        --outputs-pt workdir/uda/<run>/outputs/epoch_0100_outputs.pt

    # Mode 3: Load MRI scores from checkpoint + raw data (fallback)
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/b1_pet_metadata_runner.py \\
        --checkpoint workdir/uda/<run>/best_by_source_val.pt \\
        --backbone resnet10 --num-classes 2 --variant prostate_clinical

    # Mode 4: Dry-run without any MRI model (test pipeline with dummy scores)
    .venv-cnn\\Scripts\\python.exe scripts/runners/pet_sidecar/b1_pet_metadata_runner.py --dry-run

Output (timestamped run directory)::

    workdir/pet/YYYYMMDD_HHMMSS_b1_pet_metadata_<validator>/
        b1_results.json                    — AUC, accuracy, confusion matrix
        b1_predictions.csv                 — per-patient predictions
        b1_feature_importance.csv          — LR coefficients / odds ratios
        summary.txt                        — human-readable experiment summary
        b1_pet_metadata_runner_*.txt       — full log file
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

from b1_pet_metadata_sidecar import (
    build_feature_sets,
    evaluate_raw_mri,
    find_best_alpha,
    load_pet_metadata,
    run_loocv,
    run_permutation_test,
    run_score_fusion,
)
from mri_score_loader import (
    generate_dummy_mri_scores,
    load_mri_scores_from_checkpoint,
    load_mri_scores_from_outputs,
    load_mri_scores_from_run_dir,
)
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
OUTPUT_DIR = PROJECT_ROOT / "workdir" / "pet"

# Logging is configured in main() after the run directory is created.
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B1: PET Metadata Sidecar — post-hoc LR combining MRI score + PET spreadsheet metadata"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--run-dir",
        type=Path,
        help="Path to experiment run directory (Mode 1: average across folds)",
    )
    g.add_argument(
        "--outputs-pt",
        type=Path,
        help="Path to epoch_*_outputs.pt file (Mode 2: single file)",
    )
    g.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint .pt file (Mode 3: run fresh inference)",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Use dummy MRI scores (Mode 4: test pipeline without model)",
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
        "--backbone",
        default="resnet10",
        help="Model backbone (Mode 3 only, default: resnet10)",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (Mode 3 only, default: 2)",
    )
    p.add_argument(
        "--variant",
        default="prostate_clinical",
        help="Model variant (Mode 3 only, default: prostate_clinical)",
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

    Naming: ``YYYYMMDD_HHMMSS_b1_pet_metadata_<validator>[_C<val>]``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_dry = args.dry_run or (
        not args.run_dir and not args.outputs_pt and not args.checkpoint
    )
    parts = [timestamp, "b1_pet_metadata"]
    parts.append("dryrun" if is_dry else args.validator)
    if args.classifier != "lr":
        parts.append(args.classifier)
    if args.C != 1.0:
        parts.append(f"C{args.C}")
    if args.class_weight:
        parts.append("cw_balanced")
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
        / f"b1_pet_metadata_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
) -> None:
    """Write a human-readable ``summary.txt`` into the run directory."""
    summary_file = run_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("B1: PET METADATA SIDECAR — EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(
            f"Patients:  {len(common_pids)} "
            f"({int(y.sum())} positive, {int(len(y) - y.sum())} negative)\n"
        )
        f.write(
            f"Classifier: {CLASSIFIER_DISPLAY_NAMES.get(args.classifier, args.classifier)}\n"
        )
        f.write(f"LR regularization: C={args.C}\n")
        if args.run_dir:
            f.write(
                f"MRI source: {args.run_dir}\n"
                f"  da_weight={args.da_weight}, validator={args.validator}\n"
            )
        elif args.outputs_pt:
            f.write(f"MRI source: {args.outputs_pt}\n")
        elif args.checkpoint:
            f.write(f"MRI source: {args.checkpoint}\n")
        else:
            f.write("MRI source: dry-run (dummy scores)\n")
        f.write(f"Run directory: {run_dir}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 70 + "\n")
        for name, res in all_results.items():
            if name == "permutation_test":
                continue
            f.write(
                f"  {name:<22s} {format_metric_block(res, PRIMARY_METRIC_SPECS, include_ci=True)}\n"
            )
            f.write(
                f"{'':26s}{format_metric_block(res, THRESHOLD_METRIC_SPECS, include_ci=True)}\n"
            )

        if comparisons:
            f.write("\nPAIRED DELTAS VS MRI-ONLY:\n")
            f.write("-" * 70 + "\n")
            for name, comp in comparisons.items():
                f.write(f"  {name:<22s} {format_delta_block(comp, include_ci=True)}\n")
                transitions = comp.get("transition_counts", {})
                f.write(
                    f"{'':26s}rescued FN={transitions.get('rescued_false_negatives', 0)}  "
                    f"new FP={transitions.get('new_false_positives', 0)}  "
                    f"changed={transitions.get('changed_predictions', 0)}\n"
                )

        if "permutation_test" in all_results:
            f.write("\nPERMUTATION TEST:\n")
            f.write("-" * 70 + "\n")
            for name, perm in all_results["permutation_test"].items():
                f.write(
                    f"  {name:<22s} p = {perm['p_value']:.4f}  "
                    f"(null AUC = {perm['null_mean']:.3f} "
                    f"+/- {perm['null_std']:.3f})\n"
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
    log.info("B1 run directory: %s", run_dir)
    log.info("Arguments: %s", vars(args))

    # --- Step 1: Load PET metadata ---
    pet_meta = load_pet_metadata(args.pet_xlsx)

    # --- Step 2: Load MRI scores ---
    if args.run_dir:
        mri_scores = load_mri_scores_from_run_dir(
            args.run_dir,
            da_weight=args.da_weight,
            epoch=args.epoch,
            validator=args.validator,
        )
    elif args.outputs_pt:
        mri_scores = load_mri_scores_from_outputs(args.outputs_pt)
    elif args.checkpoint:
        mri_scores = load_mri_scores_from_checkpoint(
            args.checkpoint,
            backbone=args.backbone,
            num_classes=args.num_classes,
            variant=args.variant,
        )
    else:
        if not args.dry_run:
            log.warning(
                "No --run-dir, --outputs-pt, or --checkpoint specified. "
                "Defaulting to --dry-run mode."
            )
        mri_scores = generate_dummy_mri_scores(
            list(pet_meta.index), labels=pet_meta["label"]
        )

    # --- Step 3: Filter to common patients with valid labels ---
    common_pids = sorted(set(pet_meta.index) & set(mri_scores.keys()))
    common_pids = [pid for pid in common_pids if pd.notna(pet_meta.loc[pid, "label"])]
    log.info(
        "Common patients with valid labels: %d (of %d PET, %d MRI)",
        len(common_pids),
        len(pet_meta),
        len(mri_scores),
    )

    if len(common_pids) < 5:
        log.error(
            "Too few patients (%d) for meaningful LOOCV. Aborting.", len(common_pids)
        )
        sys.exit(1)

    # --- Step 4: Build feature matrices ---
    pet_sub = pet_meta.loc[common_pids]
    y = pet_sub["label"].values.astype(int)
    feature_sets = build_feature_sets(pet_sub, mri_scores, common_pids)

    # --- Step 5: Run LOOCV for each feature set ---
    clf_display = CLASSIFIER_DISPLAY_NAMES.get(args.classifier, args.classifier)
    print("\n" + "=" * 60)
    print(f"B1: PET Metadata Sidecar — LOOCV Results  [{clf_display}]")
    print("=" * 60)
    print(
        f"Patients: {len(common_pids)} "
        f"({int(y.sum())} positive, {int(len(y) - y.sum())} negative)"
    )
    print()

    all_results = {}
    for name, (X, feat_names) in feature_sets.items():
        res = run_loocv(
            X,
            y,
            feat_names,
            common_pids,
            C=args.C,
            classifier=args.classifier,
            class_weight=args.class_weight,
        )
        all_results[name] = res
        label = f"{name}:"
        print(f"  {label:<22s}{format_metric_block(res, PRIMARY_METRIC_SPECS)}")
        print(f"{'':24s}{format_metric_block(res, THRESHOLD_METRIC_SPECS)}")

    # --- Step 5a: Score-level fusion (MRI raw + PET LOOCV) ---
    # This addresses the architectural flaw where re-fitting calibrated MRI
    # probabilities through LOOCV LR on n~25 destroys the signal.
    mri_raw = np.array([mri_scores[pid] for pid in common_pids])

    print()
    print("--- Score-Level Fusion (no MRI re-fitting) ---")
    res_mri_raw = evaluate_raw_mri(mri_raw, y, common_pids)
    all_results["mri_raw"] = res_mri_raw
    print(
        f"  {'mri_raw:':<22s}{format_metric_block(res_mri_raw, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':24s}{format_metric_block(res_mri_raw, THRESHOLD_METRIC_SPECS)}")

    # Get PET-only LOOCV predictions for score fusion
    pet_loocv_proba = np.array(
        [
            all_results["pet_meta_only"]["per_patient"][pid]["pred_proba"]
            for pid in common_pids
        ]
    )

    # Fixed alpha = 0.5
    res_sf_50 = run_score_fusion(mri_raw, pet_loocv_proba, y, common_pids, alpha=0.5)
    all_results["score_fusion_a50"] = res_sf_50
    print(
        f"  {'score_fusion(a=0.5):':<22s}{format_metric_block(res_sf_50, PRIMARY_METRIC_SPECS)}"
    )
    print(f"{'':24s}{format_metric_block(res_sf_50, THRESHOLD_METRIC_SPECS)}")

    # Optimized alpha
    best_alpha = find_best_alpha(mri_raw, pet_loocv_proba, y)
    res_sf_opt = run_score_fusion(
        mri_raw, pet_loocv_proba, y, common_pids, alpha=best_alpha
    )
    all_results["score_fusion_opt"] = res_sf_opt
    label = f"score_fusion(a={best_alpha:.1f}):"
    print(f"  {label:<22s}{format_metric_block(res_sf_opt, PRIMARY_METRIC_SPECS)}")
    print(f"{'':24s}{format_metric_block(res_sf_opt, THRESHOLD_METRIC_SPECS)}")

    # Detail block for the main result (mri_pet_meta)
    res_main = all_results["mri_pet_meta"]
    comparisons = {
        "pet_meta_only": compare_binary_models(
            y,
            np.array(
                [
                    all_results["mri_only"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
            np.array(
                [
                    all_results["pet_meta_only"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
        ),
        "mri_clinical": compare_binary_models(
            y,
            np.array(
                [
                    all_results["mri_only"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
            np.array(
                [
                    all_results["mri_clinical"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
        ),
        "mri_pet_meta": compare_binary_models(
            y,
            np.array(
                [
                    all_results["mri_only"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
            np.array(
                [
                    all_results["mri_pet_meta"]["per_patient"][pid]["pred_proba"]
                    for pid in common_pids
                ]
            ),
        ),
    }
    print()
    if res_main["mean_coefficients"]:
        print(
            f"Feature importance (mri_pet_meta, mean {args.classifier.upper()} coefficients):"
        )
        for name, coef, odds in zip(
            res_main["feature_names"],
            res_main["mean_coefficients"],
            res_main["odds_ratios"],
        ):
            print(f"  {name:<22s}  coef={coef:+.4f}  OR={odds:.4f}")
    else:
        print(
            f"Feature importance: not available for {clf_display} "
            "(non-linear classifier, no coefficients)"
        )

    print()
    print("Paired deltas vs MRI-only:")
    for name, comp in comparisons.items():
        print(f"  {name:<22s}{format_delta_block(comp)}")
        transitions = comp.get("transition_counts", {})
        print(
            f"{'':24s}rescued FN={transitions.get('rescued_false_negatives', 0)}  "
            f"new FP={transitions.get('new_false_positives', 0)}  "
            f"changed={transitions.get('changed_predictions', 0)}"
        )

    print()
    print("Confusion matrix (mri_pet_meta):")
    cm = np.array(res_main["confusion_matrix"])
    print(f"  TN={cm[0, 0]:2d}  FP={cm[0, 1]:2d}")
    print(f"  FN={cm[1, 0]:2d}  TP={cm[1, 1]:2d}")
    print("=" * 60)

    # --- Step 5b: Permutation test (optional) ---
    if args.permutation_test > 0:
        print()
        print(f"Running permutation test ({args.permutation_test} shuffles)...")
        print()

        perm_results = {}
        for name, (X, feat_names) in feature_sets.items():
            observed_auc = all_results[name]["auc"]
            log.info("Permutation test for %s (AUC=%.3f)...", name, observed_auc)
            perm = run_permutation_test(
                X,
                y,
                feat_names,
                common_pids,
                observed_auc=observed_auc,
                n_permutations=args.permutation_test,
                C=args.C,
                seed=42 if name != "pet_meta_only" else 123,
                classifier=args.classifier,
            )
            perm_results[name] = perm
            print(
                f"  {name:<22s} observed AUC = {perm['observed_auc']:.3f}  "
                f"null mean = {perm['null_mean']:.3f} +/- {perm['null_std']:.3f}  "
                f"p = {perm['p_value']:.4f}"
            )

        all_results["permutation_test"] = perm_results
        print()

    # --- Step 6: Save results ---
    results_path = run_dir / "b1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # Per-patient predictions CSV
    res_mri = all_results["mri_only"]
    res_pet = all_results["pet_meta_only"]
    pred_rows = []
    for pid in common_pids:
        pred_rows.append(
            {
                "patient_id": pid,
                "true_label": int(pet_sub.loc[pid, "label"]),
                "mri_score": mri_scores[pid],
                "pred_mri_only": res_mri["per_patient"][pid]["pred_proba"],
                "pred_mri_raw": res_mri_raw["per_patient"][pid]["pred_proba"],
                "pred_pet_meta_only": res_pet["per_patient"][pid]["pred_proba"],
                "pred_mri_pet_meta": res_main["per_patient"][pid]["pred_proba"],
                "pred_score_fusion": res_sf_opt["per_patient"][pid]["pred_proba"],
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_path = run_dir / "b1_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info("Predictions saved to %s", pred_path)

    metric_rows = [
        flatten_result_for_csv(name, res)
        for name, res in all_results.items()
        if name != "permutation_test"
    ]
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = run_dir / "b1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Metrics saved to %s", metrics_path)

    comp_rows = [
        flatten_comparison_for_csv(name, comp) for name, comp in comparisons.items()
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_path = run_dir / "b1_model_deltas.csv"
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
    threshold_path = run_dir / "b1_threshold_analysis.csv"
    threshold_df.to_csv(threshold_path, index=False)
    log.info("Threshold analysis saved to %s", threshold_path)

    decision_df = pd.DataFrame(decision_rows)
    decision_path = run_dir / "b1_decision_curve.csv"
    decision_df.to_csv(decision_path, index=False)
    log.info("Decision curve saved to %s", decision_path)

    # Feature importance CSV
    fi_rows = []
    if res_main["mean_coefficients"]:
        for feat_name, coef, odds in zip(
            res_main["feature_names"],
            res_main["mean_coefficients"],
            res_main["odds_ratios"],
        ):
            fi_rows.append(
                {"feature": feat_name, "coefficient": coef, "odds_ratio": odds}
            )
    else:
        # Non-linear classifier — record feature names without coefficients
        for feat_name in res_main["feature_names"]:
            fi_rows.append(
                {"feature": feat_name, "coefficient": None, "odds_ratio": None}
            )
    fi_df = pd.DataFrame(fi_rows)
    fi_path = run_dir / "b1_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    log.info("Feature importance saved to %s", fi_path)

    # --- Step 7: Generate summary ---
    _generate_summary(run_dir, args, all_results, comparisons, common_pids, y)
    log.info("All experiments completed! Results saved to: %s", run_dir)


if __name__ == "__main__":
    main()
