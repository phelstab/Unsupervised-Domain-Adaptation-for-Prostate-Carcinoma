"""H4 matrix runner: the 6-validator x 2-fusion PET matrix from Sec. 3.6.

For each unsupervised/oracle validator V, loads the OOF MRI score vector that
``scripts/a4_pool_score_extraction.py`` produced, then runs both B1 (PET
metadata sidecar) and C1 (PET image radiomics + LR) under LOOCV.  The result
is the 6 x 2 validator x fusion table promised by thesis Section 3.6:

Each cell reports balanced accuracy, AUC, sensitivity, specificity, 95 %
bootstrap CI, and optional permutation test p-value.

Usage
-----
    # Step 1: build the per-validator MRI-score cache (once).
    .venv-cnn\\Scripts\\python.exe scripts\\a4_pool_score_extraction.py

    # Step 2: run B1 + C1 for all six validators.
    .venv-cnn\\Scripts\\python.exe scripts\\runners\\pet_sidecar\\h4_matrix_runner.py

    # Optional: add permutation test (1000 shuffles) for significance.
    .venv-cnn\\Scripts\\python.exe scripts\\runners\\pet_sidecar\\h4_matrix_runner.py --permutation-test 1000

Output (timestamped run directory)::

    workdir/pet/YYYYMMDD_HHMMSS_h4_matrix[_perm1000]/
        h4_matrix.csv             - the 6 x 2 (validator x fusion) matrix
        h4_matrix.tex             - LaTeX table ready to paste into results.tex
        h4_predictions.csv        - per-patient OOF predictions for every cell
        h4_selection_audit.csv    - (method, da_weight, epoch) per fold per V
        summary.txt               - human-readable report
        h4_matrix_runner_*.txt    - full log file
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from a4_pool_score_extraction import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    VALIDATOR_REGISTRY,
    load_scores,
    load_selection,
)
from b1_pet_metadata_sidecar import (  # noqa: E402
    PET_FEATURE_COLS,
    evaluate_raw_mri,
    find_best_alpha,
    load_pet_metadata,
    run_loocv as run_b1_loocv,
    run_permutation_test as run_b1_permutation_test,
    run_score_fusion,
)
from c1_pet_radiomics_fusion import (  # noqa: E402
    N_FEATURES_TO_SELECT,
    run_loocv as run_c1_loocv,
    run_permutation_test as run_c1_permutation_test,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PET_XLSX = PROJECT_ROOT / "0ii" / "pet.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "workdir" / "pet"

VALIDATOR_DISPLAY = {
    "oracle": "Oracle",
    "src_acc": "Src-Acc",
    "entropy": "Entropy",
    "infomax": "InfoMax",
    "corr_c": "Corr-C",
    "snd": "SND",
}
# Row order in the output matrix, matching thesis Section 2 "Validators".
VALIDATOR_ORDER = ["oracle", "src_acc", "entropy", "infomax", "corr_c", "snd"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run B1 + C1 LOOCV for every validator to produce the H4 "
            "6 x 2 matrix promised by thesis Section 3.6."
        )
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory with per-validator MRI-score caches.",
    )
    p.add_argument(
        "--radiomics-csv",
        type=Path,
        default=OUTPUT_DIR / "c1_radiomics_features_with_shape_bw0p05.csv",
        help=(
            "Path to the cached C1 radiomics feature CSV.  Default matches "
            "the Zamboglou/Solari-aligned config (shape on, bin width 0.05)."
        ),
    )
    p.add_argument(
        "--pet-xlsx",
        type=Path,
        default=PET_XLSX,
        help="Path to pet.xlsx.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Parent dir for the timestamped run directory.",
    )
    p.add_argument(
        "--validators",
        nargs="+",
        choices=list(VALIDATOR_REGISTRY.keys()),
        default=None,
        help="Subset of validators to evaluate (default: all six).",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=N_FEATURES_TO_SELECT,
        help=(
            "Max features for RFE in C1 (including the MRI score anchor).  "
            f"Default: {N_FEATURES_TO_SELECT}."
        ),
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Logistic regression regularisation (default: 1.0).",
    )
    p.add_argument(
        "--permutation-test",
        type=int,
        default=0,
        metavar="N",
        help="Permutation test with N shuffles per cell (0 = skip).",
    )
    p.add_argument(
        "--include-metadata",
        action="store_true",
        help=(
            "Diagnostic variant: augment C1's feature pool with the three B1 "
            "spreadsheet features (n_psma_lesions, pirads_max, suvmax_late_max) "
            "so RFE can pick from radiomics AND metadata jointly.  Thesis "
            "Sec. 3.6 defines C1 as image radiomics only; this flag breaks "
            "that separation on purpose to test whether PET radiomics carry "
            "signal above metadata.  Run name gets the ``_with_meta`` suffix "
            "so results are not confused with the thesis-canonical C1."
        ),
    )
    p.add_argument(
        "--relaxed-label",
        action="store_true",
        help=(
            "Sensitivity analysis: use 'any PCa' (ISUP >= 1, Gleason >= 6) as "
            "the positive label instead of 'csPCa' (ISUP >= 2, Gleason >= 7). "
            "This increases N+ from 3 to 8 in the PET subset and tests whether "
            "the weak B1/C1 results are driven by statistical power (N+=3) or "
            "lack of signal.  The A4 backbone was trained on ISUP>=2 labels; "
            "only the LOOCV fusion step uses the relaxed label.  Run name gets "
            "the '_relaxed_label' suffix."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Run directory + logging
# ---------------------------------------------------------------------------


def _build_run_dir(base_dir: Path, args: argparse.Namespace) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts, "h4_matrix"]
    if args.include_metadata:
        parts.append("with_meta")
    if args.relaxed_label:
        parts.append("relaxed_label")
    if args.permutation_test > 0:
        parts.append(f"perm{args.permutation_test}")
    run_dir = base_dir / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logging(run_dir: Path) -> None:
    log_file = (
        run_dir / f"h4_matrix_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# ---------------------------------------------------------------------------
# Core evaluation for one validator
# ---------------------------------------------------------------------------


def evaluate_one_validator(
    short_name: str,
    mri_scores: dict[str, float],
    pet_meta: pd.DataFrame,
    radiomics_df: pd.DataFrame | None,
    args: argparse.Namespace,
) -> dict:
    """Run B1 and C1 LOOCV for a single validator.

    Returns a dict with nested results for B1, C1, and score-level fusion.
    """
    # 1. Align patient set: MRI score AND label AND (optional) radiomics.
    common_b1 = sorted(
        pid for pid in pet_meta.index
        if pid in mri_scores and pd.notna(pet_meta.loc[pid, "label"])
    )
    if len(common_b1) < 5:
        raise RuntimeError(
            f"validator={short_name}: only {len(common_b1)} PET patients with "
            "MRI score and label; cannot run B1."
        )

    pet_sub_b1 = pet_meta.loc[common_b1]
    y_b1 = pet_sub_b1["label"].values.astype(int)
    mri_arr_b1 = np.array([mri_scores[pid] for pid in common_b1], dtype=float)

    # 2. B1: mri_pet_meta (main combined model)
    X_b1 = np.column_stack([mri_arr_b1, pet_sub_b1[PET_FEATURE_COLS].values])
    b1_feature_names = ["mri_score"] + PET_FEATURE_COLS
    log.info("[%s] B1 LOOCV (N=%d) ...", short_name, len(common_b1))
    res_b1 = run_b1_loocv(
        X_b1,
        y_b1,
        b1_feature_names,
        common_b1,
        C=args.C,
        classifier="lr",
    )

    # 3. B1: mri_only baseline (for paired delta analysis)
    res_b1_mri_only = run_b1_loocv(
        mri_arr_b1.reshape(-1, 1),
        y_b1,
        ["mri_score"],
        common_b1,
        C=args.C,
        classifier="lr",
    )

    # 4. Score-level fusion (no MRI re-fit)
    res_mri_raw = evaluate_raw_mri(mri_arr_b1, y_b1, common_b1)
    res_pet_meta_only = run_b1_loocv(
        pet_sub_b1[PET_FEATURE_COLS].values.astype(float),
        y_b1,
        PET_FEATURE_COLS,
        common_b1,
        C=args.C,
        classifier="lr",
    )
    pet_only_proba = np.array(
        [res_pet_meta_only["per_patient"][pid]["pred_proba"] for pid in common_b1]
    )
    best_alpha_b1 = find_best_alpha(mri_arr_b1, pet_only_proba, y_b1)
    res_b1_sf = run_score_fusion(
        mri_arr_b1, pet_only_proba, y_b1, common_b1, alpha=best_alpha_b1
    )

    perm_b1 = None
    if args.permutation_test > 0:
        log.info(
            "[%s] B1 permutation test (%d shuffles) ...",
            short_name,
            args.permutation_test,
        )
        perm_b1 = run_b1_permutation_test(
            X_b1,
            y_b1,
            b1_feature_names,
            common_b1,
            observed_auc=res_b1["auc"],
            n_permutations=args.permutation_test,
            C=args.C,
            classifier="lr",
        )

    result = {
        "validator": short_name,
        "n_b1": len(common_b1),
        "b1_mri_only": res_b1_mri_only,
        "b1_mri_pet_meta": res_b1,
        "b1_score_fusion": res_b1_sf,
        "b1_best_alpha": best_alpha_b1,
        "b1_mri_raw": res_mri_raw,
        "b1_permutation": perm_b1,
    }

    # 5. C1 (only if radiomics is available).
    if radiomics_df is not None:
        common_c1 = sorted(
            pid for pid in common_b1 if pid in radiomics_df.index
        )
        if len(common_c1) < 5:
            log.warning(
                "[%s] too few patients with radiomics (%d); skipping C1",
                short_name,
                len(common_c1),
            )
        else:
            y_c1 = np.array(
                [int(pet_meta.loc[pid, "label"]) for pid in common_c1]
            )
            mri_arr_c1 = np.array(
                [mri_scores[pid] for pid in common_c1], dtype=float
            )
            rad_sub = radiomics_df.loc[common_c1]

            # Diagnostic variant: augment C1's pool with the 3 B1 metadata
            # columns so RFE picks from radiomics AND spreadsheet features.
            # Thesis Sec. 3.6 keeps C1 = image radiomics only; this flag is
            # a post-hoc exploration only, reported under a distinct cell
            # label so no one confuses it with the canonical C1.
            if args.include_metadata:
                meta_aug = pet_meta.loc[common_c1, PET_FEATURE_COLS].astype(float)
                rad_sub = pd.concat([rad_sub, meta_aug], axis=1)
                log.info(
                    "[%s] include-metadata: C1 pool now has %d columns "
                    "(%d radiomics + %d B1 metadata)",
                    short_name,
                    len(rad_sub.columns),
                    len(radiomics_df.columns),
                    len(PET_FEATURE_COLS),
                )

            log.info("[%s] C1 LOOCV (N=%d) ...", short_name, len(common_c1))
            res_c1 = run_c1_loocv(
                mri_arr_c1,
                rad_sub,
                y_c1,
                common_c1,
                n_features_to_select=args.n_features,
                C=args.C,
                classifier="lr",
            )

            res_c1_rad_only = run_c1_loocv(
                np.zeros(len(y_c1)),
                rad_sub,
                y_c1,
                common_c1,
                n_features_to_select=args.n_features,
                C=args.C,
                classifier="lr",
            )
            rad_only_proba = np.array(
                [res_c1_rad_only["per_patient"][pid]["pred_proba"] for pid in common_c1]
            )
            best_alpha_c1 = find_best_alpha(mri_arr_c1, rad_only_proba, y_c1)
            res_c1_sf = run_score_fusion(
                mri_arr_c1, rad_only_proba, y_c1, common_c1, alpha=best_alpha_c1
            )

            perm_c1 = None
            if args.permutation_test > 0:
                log.info(
                    "[%s] C1 permutation test (%d shuffles) ...",
                    short_name,
                    args.permutation_test,
                )
                perm_c1 = run_c1_permutation_test(
                    mri_arr_c1,
                    rad_sub,
                    y_c1,
                    common_c1,
                    observed_auc=res_c1["auc"],
                    n_permutations=args.permutation_test,
                    n_features_to_select=args.n_features,
                    C=args.C,
                    classifier="lr",
                )

            result.update(
                {
                    "n_c1": len(common_c1),
                    "c1_mri_pet_radiomics": res_c1,
                    "c1_radiomics_only": res_c1_rad_only,
                    "c1_score_fusion": res_c1_sf,
                    "c1_best_alpha": best_alpha_c1,
                    "c1_permutation": perm_c1,
                }
            )
    return result


# ---------------------------------------------------------------------------
# Table + LaTeX emitter
# ---------------------------------------------------------------------------


def _metric_row(validator: str, fusion: str, res: dict, perm: dict | None) -> dict:
    cm = res.get("confusion_matrix") or [[0, 0], [0, 0]]
    sens_ci = res.get("sensitivity_ci", (np.nan, np.nan))
    spec_ci = res.get("specificity_ci", (np.nan, np.nan))
    auc_ci = res.get("auc_ci", (np.nan, np.nan))
    return {
        "validator": VALIDATOR_DISPLAY.get(validator, validator),
        "fusion": fusion,
        "auc": res.get("auc"),
        "auc_ci_low": auc_ci[0] if isinstance(auc_ci, (list, tuple)) else np.nan,
        "auc_ci_high": auc_ci[1] if isinstance(auc_ci, (list, tuple)) else np.nan,
        "balanced_accuracy": res.get("balanced_accuracy"),
        "sensitivity": res.get("sensitivity"),
        "sensitivity_ci_low": sens_ci[0] if isinstance(sens_ci, (list, tuple)) else np.nan,
        "sensitivity_ci_high": sens_ci[1] if isinstance(sens_ci, (list, tuple)) else np.nan,
        "specificity": res.get("specificity"),
        "specificity_ci_low": spec_ci[0] if isinstance(spec_ci, (list, tuple)) else np.nan,
        "specificity_ci_high": spec_ci[1] if isinstance(spec_ci, (list, tuple)) else np.nan,
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
        "perm_p": perm.get("p_value") if perm else np.nan,
    }


def build_matrix_rows(per_validator: dict) -> list[dict]:
    """Flatten per-validator results into the B1/C1 matrix rows."""
    rows: list[dict] = []
    for validator in VALIDATOR_ORDER:
        if validator not in per_validator:
            continue
        r = per_validator[validator]

        # MRI-only (same across B1/C1 for this validator -> report once as context)
        rows.append(
            _metric_row(validator, "MRI-only", r["b1_mri_only"], None)
        )
        # B1 main: MRI + PET metadata
        rows.append(
            _metric_row(
                validator, "B1: MRI + PET meta", r["b1_mri_pet_meta"], r["b1_permutation"]
            )
        )
        # B1 score-level fusion
        rows.append(
            {
                **_metric_row(validator, "B1: Score fusion", r["b1_score_fusion"], None),
                "alpha": r["b1_best_alpha"],
            }
        )
        if "c1_mri_pet_radiomics" in r:
            rows.append(
                _metric_row(
                    validator,
                    "C1: MRI + PET radiomics",
                    r["c1_mri_pet_radiomics"],
                    r.get("c1_permutation"),
                )
            )
            rows.append(
                {
                    **_metric_row(
                        validator, "C1: Score fusion", r["c1_score_fusion"], None
                    ),
                    "alpha": r.get("c1_best_alpha"),
                }
            )
    return rows


def emit_latex(
    matrix_rows: list[dict], output_path: Path, include_metadata: bool = False
) -> None:
    """Write a ready-to-paste LaTeX table summarising the 6 x 2 matrix."""
    if include_metadata:
        c1_label = "C1$^{\\dagger}$ (MRI+PET radiomics+meta)"
        caption_note = (
            " In this diagnostic variant (\\texttt{--include-metadata}), "
            "the C1 column additionally lets RFE pick from the three B1 "
            "spreadsheet features (\\texttt{n\\_psma\\_lesions}, "
            "\\texttt{pirads\\_max}, \\texttt{suvmax\\_late\\_max}) -- "
            "breaking the strict radiomics/metadata separation of "
            "thesis Sec. 3.6 to test whether image radiomics add "
            "signal above clinician-read metadata."
        )
        label = "tab:h4-matrix-with-meta"
    else:
        c1_label = "C1 (MRI+PET radiomics)"
        caption_note = ""
        label = "tab:h4-matrix"

    lines = [
        r"% Generated by h4_matrix_runner.py -- do not edit by hand",
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{H4 validator $\times$ fusion matrix on the UULM PET/MRI "
        r"subset (LOOCV).  For every unsupervised/oracle validator $V$, an "
        r"A4 checkpoint is selected per fold by $h^{\ast}_{V}=\argmax_{h \in "
        r"\mathcal{H}_{A4}} V(h)$ (Section~\ref{sec:exp-pet}).  Balanced "
        r"accuracy and AUC at threshold 0.5 with 95\,\% bootstrap CI."
        + caption_note
        + r"}",
        rf"  \label{{{label}}}",
        r"  \footnotesize",
        r"  \begin{tabular}{l cc cc cc}",
        r"    \hline",
        r"    & \multicolumn{2}{c}{MRI-only}"
        r" & \multicolumn{2}{c}{B1 (MRI+PET meta)}"
        rf" & \multicolumn{{2}}{{c}}{{{c1_label}}} \\",
        r"    Validator & AUC & BalAcc & AUC & BalAcc & AUC & BalAcc \\",
        r"    \hline",
    ]

    by_v = {v: {r["fusion"]: r for r in matrix_rows if r["validator"] == VALIDATOR_DISPLAY[v]}
            for v in VALIDATOR_ORDER}

    def _fmt(auc: float | None) -> str:
        return "--" if auc is None or np.isnan(auc) else f"{auc:.2f}"

    for validator in VALIDATOR_ORDER:
        cells = by_v.get(validator, {})
        mri = cells.get("MRI-only", {})
        b1 = cells.get("B1: MRI + PET meta", {})
        c1 = cells.get("C1: MRI + PET radiomics", {})
        row = (
            f"    {VALIDATOR_DISPLAY[validator]} "
            f"& {_fmt(mri.get('auc'))} & {_fmt(mri.get('balanced_accuracy'))} "
            f"& {_fmt(b1.get('auc'))} & {_fmt(b1.get('balanced_accuracy'))} "
            f"& {_fmt(c1.get('auc'))} & {_fmt(c1.get('balanced_accuracy'))} \\\\"
        )
        lines.append(row)

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary writer + selection audit
# ---------------------------------------------------------------------------


def _generate_summary(
    run_dir: Path,
    args: argparse.Namespace,
    matrix_rows: list[dict],
    per_validator: dict,
    selection_audit_path: Path,
    matrix_csv: Path,
    tex_path: Path,
) -> None:
    summary_file = run_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("H4: VALIDATOR x FUSION MATRIX -- EXPERIMENT III SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Cache dir: {args.cache_dir}\n")
        f.write(f"Radiomics: {args.radiomics_csv}\n")
        f.write(f"C1 RFE max features: {args.n_features}\n")
        f.write(f"Permutation test: {args.permutation_test} shuffles per cell\n\n")

        f.write("VALIDATOR x FUSION (AUC, BalAcc):\n")
        f.write("-" * 72 + "\n")
        for validator in VALIDATOR_ORDER:
            if validator not in per_validator:
                f.write(f"  {VALIDATOR_DISPLAY[validator]:<10s} (not evaluated)\n")
                continue
            r = per_validator[validator]
            mri = r["b1_mri_only"]
            b1 = r["b1_mri_pet_meta"]
            c1 = r.get("c1_mri_pet_radiomics")
            n_b1 = r["n_b1"]
            n_c1 = r.get("n_c1", "N/A")
            f.write(
                f"  {VALIDATOR_DISPLAY[validator]:<10s}"
                f"  MRI-only: AUC={mri['auc']:.3f} BalAcc={mri['balanced_accuracy']:.3f} (N={n_b1})\n"
            )
            f.write(
                f"  {'':10s}"
                f"  B1      : AUC={b1['auc']:.3f} BalAcc={b1['balanced_accuracy']:.3f}"
            )
            if r["b1_permutation"]:
                f.write(f"  p={r['b1_permutation']['p_value']:.4f}")
            f.write("\n")
            if c1 is not None:
                f.write(
                    f"  {'':10s}"
                    f"  C1      : AUC={c1['auc']:.3f} BalAcc={c1['balanced_accuracy']:.3f} (N={n_c1})"
                )
                if r.get("c1_permutation"):
                    f.write(f"  p={r['c1_permutation']['p_value']:.4f}")
                f.write("\n")
            f.write("\n")

        f.write("Files:\n")
        f.write(f"  matrix CSV       : {matrix_csv}\n")
        f.write(f"  matrix LaTeX     : {tex_path}\n")
        f.write(f"  selection audit  : {selection_audit_path}\n")
        f.write("\n")
        f.write("H4 verdict guidance:\n")
        f.write(
            "  A *consistent* AUC improvement of B1/C1 over MRI-only across "
            "multiple validators is evidence that PET fusion reduces residual "
            "lambda (Thesis Sec. 3.6, H4).  An inconsistent or negative "
            "pattern, given the small N=24-25 cohort, is treated as "
            "inconclusive and motivates future data collection.\n"
        )


def save_selection_audit(cache_dir: Path, run_dir: Path) -> Path:
    """Consolidate every {validator}.selection.json into one CSV for the thesis."""
    rows: list[dict] = []
    for v in VALIDATOR_ORDER:
        try:
            sel = load_selection(v, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        for fs in sel["fold_selections"]:
            rows.append(
                {
                    "validator": VALIDATOR_DISPLAY.get(v, v),
                    "fold": fs["fold"],
                    "algorithm": fs["algorithm"].upper(),
                    "da_weight": fs["da_weight"],
                    "epoch": fs["epoch"],
                    "validator_score": fs["score"],
                    "target_val_bal_acc": fs["target_val_bal_acc"],
                    "n_patients": fs["n_patients"],
                }
            )
    df = pd.DataFrame(rows)
    path = run_dir / "h4_selection_audit.csv"
    df.to_csv(path, index=False)
    log.info("Selection audit saved to %s (%d rows)", path, len(df))
    return path


def save_predictions(per_validator: dict, run_dir: Path) -> Path:
    """One long-format CSV with every (validator x fusion x patient) prediction."""
    rows: list[dict] = []
    for validator, r in per_validator.items():
        disp = VALIDATOR_DISPLAY.get(validator, validator)
        for fusion_key, fusion_label in [
            ("b1_mri_only", "MRI-only"),
            ("b1_mri_pet_meta", "B1: MRI + PET meta"),
            ("b1_score_fusion", "B1: Score fusion"),
            ("c1_mri_pet_radiomics", "C1: MRI + PET radiomics"),
            ("c1_score_fusion", "C1: Score fusion"),
        ]:
            res = r.get(fusion_key)
            if res is None:
                continue
            for pid, pred in res["per_patient"].items():
                rows.append(
                    {
                        "validator": disp,
                        "fusion": fusion_label,
                        "patient_id": pid,
                        "true_label": pred["true_label"],
                        "pred_proba": pred["pred_proba"],
                        "pred_class": pred["pred_class"],
                    }
                )
    df = pd.DataFrame(rows)
    path = run_dir / "h4_predictions.csv"
    df.to_csv(path, index=False)
    log.info("Per-patient predictions saved to %s (%d rows)", path, len(df))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    run_dir = _build_run_dir(args.output_dir, args)
    _setup_logging(run_dir)
    log.info("H4 matrix run dir: %s", run_dir)
    log.info("Arguments: %s", vars(args))

    # --- Load PET metadata + radiomics ---
    pet_meta = load_pet_metadata(args.pet_xlsx)

    if args.relaxed_label:
        # Override label: any PCa (ISUP >= 1, i.e. Gleason >= 6) instead of
        # csPCa (ISUP >= 2).  Re-read 'Biopsie positiv?' from pet.xlsx which
        # is 1 iff Gleason >= 6 and 0 iff Gleason == 0 (biopsy negative).
        raw_df = pd.read_excel(args.pet_xlsx, sheet_name="Auswertung")
        raw_df["PatientenID"] = raw_df["PatientenID"].apply(
            lambda x: str(int(x)).zfill(10) if pd.notna(x) else None
        )
        raw_df = raw_df.set_index("PatientenID")
        relaxed = pd.to_numeric(raw_df["Biopsie positiv?"], errors="coerce")
        for pid in pet_meta.index:
            if pid in relaxed.index and pd.notna(relaxed[pid]):
                pet_meta.loc[pid, "label"] = float(relaxed[pid])
        n_pos = int((pet_meta["label"] == 1).sum())
        n_neg = int((pet_meta["label"] == 0).sum())
        n_nan = int(pet_meta["label"].isna().sum())
        log.info(
            "Relaxed label applied (ISUP>=1): N+=%d, N-=%d, NaN=%d",
            n_pos, n_neg, n_nan,
        )

    radiomics_df: pd.DataFrame | None = None
    if args.radiomics_csv.exists():
        radiomics_df = pd.read_csv(args.radiomics_csv, index_col="patient_id")
        radiomics_df.index = radiomics_df.index.astype(str).str.zfill(10)
        log.info(
            "Loaded radiomics: %d patients x %d features from %s",
            len(radiomics_df),
            len(radiomics_df.columns),
            args.radiomics_csv,
        )
    else:
        log.warning(
            "Radiomics CSV not found: %s.  C1 cells will be empty; "
            "run scripts/runners/pet_sidecar/c1_pet_radiomics_runner.py first "
            "(it writes the cached CSV for subsequent H4 runs).",
            args.radiomics_csv,
        )

    # --- Per-validator evaluation ---
    validators = args.validators or VALIDATOR_ORDER
    per_validator: dict[str, dict] = {}
    for validator in validators:
        try:
            mri_scores = load_scores(validator, cache_dir=args.cache_dir)
        except FileNotFoundError as exc:
            log.error(
                "Validator %s missing score cache: %s. Skipping.", validator, exc
            )
            continue
        mri_scores = {k: float(v) for k, v in mri_scores.items()}
        log.info(
            "[%s] loaded %d patient MRI scores", validator, len(mri_scores)
        )
        try:
            per_validator[validator] = evaluate_one_validator(
                validator, mri_scores, pet_meta, radiomics_df, args
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Validator %s failed: %s", validator, exc)

    if not per_validator:
        log.error("No validators produced results; aborting before save.")
        return 1

    # --- Assemble outputs ---
    matrix_rows = build_matrix_rows(per_validator)
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_csv = run_dir / "h4_matrix.csv"
    matrix_df.to_csv(matrix_csv, index=False)
    log.info("Matrix saved to %s", matrix_csv)

    tex_path = run_dir / "h4_matrix.tex"
    emit_latex(matrix_rows, tex_path, include_metadata=args.include_metadata)
    log.info("LaTeX table saved to %s", tex_path)

    selection_audit_path = save_selection_audit(args.cache_dir, run_dir)
    save_predictions(per_validator, run_dir)

    # Dump raw per-validator JSON for later analysis
    def _default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return str(obj)

    results_path = run_dir / "h4_results.json"
    with open(results_path, "w") as f:
        json.dump(per_validator, f, indent=2, default=_default)
    log.info("Raw per-validator results saved to %s", results_path)

    _generate_summary(
        run_dir,
        args,
        matrix_rows,
        per_validator,
        selection_audit_path,
        matrix_csv,
        tex_path,
    )

    # Print matrix to stdout for quick inspection
    print("\n" + "=" * 72)
    print("H4 VALIDATOR x FUSION MATRIX  (LOOCV, AUC / BalAcc)")
    print("=" * 72)
    print(f"{'Validator':<10s}  {'MRI-only':<18s}  {'B1':<18s}  {'C1':<18s}")
    for validator in VALIDATOR_ORDER:
        if validator not in per_validator:
            continue
        r = per_validator[validator]
        mri = r["b1_mri_only"]
        b1 = r["b1_mri_pet_meta"]
        c1 = r.get("c1_mri_pet_radiomics")
        c1_txt = (
            f"{c1['auc']:.3f}/{c1['balanced_accuracy']:.3f}" if c1 else "   -- / --"
        )
        print(
            f"{VALIDATOR_DISPLAY[validator]:<10s}  "
            f"{mri['auc']:.3f}/{mri['balanced_accuracy']:.3f}     "
            f"{b1['auc']:.3f}/{b1['balanced_accuracy']:.3f}     "
            f"{c1_txt}"
        )
    print("=" * 72)
    print(f"Full outputs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
