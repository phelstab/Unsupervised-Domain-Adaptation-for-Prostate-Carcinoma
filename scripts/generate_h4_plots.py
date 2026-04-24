"""Generate thesis figures for the H4 PET-enriched extensions (B1, C1).

Produces four PDFs commonly used in PSMA-PET radiomics / logistic regression
studies on small cohorts:

    fig_h4_roc.pdf            - per-validator ROC curves for
                                MRI-only, B1 (MRI+PET meta), and
                                C1 (MRI+PET radiomics).  Clinically standard
                                visualisation from Papp 2021, Solari 2022,
                                Zamboglou 2019.

    fig_h4_feature_frequency.pdf
                              - horizontal bar chart of how often each
                                PyRadiomics feature was retained by
                                nested-LOOCV RFE in the canonical C1 run.
                                Mirrors Solari 2022 Fig. 2 and Papp 2021.

    fig_h4_budget_sensitivity.pdf
                              - AUC of the combined radiomics + metadata
                                model as a function of the RFE feature
                                budget (k = 4, 6, 10).  Shows that k = 6
                                is the empirical optimum on N = 24.

    fig_h4_patient_ranking.pdf
                              - waterfall / sorted bar chart of per-patient
                                C1 predicted probabilities with true-label
                                markers overlaid.  Exposes the failure
                                mode visible in the data: the three
                                csPCa-positive patients are not ranked
                                in the top three.

Usage
-----
    .venv-cnn\\Scripts\\python.exe scripts\\generate_h4_plots.py \\
        --h4-run workdir\\pet\\YYYYMMDD_HHMMSS_h4_matrix \\
        --diagnostic-txt workdir\\pet\\diagnose_pet_features_YYYYMMDD_HHMMSS.txt \\
        --output Thesis\\figures\\pet

Inputs the H4 run directory (requires h4_results.json + h4_predictions.csv +
h4_selection_audit.csv) and a diagnostic report to parse the budget-
sensitivity numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

VALIDATOR_ORDER = ["oracle", "src_acc", "entropy", "infomax", "corr_c", "snd"]
VALIDATOR_DISPLAY = {
    "oracle": "Oracle",
    "src_acc": "Src-Acc",
    "entropy": "Entropy",
    "infomax": "InfoMax",
    "corr_c": "Corr-C",
    "snd": "SND",
}
# Colour-blind friendly qualitative palette (Wong 2011, Okabe-Ito subset)
COLOURS = {
    "mri": "#555555",
    "b1": "#0072B2",
    "c1": "#D55E00",
    "c1_meta": "#009E73",
    "oracle_ref": "#56B4E9",
}

LEGEND_BOX_KW = dict(
    frameon=True,
    facecolor="white",
    edgecolor="0.85",
    framealpha=0.88,
)


def _pretty_feature_name(name: str) -> str:
    name = re.sub(r"^original_", "", name)
    parts = name.split("_", 1)
    if len(parts) == 2:
        category, feature = parts
        feature = re.sub(r"([a-z])([A-Z])", r"\1 \2", feature)
        return f"{category}  {feature}"
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_h4(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(run_dir / "h4_results.json") as f:
        results = json.load(f)
    predictions = pd.read_csv(run_dir / "h4_predictions.csv")
    audit = pd.read_csv(run_dir / "h4_selection_audit.csv")
    return results, predictions, audit


def _roc_points(y_true: np.ndarray, scores: np.ndarray):
    """Compute ROC (fpr, tpr, auc) without sklearn import drama."""
    order = np.argsort(-scores, kind="stable")
    y = y_true[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(1, y.sum())
    fpr = fp / max(1, (1 - y).sum())
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Figure 1: per-validator ROC curves  (MRI-only, B1, C1)
# ---------------------------------------------------------------------------


def fig_roc(
    results: dict,
    predictions: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(6.5, 4.8), sharex=True, sharey=True,
    )
    axes_flat = axes.ravel()

    # (legend_short, result_key, csv_fusion, colour, linestyle, linewidth)
    fusion_specs = [
        ("MRI-only", "b1_mri_only", "MRI-only", COLOURS["mri"], "-", 1.0),
        ("B1", "b1_mri_pet_meta", "B1: MRI + PET meta", COLOURS["b1"], "-", 1.2),
        ("C1", "c1_mri_pet_radiomics", "C1: MRI + PET radiomics", COLOURS["c1"], "-", 1.2),
    ]

    for i, v in enumerate(VALIDATOR_ORDER):
        ax = axes_flat[i]
        disp = VALIDATOR_DISPLAY[v]

        for legend_short, result_key, csv_fusion, colour, ls, lw in fusion_specs:
            res = results.get(v, {}).get(result_key)
            if res is None:
                continue

            roc_data = res.get("roc_curve")
            if roc_data is not None:
                fpr = np.array(roc_data["fpr"])
                tpr = np.array(roc_data["tpr"])
                auc = res.get("auc", float(np.trapezoid(tpr, fpr)))
            else:
                sub = predictions[
                    (predictions["validator"] == disp)
                    & (predictions["fusion"] == csv_fusion)
                ]
                if sub.empty:
                    continue
                y = sub["true_label"].values.astype(int)
                p = sub["pred_proba"].values.astype(float)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                fpr, tpr, auc = _roc_points(y, p)

            ax.step(
                fpr,
                tpr,
                where="post",
                color=colour,
                linewidth=lw,
                linestyle=ls,
                label=f"{legend_short}: {auc:.2f}",
            )

        ax.plot([0, 1], [0, 1], "--", color="0.80", linewidth=0.6, zorder=0)
        ax.set_title(disp, fontweight="semibold", pad=4)
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, 0.5, 1.0]))
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.0, 0.5, 1.0]))
        ax.legend(
            loc="lower right",
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.3,
            fontsize=6.2,
            **LEGEND_BOX_KW,
        )

    for ax in axes_flat[3:]:
        ax.set_xlabel("False-positive rate")
    for ax in (axes_flat[0], axes_flat[3]):
        ax.set_ylabel("True-positive rate")

    fig.tight_layout(h_pad=1.8, w_pad=0.8)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ROC panel: 2\u00d73 grid, 3 curves each")


# ---------------------------------------------------------------------------
# Figure 2: C1 feature selection frequency (horizontal bars)
# ---------------------------------------------------------------------------


def fig_feature_frequency(results: dict, out_path: Path, n_top: int = 12) -> None:
    freq_rows: list[tuple[str, str, int]] = []
    for v in VALIDATOR_ORDER:
        res = results.get(v, {}).get("c1_mri_pet_radiomics")
        if res is None:
            continue
        freq = res.get("feature_selection_frequency", {})
        for feat, count in freq.items():
            freq_rows.append((feat, v, int(count)))

    if not freq_rows:
        return
    df = pd.DataFrame(freq_rows, columns=["feature", "validator", "count"])
    df = df[df["feature"] != "mri_score"]

    # Collapse across validators (counts are identical from shared RFE)
    mean_count = df.groupby("feature")["count"].mean().sort_values(ascending=False)
    top_features = mean_count.head(n_top)

    pretty = [_pretty_feature_name(f) for f in top_features.index]
    counts = top_features.values.astype(int)
    n_folds = 24

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(counts))
    bars = ax.barh(
        y, counts, color=COLOURS["c1"], edgecolor="white",
        linewidth=0.4, height=0.65,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pretty, fontsize=7.5)
    ax.invert_yaxis()

    ax.set_xlabel(f"Folds selected (of {n_folds})")
    ax.set_title(
        "C1 feature-selection frequency across LOOCV folds",
        fontweight="semibold",
        pad=6,
    )

    ax.axvline(
        n_folds / 2, color="0.65", linewidth=0.6,
        linestyle="--", alpha=0.7, zorder=0,
    )
    ax.text(
        n_folds / 2 + 0.3, len(counts) - 0.3,
        "50% of folds",
        fontsize=6.5, color="0.45", style="italic", va="top",
    )

    ax.set_xlim(0, n_folds)
    ax.set_xticks([0, 6, 12, 18, 24])

    for yi, val in zip(y, counts):
        ax.text(val + 0.3, yi, str(val), va="center", fontsize=7, color="0.3")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Feature frequency: top {len(counts)} features")


# ---------------------------------------------------------------------------
# Figure 3: RFE budget sensitivity (from diagnostic report)
# ---------------------------------------------------------------------------


def _parse_diagnostic_budget(txt_path: Path) -> list[tuple[str, int, float]]:
    """Scrape the 'Config ... AUC ...' table from diagnose_pet_features output."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    rows: list[tuple[str, int, float]] = []
    pattern = re.compile(
        r"\((?P<tag>[a-e])\)\s+pool\s*=\s*(?P<pool>[^,]+),\s*budget\s*=\s*(?P<k>\d+)"
        r"\s{2,}(?P<auc>\d+\.\d+)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            rows.append(
                (
                    m.group("pool").strip(),
                    int(m.group("k")),
                    float(m.group("auc")),
                )
            )
    return rows


def fig_budget_sensitivity(diag_txt: Path, out_path: Path) -> None:
    rows = _parse_diagnostic_budget(diag_txt)
    if not rows:
        print(f"WARNING: could not parse budget rows from {diag_txt}")
        return

    by_pool: dict[str, list[tuple[int, float]]] = {}
    for pool, k, auc in rows:
        by_pool.setdefault(pool, []).append((k, auc))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    style_map = {
        "radiomics+metadata (110 cols)": (
            "o", "-", COLOURS["c1_meta"], 7,
            "radiomics + metadata (110 cols)",
            COLOURS["c1_meta"], COLOURS["c1_meta"],
        ),
        "radiomics only (107 cols)": (
            "s", "--", COLOURS["c1"], 6,
            "radiomics only (107 cols)",
            COLOURS["c1"], COLOURS["c1"],
        ),
        "metadata only (3 cols)": (
            "^", "none", COLOURS["b1"], 8,
            "metadata only (3 cols)",
            "white", COLOURS["b1"],
        ),
    }

    # Collect all (k, auc) pairs at each x position for annotation offset
    all_points_at_k: dict[int, list[tuple[float, str]]] = {}
    for pool, pts in by_pool.items():
        for k_val, auc_val in pts:
            colour = style_map.get(pool, ("o", "-", "black", 6, pool, "black", "black"))[2]
            all_points_at_k.setdefault(k_val, []).append((auc_val, colour))

    for pool, pts in by_pool.items():
        pts = sorted(pts)
        ks = [p[0] for p in pts]
        aucs = [p[1] for p in pts]
        marker, ls, colour, ms, label, mfc, mec = style_map.get(
            pool, ("o", "-", "black", 6, pool, "black", "black")
        )
        plot_kw = dict(
            marker=marker,
            markersize=ms,
            color=colour,
            label=label,
            zorder=5,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            markeredgewidth=0.8,
        )
        if len(pts) == 1:
            ax.plot(ks, aucs, linestyle="none", **plot_kw)
        else:
            ax.plot(ks, aucs, linestyle=ls, linewidth=1.2, **plot_kw)

    # Annotate AUC values with smart offset to avoid overlap
    for k_val, entries in all_points_at_k.items():
        entries_sorted = sorted(entries, key=lambda e: e[0])
        # Group entries that are within 0.005 AUC of each other
        placed: list[tuple[float, float]] = []  # (x_off, y_off) already used
        for j, (auc_val, colour) in enumerate(entries_sorted):
            # Count how many prior entries are at ~same AUC
            same_count = sum(
                1 for prev_auc, _ in entries_sorted[:j]
                if abs(prev_auc - auc_val) < 0.005
            )
            if same_count > 0:
                # Offset to the right and below for colliding annotations
                xytext = (22, -10)
                ha = "left"
                va = "top"
            else:
                # Check if another point is close (within 0.03 AUC)
                close = any(
                    abs(other_auc - auc_val) < 0.03 and other_auc != auc_val
                    for other_auc, _ in entries_sorted
                )
                if close and j == 0:
                    xytext = (0, -12)
                    ha = "center"
                    va = "top"
                else:
                    xytext = (0, 8)
                    ha = "center"
                    va = "bottom"
            ax.annotate(
                f"{auc_val:.2f}",
                (k_val, auc_val),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                va=va,
                fontsize=7,
                color=colour,
                fontweight="semibold",
            )

    ax.axhline(0.5, color="0.75", linewidth=0.6, linestyle="--", zorder=0)
    ax.text(
        10.3, 0.505, "chance", fontsize=6.5, color="0.55",
        va="bottom", style="italic",
    )

    ax.set_xlabel("RFE feature budget $k$ (incl. MRI anchor)")
    ax.set_ylabel("LOOCV AUC")
    ax.set_title(
        "Budget sensitivity of C1 feature selection",
        fontweight="semibold",
        pad=6,
    )
    all_ks = sorted({p[1] for p in rows})
    ax.set_xticks(all_ks)
    ax.set_xlim(min(all_ks) - 0.8, max(all_ks) + 0.8)
    ymax = min(1.0, max(auc for _, _, auc in rows) + 0.10)
    ax.set_ylim(0.35, ymax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        labelspacing=0.5,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Budget sensitivity: {len(rows)} configs parsed")


# ---------------------------------------------------------------------------
# Figure 4: per-patient C1 ranking (failure-mode waterfall)
# ---------------------------------------------------------------------------


def fig_patient_ranking(
    predictions: pd.DataFrame, out_path: Path, validator: str = "Src-Acc"
) -> None:
    sub = predictions[
        (predictions["validator"] == validator)
        & (predictions["fusion"] == "C1: MRI + PET radiomics")
    ].copy()
    if sub.empty:
        return
    sub = sub.sort_values("pred_proba", ascending=False).reset_index(drop=True)
    n = len(sub)
    x = np.arange(n)

    colour_pos = COLOURS["c1"]
    colour_neg = "#C0C0C0"
    colours = [colour_pos if y == 1 else colour_neg for y in sub["true_label"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(
        x, sub["pred_proba"], color=colours, width=0.85,
        edgecolor="0.40", linewidth=0.3,
    )
    ax.axhline(0.5, color="0.55", linewidth=0.6, linestyle="--", zorder=0)
    ax.text(
        n - 0.8, 0.515, "$\\tau = 0.5$",
        fontsize=6.5, color="0.45", ha="right", va="bottom", style="italic",
    )

    # Annotate csPCa-positive patients with offset-based staggering
    pos_indices = sub.index[sub["true_label"] == 1].tolist()
    # Assign offset points: alternate left/right and stagger vertically
    offsets_pts: dict[int, tuple[float, float]] = {}
    for j, idx in enumerate(pos_indices):
        # Check proximity to other positive indices
        nearby = any(
            abs(idx - other) <= 2 and other != idx
            for other in pos_indices
        )
        if nearby:
            if j == 0:
                offsets_pts[idx] = (0, 10)
            elif j == 1:
                offsets_pts[idx] = (-18, 22)
            else:
                offsets_pts[idx] = (18, 34)
        else:
            offsets_pts[idx] = (0, 10)

    for idx in pos_indices:
        row = sub.iloc[idx]
        pid = f"{int(row['patient_id']):07d}"
        dx, dy = offsets_pts[idx]
        ax.annotate(
            pid,
            xy=(idx, row["pred_proba"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=colour_pos,
            fontweight="semibold",
            arrowprops=dict(
                arrowstyle="-",
                color=colour_pos,
                linewidth=0.5,
                shrinkA=0,
                shrinkB=1,
            ),
        )

    ax.set_xticks([])
    ax.set_xlabel(
        f"Patients sorted by C1 predicted probability "
        f"({validator} validator, $N = {n}$)"
    )
    ax.set_ylabel("Predicted $P(\\mathrm{csPCa})$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_title(
        f"Per-patient C1 predicted probabilities ({validator})",
        fontweight="semibold",
        pad=6,
    )

    ax.legend(
        handles=[
            Patch(facecolor=colour_pos, edgecolor="0.40", linewidth=0.5,
                  label="csPCa-positive (ISUP $\\geq$ 2)"),
            Patch(facecolor=colour_neg, edgecolor="0.40", linewidth=0.5,
                  label="csPCa-negative"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.2,
        handletextpad=0.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Patient ranking: {n} patients, {sub['true_label'].sum()} positive")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h4-run", type=Path, required=True)
    p.add_argument("--diagnostic-txt", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT.parent / "UDA" / "Thesis" / "figures" / "pet",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results, predictions, _audit = load_h4(args.h4_run)

    print("Generating H4 thesis figures ...")

    fig_roc(results, predictions, args.output / "fig_h4_roc.pdf")
    print(f"-> {args.output / 'fig_h4_roc.pdf'}")

    fig_feature_frequency(results, args.output / "fig_h4_feature_frequency.pdf")
    print(f"-> {args.output / 'fig_h4_feature_frequency.pdf'}")

    fig_budget_sensitivity(
        args.diagnostic_txt, args.output / "fig_h4_budget_sensitivity.pdf"
    )
    print(f"-> {args.output / 'fig_h4_budget_sensitivity.pdf'}")

    fig_patient_ranking(
        predictions, args.output / "fig_h4_patient_ranking.pdf"
    )
    print(f"-> {args.output / 'fig_h4_patient_ranking.pdf'}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
