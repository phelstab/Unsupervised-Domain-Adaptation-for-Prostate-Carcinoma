#!/usr/bin/env python3
"""Generate the PET SUV boxplot figure for the thesis (Table/Figure in Appendix A).

Reads  : 0ii/pet.xlsx  (UKU PET sub-cohort spreadsheet)
Writes : <THESIS_DIR>/figures/pet/fig_pet_suv_boxplot.pdf

Usage:
    python scripts/plot_pet_suv_boxplot.py
    python scripts/plot_pet_suv_boxplot.py --outdir /custom/output/path
"""

import argparse
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_DIR = REPO_ROOT.parent / "Thesis"
PET_XLSX = REPO_ROOT / "0ii" / "pet.xlsx"

# Column offsets (0-indexed) for SUV values per lesion slot (1-7)
LESION_SUV_OFFSETS = [
    (34, 35, 36, 37),    # Lesion 1: SUVmean_early, SUVmax_early, SUVmean_late, SUVmax_late
    (47, 48, 49, 50),    # Lesion 2
    (60, 61, 62, 63),    # Lesion 3
    (73, 74, 75, 76),    # Lesion 4
    (86, 87, 88, 89),    # Lesion 5
    (99, 100, 101, 102), # Lesion 6
    (112, 113, 114, 115),# Lesion 7
]
ACTIVITY_COL = 6  # Aktivität (MBq)


def parse_float(v):
    """Try to convert a cell value to float; return None on failure."""
    if v is None or v == "" or v == "-":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_pet_data(xlsx_path: Path):
    """Extract per-patient activity and per-lesion SUV values from pet.xlsx."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Auswertung"]

    patients = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue

        activity = parse_float(row[ACTIVITY_COL])
        suv_mean_early, suv_max_early = [], []
        suv_mean_late, suv_max_late = [], []

        for me, mxe, ml, mxl in LESION_SUV_OFFSETS:
            for ci, lst in [
                (me, suv_mean_early), (mxe, suv_max_early),
                (ml, suv_mean_late),  (mxl, suv_max_late),
            ]:
                if ci < len(row):
                    v = parse_float(row[ci])
                    if v is not None:
                        lst.append(v)

        patients.append({
            "activity": activity,
            "suv_mean_early": suv_mean_early,
            "suv_max_early": suv_max_early,
            "suv_mean_late": suv_mean_late,
            "suv_max_late": suv_max_late,
        })
    return patients


def make_figure(patients, outpath: Path):
    """Create the two-panel boxplot and save as PDF."""
    suv_me  = [v for p in patients for v in p["suv_mean_early"]]
    suv_mxe = [v for p in patients for v in p["suv_max_early"]]
    suv_ml  = [v for p in patients for v in p["suv_mean_late"]]
    suv_mxl = [v for p in patients for v in p["suv_max_late"]]
    activities = [p["activity"] for p in patients if p["activity"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1, 2]})

    # --- Left panel: injected activity ---
    axes[0].boxplot(
        [activities], widths=0.5, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor="#d4e6f1", edgecolor="#2c3e50"),
        medianprops=dict(color="#c0392b", linewidth=1.5),
        whiskerprops=dict(color="#2c3e50"),
        capprops=dict(color="#2c3e50"),
        flierprops=dict(marker="o", markerfacecolor="#2c3e50", markersize=4),
    )
    axes[0].set_ylabel("MBq", fontsize=9)
    axes[0].set_xticklabels(["Activity"], fontsize=8)
    axes[0].set_title("Injected\nactivity", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis="y", labelsize=8)

    # --- Right panel: lesion-level SUV (y-axis clipped, outliers annotated) ---
    data = [suv_me, suv_mxe, suv_ml, suv_mxl]
    labels = [
        "SUV$_{mean}$\nearly", "SUV$_{max}$\nearly",
        "SUV$_{mean}$\nlate",  "SUV$_{max}$\nlate",
    ]
    colors = ["#d5f5e3", "#abebc6", "#d6eaf8", "#aed6f1"]

    bp = axes[1].boxplot(
        data, widths=0.6, patch_artist=True, showfliers=False,
        medianprops=dict(color="#c0392b", linewidth=1.5),
        whiskerprops=dict(color="#2c3e50"),
        capprops=dict(color="#2c3e50"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#2c3e50")

    # Jittered strip plot
    rng = np.random.default_rng(42)
    for i, d in enumerate(data, start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        axes[1].scatter(
            np.full(len(d), i) + jitter, d, s=12, alpha=0.5,
            color="#2c3e50", edgecolors="none", zorder=3,
        )

    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("SUV", fontsize=9)
    axes[1].set_title(
        "$^{68}$Ga-PSMA SUV per lesion", fontsize=9, fontweight="bold"
    )
    axes[1].tick_params(axis="both", labelsize=8)
    axes[1].set_ylim(0, 35)

    # Annotate clipped outliers
    axes[1].annotate(
        "102.1 ↑", xy=(4, 34), fontsize=7, ha="center",
        color="#c0392b", fontweight="bold",
    )
    axes[1].annotate(
        "63.1 ↑", xy=(3, 34), fontsize=7, ha="center",
        color="#c0392b", fontweight="bold",
    )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=THESIS_DIR / "figures" / "pet",
        help="Output directory for the PDF (default: Thesis/figures/pet/)",
    )
    args = parser.parse_args()

    patients = load_pet_data(PET_XLSX)
    print(f"Loaded {len(patients)} patients from {PET_XLSX}")
    make_figure(patients, args.outdir / "fig_pet_suv_boxplot.pdf")


if __name__ == "__main__":
    main()
