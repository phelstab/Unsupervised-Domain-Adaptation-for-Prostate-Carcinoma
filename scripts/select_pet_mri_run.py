"""Select the MRI UDA run used as input for PET late-fusion experiments.

Policy implemented here:

1. Restrict to the A4-style run family used by the PET work:
   - target pair: RUMC+PCNN+ZGT -> UULM
   - model variant: prostate_clinical
   - prostate prior enabled
   - clinical branch enabled
   - regularized runs only
   - checkpoint_validator must be source_val
2. Among those runs, choose the run whose *best source-val-selected experiment*
   has the highest saved target balanced accuracy in ``results.json``.
3. Return both the run directory and the selected DA weight.

B/C still use ``source_val`` for per-fold epoch selection inside the chosen
run. The run-family comparison itself is retrospective because it uses saved
target metrics from completed MRI runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


RUNS_DIR = (
    Path(__file__).resolve().parent.parent
    / "runs"
    / "UULM_binary_class_3CV_RESNET10_GLAND_META"
)
TARGET_PAIR = "RUMCplusPCNNplusZGT-to-UULM"
MODEL_VARIANT = "prostate_clinical"


@dataclass
class SelectedRun:
    run_dir: Path
    da_weight: str
    algorithm: str
    target_bal_acc: float
    target_auc: float


def _parse_algorithm(run_name: str) -> str:
    """Extract algorithm token from the run directory name."""
    parts = run_name.lower().split("_")
    known = ["mcc", "bnm", "hybrid", "mmd", "dann", "mcd", "coral", "entropy", "daarda"]
    for algo in known:
        if algo in parts:
            return algo.upper()
    return "UNKNOWN"


def _load_results(results_json: Path) -> list[dict]:
    with open(results_json, encoding="utf-8") as f:
        content = f.read().replace("NaN", "null")
    data = json.loads(content)
    if not isinstance(data, list):
        return []
    return data


def _is_valid_pet_backbone_run(run_dir: Path, exp: dict) -> bool:
    run_name = run_dir.name
    if TARGET_PAIR not in run_name:
        return False
    if "source_val" not in run_name:
        return False
    if MODEL_VARIANT not in run_name:
        return False
    if "whole_gland" not in run_name:
        return False
    if "do0." not in run_name or "wd0." not in run_name:
        return False

    if exp.get("checkpoint_validator") != "source_val":
        return False
    if exp.get("model_variant") != MODEL_VARIANT:
        return False
    if not exp.get("use_prostate_prior", False):
        return False
    if not exp.get("use_clinical", False):
        return False
    if exp.get("target_center") != "UULM":
        return False
    if exp.get("source_center") != "RUMC+PCNN+ZGT":
        return False
    return True


def select_best_pet_mri_run(runs_dir: Path = RUNS_DIR) -> SelectedRun:
    """Select the MRI run/DA weight for PET late-fusion experiments."""
    candidates: list[SelectedRun] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_json = run_dir / "results.json"
        if not results_json.exists():
            continue

        try:
            experiments = _load_results(results_json)
        except Exception:
            continue

        for exp in experiments:
            if not _is_valid_pet_backbone_run(run_dir, exp):
                continue

            config = exp.get("config", {})
            da_weight = config.get("da_weight")
            if da_weight is None:
                continue

            candidates.append(
                SelectedRun(
                    run_dir=run_dir,
                    da_weight=str(da_weight),
                    algorithm=_parse_algorithm(run_dir.name),
                    target_bal_acc=float(
                        exp.get("final_target_test_balanced_accuracy", -1.0)
                    ),
                    target_auc=float(exp.get("final_target_test_auc", -1.0)),
                )
            )

    if not candidates:
        raise FileNotFoundError(
            f"No eligible PET MRI backbone runs found in {runs_dir}"
        )

    # Primary sort: target balanced accuracy. Secondary: target AUC.
    best = max(candidates, key=lambda c: (c.target_bal_acc, c.target_auc))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select MRI run used by PET late-fusion experiments"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing completed MRI UDA runs",
    )
    args = parser.parse_args()

    best = select_best_pet_mri_run(args.runs_dir)
    print(f"RUN_DIR={best.run_dir}")
    print(f"DA_WEIGHT={best.da_weight}")
    print(f"ALGORITHM={best.algorithm}")
    print(f"TARGET_BAL_ACC={best.target_bal_acc:.4f}")
    print(f"TARGET_AUC={best.target_auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
