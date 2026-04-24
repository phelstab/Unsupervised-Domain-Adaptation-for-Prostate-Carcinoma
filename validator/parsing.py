"""Parsing helpers for run names and result files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .constants import KNOWN_ALGORITHMS, KNOWN_BACKBONES
from .models import AlgorithmId, ExperimentRecord


def tokenize_name(value: str) -> list[str]:
    """Split a name into lowercase alphanumeric tokens."""
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def parse_algorithm_id(
    run_name: str,
    collection_name: str | None = None,
) -> AlgorithmId:
    """Parse algorithm and optional backbone from naming conventions."""
    run_tokens = tokenize_name(run_name)
    collection_tokens = tokenize_name(collection_name or "")
    backbone = None
    for candidate in KNOWN_BACKBONES:
        if candidate.lower() in run_tokens:
            backbone = candidate
            break
    if backbone is None:
        for candidate in KNOWN_BACKBONES:
            if candidate.lower() in collection_tokens:
                backbone = candidate
                break

    for candidate in KNOWN_ALGORITHMS:
        if candidate.lower() in run_tokens:
            return AlgorithmId(algorithm=candidate, backbone=backbone)

    return AlgorithmId(algorithm="UNKNOWN", backbone=backbone)


def detect_regularization(run_name: str) -> bool:
    """Infer whether a run used regularization from its directory name."""
    return "do0." in run_name or "_wd0." in run_name


def load_json_file(path: Path) -> object:
    """Load JSON while replacing literal ``NaN`` tokens."""
    content = path.read_text(encoding="utf-8").replace("NaN", "null")
    return json.loads(content)


def parse_results_file(path: Path) -> list[ExperimentRecord]:
    """Parse all experiment entries from ``results.json``."""
    data = load_json_file(path)
    if not isinstance(data, list):
        return []

    experiments: list[ExperimentRecord] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        config = entry.get("config", {})
        experiments.append(
            ExperimentRecord(
                da_weight=float(config.get("da_weight", 0.0)),
                source_center=str(entry.get("source_center", "RUMC")),
                target_center=str(entry.get("target_center", "")),
                best_epoch=int(entry.get("best_epoch", 0)),
                best_source_val_bal_acc=float(
                    entry.get("best_source_val_bal_acc", 0.0)
                ),
                best_target_val_bal_acc=float(
                    entry.get("best_target_val_bal_acc", 0.0)
                ),
            )
        )
    return experiments


def build_target_pair(source_center: str, target_center: str) -> str:
    """Build the expected target-pair directory name from metadata."""
    source_value = source_center.strip()
    target_value = target_center.strip()
    if not source_value or not target_value:
        return target_value
    return f"{source_value}_to_{target_value}"


def experiment_matches_target_pair(
    experiment: ExperimentRecord,
    target_pair: str,
) -> bool:
    """Return ``True`` when the experiment belongs to a target pair."""
    exact_pair = build_target_pair(
        experiment.source_center,
        experiment.target_center,
    )
    return exact_pair == target_pair or experiment.target_center in target_pair


def parse_epoch_from_name(file_name: str) -> int:
    """Extract the epoch integer from an output file name."""
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse epoch from {file_name}")
    return int(parts[1])


def parse_fold_from_name(name: str | None) -> int | None:
    """Extract the numeric fold index from a fold directory name."""
    if not name:
        return None
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    return int(parts[1])
