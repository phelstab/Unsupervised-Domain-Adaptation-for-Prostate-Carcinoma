"""Core data models for validator table generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmId:
    """Algorithm identifier with an optional backbone."""

    algorithm: str
    backbone: str | None = None

    @property
    def canonical_label(self) -> str:
        """Return a stable label for grouping and logging."""
        if self.backbone:
            return f"{self.backbone}_{self.algorithm}"
        return self.algorithm

    def display_label(self, include_backbone: bool) -> str:
        """Return a human-friendly label for markdown tables."""
        if include_backbone and self.backbone:
            return self.canonical_label
        return self.algorithm


@dataclass(frozen=True)
class ExperimentRecord:
    """Experiment metadata extracted from ``results.json``."""

    da_weight: float
    source_center: str
    target_center: str
    best_epoch: int = 0
    best_source_val_bal_acc: float = 0.0
    best_target_val_bal_acc: float = 0.0


@dataclass(frozen=True)
class RunInfo:
    """Metadata about a single training run directory."""

    run_dir: Path
    run_name: str
    algorithm_id: AlgorithmId
    has_regularization: bool
    experiments: list[ExperimentRecord]


@dataclass(frozen=True)
class ResolvedRun:
    """Resolved run variant that should be loaded into the report."""

    reg_setting: str
    algorithm_id: AlgorithmId
    target_pair: str
    da_dir_name: str
    run_dir: Path
    run_info: RunInfo
    source_path: Path
    is_complete: bool
    is_legacy_layout: bool = False


@dataclass(frozen=True)
class CheckpointData:
    """In-memory representation of one saved checkpoint output file."""

    run_name: str
    algorithm_id: AlgorithmId
    da_weight: float
    epoch: int
    target_pair: str
    source_probs: Any
    source_labels: Any
    target_features: Any
    target_probs: Any
    target_labels: Any
    target_predictions: Any
    source_val_bal_acc: float
    target_val_bal_acc: float
    fold: int | None = None


@dataclass(frozen=True)
class ValidatorResult:
    """Result produced when a validator selects a checkpoint."""

    checkpoint: CheckpointData
    score: float
    target_bal_acc: float


AlgorithmPools = dict[AlgorithmId, list[CheckpointData]]
TargetPools = dict[str, AlgorithmPools]
RegularizationPools = dict[str, TargetPools]


@dataclass
class CollectionData:
    """Loaded checkpoints and metadata for one collection directory."""

    name: str
    path: Path
    checkpoints_by_reg: RegularizationPools = field(
        default_factory=lambda: {"baseline": {}, "regularized": {}}
    )
    include_backbone: bool = False
    resolved_runs: list[ResolvedRun] = field(default_factory=list)
