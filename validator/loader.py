"""Load checkpoints from repo folders into structured collections."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .discovery import discover_run_directories
from .models import CheckpointData, CollectionData, ExperimentRecord
from .models import RegularizationPools, ResolvedRun, RunInfo
from .parsing import detect_regularization, experiment_matches_target_pair
from .parsing import parse_algorithm_id, parse_epoch_from_name
from .parsing import parse_fold_from_name
from .parsing import parse_results_file


@dataclass(frozen=True)
class OutputLocation:
    """Concrete output directory with an optional fold index."""

    outputs_dir: Path
    fold: int | None = None


@dataclass(frozen=True)
class LayoutInspection:
    """Resolved output layout for a DA-weight directory."""

    output_locations: list[OutputLocation]
    is_complete: bool


def parse_run_info(
    run_dir: Path,
    collection_name: str,
) -> RunInfo | None:
    """Parse one run directory into structured metadata."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None

    experiments = parse_results_file(results_path)
    if not experiments:
        return None

    return RunInfo(
        run_dir=run_dir,
        run_name=run_dir.name,
        algorithm_id=parse_algorithm_id(run_dir.name, collection_name),
        has_regularization=detect_regularization(run_dir.name),
        experiments=experiments,
    )


def load_collection(
    collection_path: Path,
    run_filter: str | None = None,
    verbose: bool = True,
) -> CollectionData:
    """Load all resolved checkpoints from a collection directory."""
    run_infos: list[RunInfo] = []
    for run_dir in discover_run_directories(collection_path):
        if run_filter and run_filter not in run_dir.name:
            continue
        run_info = parse_run_info(run_dir, collection_path.name)
        if run_info is not None:
            run_infos.append(run_info)

    include_backbone = should_include_backbone(run_infos)
    resolved_runs = resolve_runs(run_infos)

    if verbose and resolved_runs:
        print(f"Resolved runs for {collection_path.name}:")
        for resolved_run in resolved_runs:
            status = "COMPLETE" if resolved_run.is_complete else "INCOMPLETE"
            label = resolved_run.algorithm_id.canonical_label
            print(
                "  "
                f"{label} {resolved_run.target_pair}/"
                f"{resolved_run.da_dir_name} "
                f"-> {resolved_run.run_dir.name} [{status}]"
            )

    checkpoints = initialize_checkpoint_store()
    for resolved_run in resolved_runs:
        loaded_count = load_resolved_run(resolved_run, checkpoints)
        if verbose:
            label = resolved_run.algorithm_id.canonical_label
            print(
                "  "
                f"Loaded {loaded_count} outputs for {label} "
                f"{resolved_run.target_pair}/{resolved_run.da_dir_name}"
            )

    return CollectionData(
        name=collection_path.name,
        path=collection_path,
        checkpoints_by_reg=checkpoints,
        include_backbone=include_backbone,
        resolved_runs=resolved_runs,
    )


def initialize_checkpoint_store() -> RegularizationPools:
    """Create the nested result structure used during loading."""
    return {"baseline": {}, "regularized": {}}


def should_include_backbone(run_infos: list[RunInfo]) -> bool:
    """Include backbone prefixes only when a collection mixes backbones."""
    backbones = {
        run_info.algorithm_id.backbone
        for run_info in run_infos
        if run_info.algorithm_id.backbone
    }
    return len(backbones) > 1


def resolve_runs(run_infos: list[RunInfo]) -> list[ResolvedRun]:
    """Resolve reruns so each DA-weight slot uses a single directory."""
    resolved: dict[tuple[str, str, str, str], ResolvedRun] = {}

    for run_info in sorted(run_infos, key=lambda item: item.run_dir.name):
        reg_setting = (
            "regularized" if run_info.has_regularization else "baseline"
        )
        for target_dir in iter_target_directories(run_info.run_dir):
            da_directories = [
                path
                for path in sorted(target_dir.iterdir())
                if path.is_dir() and path.name.startswith("da_")
            ]
            if da_directories:
                for da_dir in da_directories:
                    inspection = inspect_da_directory(da_dir)
                    if inspection is None:
                        continue
                    resolved_run = ResolvedRun(
                        reg_setting=reg_setting,
                        algorithm_id=run_info.algorithm_id,
                        target_pair=target_dir.name,
                        da_dir_name=da_dir.name,
                        run_dir=run_info.run_dir,
                        run_info=run_info,
                        source_path=da_dir,
                        is_complete=inspection.is_complete,
                    )
                    key = resolved_key(resolved_run)
                    current = resolved.get(key)
                    if current is None or prefer_resolved_run(
                        current,
                        resolved_run,
                    ):
                        resolved[key] = resolved_run
                continue

            legacy_candidate = build_legacy_resolved_run(
                run_info,
                target_dir,
                reg_setting,
            )
            if legacy_candidate is None:
                continue
            key = resolved_key(legacy_candidate)
            current = resolved.get(key)
            if current is None or prefer_resolved_run(
                current,
                legacy_candidate,
            ):
                resolved[key] = legacy_candidate

    return sorted(
        resolved.values(),
        key=lambda item: (
            item.reg_setting,
            item.target_pair,
            item.algorithm_id.canonical_label,
            item.da_dir_name,
        ),
    )


def resolved_key(resolved_run: ResolvedRun) -> tuple[str, str, str, str]:
    """Build the dictionary key used for rerun resolution."""
    return (
        resolved_run.reg_setting,
        resolved_run.algorithm_id.canonical_label,
        resolved_run.target_pair,
        resolved_run.da_dir_name,
    )


def prefer_resolved_run(
    existing: ResolvedRun,
    candidate: ResolvedRun,
) -> bool:
    """Prefer complete runs, then the newest run name."""
    if candidate.is_complete and not existing.is_complete:
        return True
    if candidate.is_complete != existing.is_complete:
        return False
    return candidate.run_dir.name > existing.run_dir.name


def iter_target_directories(run_dir: Path) -> list[Path]:
    """Return target-pair directories inside a run."""
    return [
        path
        for path in sorted(run_dir.iterdir())
        if path.is_dir() and "_to_" in path.name
    ]


def inspect_da_directory(da_dir: Path) -> LayoutInspection | None:
    """Inspect a DA-weight directory for flat or CV output layouts."""
    fold_dirs = [
        path
        for path in sorted(da_dir.iterdir())
        if path.is_dir() and path.name.startswith("fold_")
    ]
    if fold_dirs:
        output_locations: list[OutputLocation] = []
        complete = True
        for fold_dir in fold_dirs:
            outputs_dir = fold_dir / "outputs"
            has_outputs = (
                outputs_dir.exists() and has_output_files(outputs_dir)
            )
            if not has_outputs:
                complete = False
                continue
            output_locations.append(
                OutputLocation(
                    outputs_dir=outputs_dir,
                    fold=parse_fold_from_name(fold_dir.name),
                )
            )
            checkpoints_dir = fold_dir / "checkpoints"
            final_checkpoint = checkpoints_dir / "epoch_0099.pt"
            if not final_checkpoint.exists():
                complete = False
        if not output_locations:
            return None
        return LayoutInspection(
            output_locations=output_locations,
            is_complete=complete,
        )

    outputs_dir = da_dir / "outputs"
    if not outputs_dir.exists() or not has_output_files(outputs_dir):
        return None
    checkpoints_dir = da_dir / "checkpoints"
    is_complete = (checkpoints_dir / "epoch_0099.pt").exists()
    return LayoutInspection(
        output_locations=[OutputLocation(outputs_dir=outputs_dir)],
        is_complete=is_complete,
    )


def build_legacy_resolved_run(
    run_info: RunInfo,
    target_dir: Path,
    reg_setting: str,
) -> ResolvedRun | None:
    """Build a resolved entry for the legacy flat target layout."""
    outputs_dir = target_dir / "outputs"
    if not outputs_dir.exists() or not has_output_files(outputs_dir):
        return None

    matching = [
        experiment
        for experiment in run_info.experiments
        if experiment_matches_target_pair(experiment, target_dir.name)
    ]
    if not matching:
        return None

    da_dir_name = f"da_{matching[-1].da_weight}"
    return ResolvedRun(
        reg_setting=reg_setting,
        algorithm_id=run_info.algorithm_id,
        target_pair=target_dir.name,
        da_dir_name=da_dir_name,
        run_dir=run_info.run_dir,
        run_info=run_info,
        source_path=target_dir,
        is_complete=False,
        is_legacy_layout=True,
    )


def has_output_files(outputs_dir: Path) -> bool:
    """Return ``True`` when an outputs directory has epoch files."""
    return any(outputs_dir.glob("epoch_*_outputs.pt"))


def load_resolved_run(
    resolved_run: ResolvedRun,
    checkpoints: RegularizationPools,
) -> int:
    """Load all checkpoint outputs for a resolved run selection."""
    da_weight = parse_da_weight(resolved_run.da_dir_name)
    experiment = select_experiment(
        resolved_run.run_info.experiments,
        resolved_run.target_pair,
        da_weight,
    )
    if resolved_run.is_legacy_layout:
        output_locations = [
            OutputLocation(outputs_dir=resolved_run.source_path / "outputs")
        ]
    else:
        inspection = inspect_da_directory(resolved_run.source_path)
        if inspection is None:
            return 0
        output_locations = inspection.output_locations

    loaded = 0
    for output_location in output_locations:
        for output_file in sorted(
            output_location.outputs_dir.glob("epoch_*_outputs.pt")
        ):
            checkpoint = load_checkpoint_output(
                output_file=output_file,
                run_info=resolved_run.run_info,
                target_pair=resolved_run.target_pair,
                da_weight=da_weight,
                experiment=experiment,
                fold=output_location.fold,
            )
            if checkpoint is None:
                continue
            reg_bucket = checkpoints[resolved_run.reg_setting]
            target_bucket = reg_bucket.setdefault(resolved_run.target_pair, {})
            pool = target_bucket.setdefault(resolved_run.algorithm_id, [])
            pool.append(checkpoint)
            loaded += 1
    return loaded


def parse_da_weight(da_dir_name: str) -> float:
    """Parse a DA-weight directory name like ``da_0.1``."""
    return float(da_dir_name.split("_", 1)[1])


def select_experiment(
    experiments: list[ExperimentRecord],
    target_pair: str,
    da_weight: float,
) -> ExperimentRecord:
    """Find the best matching experiment metadata for a DA-weight slot."""
    matches = [
        experiment
        for experiment in experiments
        if experiment_matches_target_pair(experiment, target_pair)
        and abs(experiment.da_weight - da_weight) < 1e-6
    ]
    if matches:
        return matches[0]

    fallback = [
        experiment
        for experiment in experiments
        if experiment_matches_target_pair(experiment, target_pair)
    ]
    if fallback:
        return fallback[-1]

    return ExperimentRecord(
        da_weight=da_weight,
        source_center="",
        target_center="",
    )


def load_checkpoint_output(
    output_file: Path,
    run_info: RunInfo,
    target_pair: str,
    da_weight: float,
    experiment: ExperimentRecord,
    fold: int | None,
) -> CheckpointData | None:
    """Load one ``epoch_*_outputs.pt`` file into ``CheckpointData``."""
    numpy = import_module("numpy")
    torch = import_module("torch")

    data = torch.load(output_file, map_location="cpu", weights_only=False)
    source_val = data.get("source_val", {})
    target_val = data.get("target_val", {})

    target_features = numpy.asarray(target_val.get("features", []))
    target_probs = numpy.asarray(target_val.get("probabilities", []))
    target_labels = numpy.asarray(target_val.get("labels", []))
    target_predictions = numpy.asarray(target_val.get("predictions", []))

    if target_features.size == 0 or target_probs.size == 0:
        return None
    if target_labels.size == 0:
        return None
    if target_predictions.size == 0:
        target_predictions = target_probs.argmax(axis=1)

    return CheckpointData(
        run_name=run_info.run_name,
        algorithm_id=run_info.algorithm_id,
        da_weight=da_weight,
        epoch=parse_epoch_from_name(output_file.name),
        target_pair=target_pair,
        source_probs=numpy.asarray(source_val.get("probabilities", [])),
        source_labels=numpy.asarray(source_val.get("labels", [])),
        target_features=target_features,
        target_probs=target_probs,
        target_labels=target_labels,
        target_predictions=target_predictions,
        source_val_bal_acc=experiment.best_source_val_bal_acc,
        target_val_bal_acc=experiment.best_target_val_bal_acc,
        fold=fold,
    )
