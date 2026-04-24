"""Markdown report rendering for validator table outputs."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

from .constants import KNOWN_ALGORITHMS, REGULARIZATION_DESCRIPTIONS
from .constants import REGULARIZATION_TITLES
from .models import AlgorithmId, AlgorithmPools, CheckpointData, CollectionData
from .validators import Validator


def render_collection_report(
    collection: CollectionData,
    validators: list[Validator],
) -> str:
    """Render a markdown report for a single collection."""
    lines = [
        "# UDA Validator x Algorithm Results",
        "",
        f"Generated from: `{collection.path}`",
        "",
    ]
    lines.extend(render_legend())

    for reg_setting in ("baseline", "regularized"):
        reg_data = collection.checkpoints_by_reg.get(reg_setting, {})
        if not reg_data:
            continue
        lines.extend(
            render_regularization_section(
                reg_setting,
                reg_data,
                validators,
                collection.include_backbone,
            )
        )

    lines.extend(render_summary(collection))
    return "\n".join(lines).strip() + "\n"


def render_combined_report(
    collections: list[CollectionData],
    validators: list[Validator],
) -> str:
    """Render multiple collections into one markdown document."""
    lines = ["# UDA Validator x Algorithm Results", ""]
    for collection in collections:
        lines.append(f"## {collection.name}")
        lines.append("")
        report_body = render_collection_report(collection, validators)
        report_lines = report_body.splitlines()
        for line in report_lines[1:]:
            lines.append(line)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_legend() -> list[str]:
    """Render the shared legend that explains the table values."""
    return [
        "## Legend",
        "",
        "Each cell reports target balanced accuracy (%) of the checkpoint",
        "selected by a validator from all epochs and DA weights.",
        "",
        "| Validator | Selection criterion | Uses target labels? |",
        "| :--- | :--- | :---: |",
        "| **Oracle** | Best target balanced accuracy | Yes |",
        "| Src-Acc | Best source validation balanced accuracy | No |",
        "| Entropy | Lowest target prediction entropy | No |",
        "| InfoMax | Highest mutual information score | No |",
        "| Corr-C | Best feature/prediction consistency | No |",
        "| SND | Highest soft neighborhood density | No |",
        "",
    ]


def render_regularization_section(
    reg_setting: str,
    reg_data: dict[str, AlgorithmPools],
    validators: list[Validator],
    include_backbone: bool,
) -> list[str]:
    """Render one regularization section with all target pairs."""
    title = REGULARIZATION_TITLES[reg_setting]
    description = REGULARIZATION_DESCRIPTIONS[reg_setting]
    lines = ["---", "", f"## {title}", "", description, ""]
    for target_pair in sorted(reg_data):
        lines.extend(
            render_target_pair_table(
                target_pair,
                reg_data[target_pair],
                validators,
                include_backbone,
            )
        )
    return lines


def render_target_pair_table(
    target_pair: str,
    algorithm_pools: AlgorithmPools,
    validators: list[Validator],
    include_backbone: bool,
) -> list[str]:
    """Render one markdown table for a target-pair block."""
    lines = [f"### {target_pair.replace('_to_', ' -> ')}", ""]
    algorithms = sort_algorithm_ids(list(algorithm_pools.keys()))
    if not algorithms:
        lines.extend(["No data available.", ""])
        return lines

    header = ["Validator"]
    header.extend(
        algorithm.display_label(include_backbone)
        for algorithm in algorithms
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append(build_separator_row(len(algorithms)))

    oracle = validators[0]
    lines.append(
        render_validator_row(
            oracle,
            algorithms,
            algorithm_pools,
            bold_label=True,
            bold_values=True,
        )
    )
    for validator in validators[1:]:
        lines.append(
            render_validator_row(
                validator,
                algorithms,
                algorithm_pools,
            )
        )
    lines.append("")
    return lines


def build_separator_row(algorithm_count: int) -> str:
    """Build the markdown separator row for a table."""
    separator_cells = [":---"]
    separator_cells.extend(":---:" for _ in range(algorithm_count))
    return "| " + " | ".join(separator_cells) + " |"


def render_validator_row(
    validator: Validator,
    algorithms: list[AlgorithmId],
    algorithm_pools: AlgorithmPools,
    bold_label: bool = False,
    bold_values: bool = False,
) -> str:
    """Render a single validator row."""
    label = validator.name
    if bold_label:
        label = f"**{label}**"

    cells = [label]
    for algorithm in algorithms:
        pool = algorithm_pools.get(algorithm, [])
        mean, std, fold_count = select_per_fold(validator, pool)
        value = format_score(mean, std, fold_count)
        if bold_values and value != "-":
            value = f"**{value}**"
        cells.append(value)
    return "| " + " | ".join(cells) + " |"


def select_per_fold(
    validator: Validator,
    pool: list[CheckpointData],
) -> tuple[float, float, int]:
    """Select the best checkpoint independently for each fold."""
    numpy = import_module("numpy")
    fold_groups: dict[int, list[CheckpointData]] = {}
    for checkpoint in pool:
        fold = checkpoint.fold if checkpoint.fold is not None else 0
        fold_groups.setdefault(fold, []).append(checkpoint)

    accuracies: list[float] = []
    for fold_pool in fold_groups.values():
        result = validator.select_best(fold_pool)
        if result is not None:
            accuracies.append(result.target_bal_acc)

    if not accuracies:
        return 0.0, 0.0, 0
    if len(accuracies) == 1:
        return float(accuracies[0]), 0.0, 1
    return (
        float(numpy.mean(accuracies)),
        float(numpy.std(accuracies)),
        len(accuracies),
    )


def format_score(mean: float, std: float, fold_count: int) -> str:
    """Format a score for a markdown cell."""
    if fold_count == 0:
        return "-"
    if fold_count == 1:
        return f"{mean:.1f}"
    return f"{mean:.1f} +/- {std:.1f}"


def render_summary(collection: CollectionData) -> list[str]:
    """Render checkpoint-count summary tables."""
    lines = ["---", "", "## Summary", ""]
    for reg_setting in ("baseline", "regularized"):
        reg_data = collection.checkpoints_by_reg.get(reg_setting, {})
        if not reg_data:
            continue
        lines.append(f"### {REGULARIZATION_TITLES[reg_setting]}")
        lines.append("")
        for target_pair in sorted(reg_data):
            lines.append(f"**{target_pair}**")
            lines.append("")
            lines.append("| Algorithm | Checkpoints | DA weights |")
            lines.append("| :--- | :---: | :--- |")
            algorithm_list = sort_algorithm_ids(
                list(reg_data[target_pair].keys())
            )
            for algorithm in algorithm_list:
                checkpoints = reg_data[target_pair][algorithm]
                weights = sorted(
                    {checkpoint.da_weight for checkpoint in checkpoints}
                )
                weight_text = ", ".join(str(weight) for weight in weights)
                label = algorithm.display_label(collection.include_backbone)
                lines.append(
                    f"| {label} | {len(checkpoints)} | {weight_text} |"
                )
            lines.append("")
    return lines


def sort_algorithm_ids(
    algorithms: Iterable[AlgorithmId],
) -> list[AlgorithmId]:
    """Sort algorithms by backbone and a known algorithm display order."""
    order = {name: index for index, name in enumerate(KNOWN_ALGORITHMS)}

    def sort_key(algorithm: AlgorithmId) -> tuple[int, int, str, str]:
        backbone = algorithm.backbone or ""
        backbone_rank = backbone_sort_rank(backbone)
        algorithm_rank = order.get(algorithm.algorithm, len(order))
        return (backbone_rank, algorithm_rank, backbone, algorithm.algorithm)

    return sorted(algorithms, key=sort_key)


def backbone_sort_rank(backbone: str) -> int:
    """Create a stable numeric rank for common ResNet backbones."""
    if not backbone:
        return -1
    digits = "".join(
        character for character in backbone if character.isdigit()
    )
    if digits:
        return int(digits)
    return 999
