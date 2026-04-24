#!/usr/bin/env python3
"""Simple sanity loader for UDA binary setup (public -> UULM).

This script mirrors the experiment data setup used by:
  scripts/runners/uda_bats_new/main.bat

It does NOT train. It only:
1) builds datasets with the same source/target config,
2) applies the same binary label mapping,
3) uses the same split logic from UDATrainer,
4) loads one batch per split as a quick validation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    baseline_src = repo_root / "models" / "MRI" / "baseline" / "src"
    cnn_src = baseline_src / "cnn"
    sys.path.append(str(cnn_src))


setup_imports()

from training_setup.data_generator import ISUPCenterDataset  # noqa: E402
from training_setup.dataset_config import DatasetConfig  # noqa: E402


def build_dataset_configs(args: argparse.Namespace) -> dict[str, DatasetConfig]:
    public_labels = Path(args.public_marksheet)
    public_raw = Path(args.public_raw_dir)
    public_preprocessed = Path(args.public_preprocessed_dir)

    dataset_configs = {
        "RUMC+PCNN+ZGT": DatasetConfig(
            center_alias="RUMC+PCNN+ZGT",
            marksheet_path=public_labels,
            data_dir=public_raw,
            preprocessed_dir=public_preprocessed,
            table_format="csv",
            table_skiprows=0,
            center_column="center",
            center_filter=["RUMC", "PCNN", "ZGT"],
            patient_id_column="patient_id",
            study_id_column="study_id",
            label_column="case_ISUP",
            label_mode="isup",
            sequence_strategy="public",
        )
    }

    dataset_configs[args.uulm_center_name] = DatasetConfig(
        center_alias=args.uulm_center_name,
        marksheet_path=args.uulm_label_file,
        data_dir=args.uulm_raw_dir,
        preprocessed_dir=args.uulm_preprocessed_dir,
        table_format="excel",
        table_skiprows=0,
        center_column=None,
        center_filter=None,
        patient_id_column=args.uulm_patient_id_column,
        study_id_column=args.uulm_study_id_column,
        label_column=args.uulm_label_column,
        label_mode="binary_negative",
        sequence_strategy="uulm",
        temporary_skip_missing_scans=True,
    )

    return dataset_configs


def _extract_binary_labels(dataset_or_subset) -> list[int]:
    """Extract labels without forcing MRI tensor loading."""
    if isinstance(dataset_or_subset, Subset):
        parent_labels = _extract_binary_labels(dataset_or_subset.dataset)
        return [int(parent_labels[i]) for i in dataset_or_subset.indices]

    if hasattr(dataset_or_subset, "isup_labels"):
        return [int(x) for x in dataset_or_subset.isup_labels]

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset_or_subset)!r}")


def _describe_split(name: str, split_dataset, logger: logging.Logger) -> None:
    labels = _extract_binary_labels(split_dataset)
    counts = Counter(labels)
    total = len(labels)

    logger.info(
        "%s: n=%d | benign(0)=%d | csPCa(1)=%d",
        name,
        total,
        counts.get(0, 0),
        counts.get(1, 0),
    )


def _probe_batch(name: str, split_dataset, batch_size: int, logger: logging.Logger) -> None:
    loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    data = batch["data"]
    labels = batch["isup_grade"]

    logger.info(
        "%s batch: data_shape=%s, dtype=%s, labels=%s",
        name,
        tuple(data.shape),
        data.dtype,
        labels.tolist(),
    )


def _split_dataset(dataset, train: float, val: float, test: float, rng: np.random.RandomState) -> dict:
    """Mirror UDATrainer._split_dataset logic exactly."""
    total = len(dataset)
    indices = rng.permutation(total).tolist()

    if total <= 5:
        return {
            "train": Subset(dataset, indices[:max(1, total - 2)]),
            "val": Subset(dataset, [indices[max(0, total - 2)]]),
            "test": Subset(dataset, [indices[total - 1]]),
        }

    train_end = int(train * total)
    val_end = int((train + val) * total)

    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)

    return {
        "train": Subset(dataset, indices[:train_end]),
        "val": Subset(dataset, indices[train_end:val_end]),
        "test": Subset(dataset, indices[val_end:]),
    }


def _split_dataset_two(dataset, train: float, rng: np.random.RandomState) -> dict:
    """Mirror UDATrainer._split_dataset_two logic exactly."""
    total = len(dataset)
    indices = rng.permutation(total).tolist()

    if total <= 3:
        return {
            "train": Subset(dataset, indices[:max(1, total - 1)]),
            "val": Subset(dataset, [indices[total - 1]]),
        }

    train_end = int(train * total)
    train_end = max(1, train_end)

    return {
        "train": Subset(dataset, indices[:train_end]),
        "val": Subset(dataset, indices[train_end:]),
    }


def _create_dataset(center: str, config: DatasetConfig, use_preprocessed: bool) -> ISUPCenterDataset:
    data_dir = str(config.preprocessed_dir if use_preprocessed else config.data_dir)
    marksheet_path = str(config.marksheet_path)
    return ISUPCenterDataset(
        center=center,
        marksheet_path=marksheet_path,
        data_dir=data_dir,
        use_preprocessed=use_preprocessed,
        binary_classification=True,
        dataset_config=config,
    )


def _subset_if_needed(dataset, size: int):
    if size != -1 and size < len(dataset):
        indices = np.random.choice(len(dataset), size, replace=False)
        return Subset(dataset, indices.tolist())
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple loader check for UDA binary setup (RUMC+PCNN+ZGT -> UULM)."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-raw", action="store_true", help="Load from raw MRI files instead of preprocessed tensors")
    parser.add_argument("--source-size", type=int, default=-1)
    parser.add_argument("--target-size", type=int, default=-1)
    parser.add_argument("--two-splits-source", action="store_true")
    parser.add_argument("--two-splits-target", action="store_true")

    parser.add_argument("--public-marksheet", type=str, default="input/picai_labels/clinical_information/marksheet.csv")
    parser.add_argument("--public-raw-dir", type=str, default="input/images")
    parser.add_argument("--public-preprocessed-dir", type=str, default="input/images_preprocessed")

    parser.add_argument("--uulm-center-name", type=str, default="UULM")
    parser.add_argument("--uulm-label-file", type=str, default="0ii/data.xlsx")
    parser.add_argument("--uulm-raw-dir", type=str, default="0ii/files/mri_data")
    parser.add_argument("--uulm-preprocessed-dir", type=str, default="0ii/files/registered")
    parser.add_argument("--uulm-label-column", type=str, default="Biopsie negativ?")
    parser.add_argument("--uulm-patient-id-column", type=str, default="Patientennr")
    parser.add_argument("--uulm-study-id-column", type=str, default="Auftragsnr")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("load_uda_binary_data_simple")

    torch.manual_seed(42)
    np.random.seed(42)

    if not Path(args.uulm_label_file).exists():
        raise FileNotFoundError(f"UULM label file not found: {args.uulm_label_file}")

    dataset_configs = build_dataset_configs(args)
    source_center = "RUMC+PCNN+ZGT"
    target_center = args.uulm_center_name
    use_preprocessed = not args.use_raw

    source_dataset = _create_dataset(source_center, dataset_configs[source_center], use_preprocessed)
    target_dataset = _create_dataset(target_center, dataset_configs[target_center], use_preprocessed)

    source_dataset = _subset_if_needed(source_dataset, args.source_size)
    target_dataset = _subset_if_needed(target_dataset, args.target_size)

    split_rng = np.random.RandomState(42)
    if args.two_splits_source:
        source_splits = _split_dataset_two(source_dataset, train=0.85, rng=split_rng)
    else:
        source_splits = _split_dataset(source_dataset, train=0.7, val=0.15, test=0.15, rng=split_rng)

    if args.two_splits_target:
        target_splits = _split_dataset_two(target_dataset, train=0.85, rng=split_rng)
    else:
        target_splits = _split_dataset(target_dataset, train=0.7, val=0.15, test=0.15, rng=split_rng)

    splits = {
        "source_train": source_splits["train"],
        "source_val": source_splits["val"],
        "source_test": source_splits.get("test"),
        "target_train": target_splits["train"],
        "target_val": target_splits["val"],
        "target_test": target_splits.get("test"),
    }

    logger.info(
        "Using setup: %s -> %s (binary UDA, use_preprocessed=%s)",
        source_center,
        target_center,
        use_preprocessed,
    )
    _describe_split("source_train", splits["source_train"], logger)
    _describe_split("source_val", splits["source_val"], logger)
    if splits["source_test"] is not None:
        _describe_split("source_test", splits["source_test"], logger)

    _describe_split("target_train", splits["target_train"], logger)
    _describe_split("target_val", splits["target_val"], logger)
    if splits["target_test"] is not None:
        _describe_split("target_test", splits["target_test"], logger)

    _probe_batch("source_train", splits["source_train"], args.batch_size, logger)
    _probe_batch("target_train", splits["target_train"], args.batch_size, logger)
    _probe_batch("source_val", splits["source_val"], args.batch_size, logger)
    _probe_batch("target_val", splits["target_val"], args.batch_size, logger)

    logger.info("Loader check finished successfully.")


if __name__ == "__main__":
    main()
