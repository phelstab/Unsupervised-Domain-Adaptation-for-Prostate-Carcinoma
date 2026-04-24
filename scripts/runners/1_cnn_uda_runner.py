#!/usr/bin/env python3

# UV Virtual Environment: .venv-cnn
# Setup: python -m uv venv .venv-cnn
# Install PyTorch: python -m uv pip install --python .venv-cnn --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# Install dependencies: python -m uv pip install --python .venv-cnn numpy matplotlib pandas scikit-learn scikit-image nibabel SimpleITK tqdm tensorboard Pillow picai_prep
# Run quick test: .venv-cnn\Scripts\python.exe scripts\runners\1_cnn_uda_runner.py --quick
# Run full test: .venv-cnn\Scripts\python.exe scripts\runners\1_cnn_uda_runner.py

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append(
    str(Path(__file__).parent.parent.parent / "models" / "MRI" / "baseline" / "src")
)

from cnn.training_setup.augmentations import AugmentationConfig
from cnn.training_setup.data_generator import ISUPCenterDataset
from cnn.training_setup.dataset_config import DatasetConfig, MetadataSourceConfig
from cnn.training_setup.preprocess_data import (
    check_preprocessed_data_exists_from_config,
    preprocess_and_save_dataset_from_config,
)
from cnn.training_setup.uda_trainer import UDATrainer


MODEL_VARIANTS = {
    "baseline",
    "prostate_prior",
    "clinical",
    "prostate_clinical",
    "pet_fusion",
}
CLINICAL_DEFAULT_COLUMN_MAP = {
    "psa": "psa",
    "psad": "psad",
    "prostate_volume": "prostate_volume",
    "age": "patient_age",
}

PROSTATE_PRIOR_SOURCE_PRESETS = {
    "bosma22b": Path(
        "input/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b"
    ),
    "guerbet23": Path(
        "input/picai_labels/anatomical_delineations/whole_gland/AI/Guerbet23"
    ),
    "heviai23": Path(
        "input/picai_labels/anatomical_delineations/zonal_pz_tz/AI/HeviAI23"
    ),
    "yuan23": Path("input/picai_labels/anatomical_delineations/zonal_pz_tz/AI/Yuan23"),
}


def parse_center_pairs(center_pair_args):
    pairs = []
    for pair in center_pair_args:
        if "_to_" not in pair:
            raise ValueError(
                f"Invalid center pair '{pair}'. Expected format SOURCE_to_TARGET"
            )
        source, target = pair.split("_to_", 1)
        pairs.append((source, target))
    return pairs


def parse_feature_column_map(raw_values: list[str] | None) -> dict[str, str]:
    """Parse CLI mappings like ['psa=PSA', 'age=AgeYears'] into a dict."""
    mapping: dict[str, str] = {}
    if not raw_values:
        return mapping

    for item in raw_values:
        if "=" not in item:
            raise ValueError(
                f"Invalid clinical map entry '{item}'. Use FEATURE=COLUMN format, e.g. psa=PSA."
            )
        feature_name, column_name = item.split("=", 1)
        feature_name = feature_name.strip()
        column_name = column_name.strip()
        if not feature_name or not column_name:
            raise ValueError(
                f"Invalid clinical map entry '{item}'. Feature and column must be non-empty."
            )
        mapping[feature_name] = column_name
    return mapping


def resolve_feature_columns(
    feature_names: list[str], column_map: dict[str, str]
) -> list[str]:
    """Resolve marksheet columns required by selected clinical features."""
    columns = []
    for feature_name in feature_names:
        column_name = column_map.get(
            feature_name,
            CLINICAL_DEFAULT_COLUMN_MAP.get(feature_name, feature_name),
        )
        columns.append(column_name)
    return list(dict.fromkeys(columns))


def resolve_prostate_prior_source_dir(source_name: str, custom_dir: str | None) -> Path:
    """Resolve source-side prior directory from preset name or custom path."""
    source_name = source_name.lower()
    if source_name == "custom":
        if not custom_dir:
            raise ValueError(
                "--prostate-prior-source custom requires --prostate-prior-source-dir"
            )
        return Path(custom_dir)

    if custom_dir:
        raise ValueError(
            "--prostate-prior-source-dir is only valid when --prostate-prior-source custom is selected"
        )

    if source_name not in PROSTATE_PRIOR_SOURCE_PRESETS:
        raise ValueError(
            f"Unsupported prostate prior source '{source_name}'. "
            f"Choose from {sorted(PROSTATE_PRIOR_SOURCE_PRESETS.keys())} or 'custom'."
        )
    return PROSTATE_PRIOR_SOURCE_PRESETS[source_name]


def build_dataset_configs(
    args,
    public_extra_columns: list[str] | None = None,
    uulm_extra_columns: list[str] | None = None,
):
    public_labels = Path(args.public_marksheet)
    public_raw = Path(args.public_raw_dir)
    public_preprocessed = Path(args.public_preprocessed_dir)

    dataset_configs = {
        "RUMC": DatasetConfig(
            center_alias="RUMC",
            marksheet_path=public_labels,
            data_dir=public_raw,
            preprocessed_dir=public_preprocessed,
            table_format="csv",
            table_skiprows=0,
            center_column="center",
            center_filter=["RUMC"],
            patient_id_column="patient_id",
            study_id_column="study_id",
            label_column="case_ISUP",
            label_mode="isup",
            sequence_strategy="public",
            extra_columns=public_extra_columns,
        ),
        "PCNN": DatasetConfig(
            center_alias="PCNN",
            marksheet_path=public_labels,
            data_dir=public_raw,
            preprocessed_dir=public_preprocessed,
            table_format="csv",
            table_skiprows=0,
            center_column="center",
            center_filter=["PCNN"],
            patient_id_column="patient_id",
            study_id_column="study_id",
            label_column="case_ISUP",
            label_mode="isup",
            sequence_strategy="public",
            extra_columns=public_extra_columns,
        ),
        "ZGT": DatasetConfig(
            center_alias="ZGT",
            marksheet_path=public_labels,
            data_dir=public_raw,
            preprocessed_dir=public_preprocessed,
            table_format="csv",
            table_skiprows=0,
            center_column="center",
            center_filter=["ZGT"],
            patient_id_column="patient_id",
            study_id_column="study_id",
            label_column="case_ISUP",
            label_mode="isup",
            sequence_strategy="public",
            extra_columns=public_extra_columns,
        ),
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
            extra_columns=public_extra_columns,
        ),
    }

    if args.uulm_use_dual_metadata:
        metadata_sources = [
            MetadataSourceConfig(
                source_name="man",
                marksheet_path=args.uulm_label_file,
                table_format="excel",
                patient_id_column=args.uulm_patient_id_column,
                study_id_column=args.uulm_study_id_column,
                label_column=args.uulm_label_column,
                label_mode="gleason",
                patient_id_pad_width=10,
                extra_column_map={
                    "patient_age": "Alter zum Zeitpunkt der Biopsie",
                    "psa": ["uPSA (ng/ml)", "iPSA (ng/ml)"],
                    "prostate_volume": "Prostatavolumen (ml)",
                    "metadata_exam_date": "Untersuchungsdatum",
                    "metadata_device": "Gerät",
                    "gleason_primary": ["Gleason-Score", "Gleason-Score gesamt"],
                },
                static_columns={
                    "pet_available": 0,
                },
            ),
            MetadataSourceConfig(
                source_name="pet",
                marksheet_path=args.uulm_pet_label_file,
                table_format="excel",
                sheet_name=args.uulm_pet_sheet_name,
                patient_id_column=args.uulm_pet_patient_id_column,
                study_id_column=args.uulm_pet_study_id_column,
                label_column=args.uulm_pet_label_column,
                label_mode="gleason",
                patient_id_pad_width=10,
                extra_column_map={
                    "patient_age": "Alter zum Zeitpunkt der Untersuchung",
                    "psa": ["uPSA", "iPSA"],
                    "prostate_volume": "Prostatavolumen (ml)",
                    "metadata_exam_date": "Untersuchungsdatum",
                    "metadata_device": "Gerät",
                    "gleason_primary": ["Gleason", "Gleason-Score"],
                    "pet_num_targets": "Anzahl Targets",
                    "pet_target_positive": "Target positiv",
                    "pet_psma_positive_lesions": "Anzahl PSMA Positive Läsionen ",
                },
                static_columns={
                    "pet_available": 1,
                },
            ),
        ]

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
            label_mode="gleason",
            sequence_strategy="uulm",
            temporary_skip_missing_scans=True,
            extra_columns=uulm_extra_columns,
            metadata_sources=metadata_sources,
        )
    else:
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
            label_mode="gleason",
            sequence_strategy="uulm",
            temporary_skip_missing_scans=True,
            extra_columns=uulm_extra_columns,
        )

    return dataset_configs


def get_default_center_pairs(args):
    if args.dataset_profile == "public_to_uulm":
        return [("RUMC+PCNN+ZGT", args.uulm_center_name)]
    return [("RUMC", "PCNN"), ("RUMC", "ZGT")]


def build_center_pairs_suffix(center_pairs):
    """Build a compact folder-safe suffix for source->target setup."""
    tags = []
    for source, target in center_pairs:
        src = source.replace("+", "plus")
        tgt = target.replace("+", "plus")
        tags.append(f"{src}-to-{tgt}")

    merged = "__".join(tags)
    if len(merged) > 80:
        merged = f"{merged[:77]}..."
    return f"_pairs-{merged}"


def build_variant_suffix(
    model_variant: str,
    use_prostate_prior: bool,
    prostate_prior_type: str,
    prostate_prior_target_mode: str,
    use_clinical: bool,
    clinical_fusion: str,
) -> str:
    """Build run-directory suffix for optional architecture branches."""
    parts = []
    if model_variant != "baseline":
        parts.append(f"mv{model_variant}")
    if use_prostate_prior:
        parts.append(f"pp{prostate_prior_type}")
        if prostate_prior_target_mode == "pseudo":
            parts.append("tgtpseudo")
    if use_clinical:
        parts.append(f"clin{clinical_fusion}")
    return "_" + "_".join(parts) if parts else ""


class UDARunner:
    def __init__(
        self,
        dataset_configs,
        center_pairs,
        base_workdir="workdir/uda",
        gpu_id=0,
        binary_classification=False,
        da_method="coral",
        lower_bound=False,
        upper_bound=False,
        dropout_rate=0.0,
        use_batchnorm=False,
        checkpoint_validator="source_val",
        checkpoint_save_interval=10,
        use_early_stopping=True,
        da_warmup_epochs=0,
        use_class_weights=False,
        backbone="resnet10",
        plot_oracle=False,
        weight_decay=0.0,
        lr_scheduler="none",
        lr_step_size=30,
        lr_gamma=0.1,
        two_splits_source=False,
        two_splits_target=False,
        daarda_divergence="js_beta",
        daarda_relax=1.0,
        daarda_grad_penalty=0.0,
        augmentation_config=None,
        target_cv_folds=1,
        model_variant="baseline",
        use_prostate_prior=False,
        prostate_prior_type="whole_gland",
        prostate_prior_source_dir: str | None = None,
        prostate_prior_target_mode="none",
        prostate_prior_target_dir: str | None = None,
        prostate_prior_cache_dir: str | None = None,
        prostate_prior_strength=1.0,
        prostate_prior_conf_thresh=0.5,
        use_clinical=False,
        clinical_features: list[str] | None = None,
        clinical_fusion="early",
        clinical_impute="median",
        clinical_missing_indicators=False,
        source_clinical_column_map: dict[str, str] | None = None,
        target_clinical_column_map: dict[str, str] | None = None,
    ):
        self.base_workdir = Path(base_workdir)
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.results = []
        self.binary_classification = binary_classification
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.backbone = backbone
        self.plot_oracle = plot_oracle
        self.augmentation_config = augmentation_config
        self.dataset_configs = dataset_configs
        self.center_pairs = center_pairs

        self.model_variant = model_variant
        self.use_prostate_prior = use_prostate_prior
        self.prostate_prior_type = prostate_prior_type
        self.prostate_prior_source_dir = prostate_prior_source_dir
        self.prostate_prior_target_mode = prostate_prior_target_mode
        self.prostate_prior_target_dir = prostate_prior_target_dir
        self.prostate_prior_cache_dir = prostate_prior_cache_dir
        self.prostate_prior_strength = prostate_prior_strength
        self.prostate_prior_conf_thresh = prostate_prior_conf_thresh

        self.use_clinical = use_clinical
        self.clinical_features = clinical_features or [
            "psa",
            "psad",
            "prostate_volume",
            "age",
        ]
        self.clinical_fusion = clinical_fusion
        self.clinical_impute = clinical_impute
        self.clinical_missing_indicators = clinical_missing_indicators
        self.source_clinical_column_map = source_clinical_column_map or {}
        self.target_clinical_column_map = target_clinical_column_map or {}

        if lower_bound and upper_bound:
            raise ValueError(
                "Cannot use both --lower-bound and --upper-bound. Choose one."
            )

        if self.upper_bound:
            da_method = "none"
            print("UPPER BOUND MODE: Training supervised on source+target combined")
        elif self.lower_bound:
            if da_method != "none":
                print(
                    f"WARNING: Lower bound mode enabled. Overriding da_method '{da_method}' to 'none'."
                )
            da_method = "none"

        self.da_method = da_method
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.checkpoint_validator = checkpoint_validator
        self.checkpoint_save_interval = checkpoint_save_interval
        self.use_early_stopping = use_early_stopping
        self.da_warmup_epochs = da_warmup_epochs
        self.use_class_weights = use_class_weights
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.two_splits_source = two_splits_source
        self.two_splits_target = two_splits_target
        self.daarda_divergence = daarda_divergence
        self.daarda_relax = daarda_relax
        self.daarda_grad_penalty = daarda_grad_penalty
        self.target_cv_folds = target_cv_folds

        mode_suffix = "_binary" if binary_classification else "_6class"
        backbone_suffix = f"_{backbone}" if backbone != "resnet10" else ""

        if self.upper_bound:
            da_suffix = "_upper_bound"
        elif self.lower_bound:
            da_suffix = "_lower_bound"
        else:
            da_suffix = f"_{da_method}"

        validator_suffix = f"_{checkpoint_validator}"
        variant_suffix = build_variant_suffix(
            model_variant=self.model_variant,
            use_prostate_prior=self.use_prostate_prior,
            prostate_prior_type=self.prostate_prior_type,
            prostate_prior_target_mode=self.prostate_prior_target_mode,
            use_clinical=self.use_clinical,
            clinical_fusion=self.clinical_fusion,
        )

        reg_parts = []
        if dropout_rate > 0:
            reg_parts.append(f"do{dropout_rate}")
        if use_batchnorm:
            reg_parts.append("bn")
        if weight_decay > 0:
            reg_parts.append(f"wd{weight_decay}")
        if lr_scheduler != "none":
            reg_parts.append(f"lr{lr_scheduler}")
        if da_method == "daarda":
            reg_parts.append(f"arda{daarda_relax}")
            reg_parts.append(daarda_divergence)
            if daarda_grad_penalty > 0:
                reg_parts.append(f"gp{daarda_grad_penalty}")
        reg_suffix = "_" + "_".join(reg_parts) if reg_parts else ""

        aug_suffix = ""
        if augmentation_config and augmentation_config.any_enabled():
            aug_suffix = "_aug"

        center_pairs_suffix = build_center_pairs_suffix(center_pairs)

        folder_name = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + mode_suffix
            + backbone_suffix
            + da_suffix
            + validator_suffix
            + variant_suffix
            + reg_suffix
            + aug_suffix
            + center_pairs_suffix
        )
        self.run_dir = self.base_workdir / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logging()
        self.logger.info(
            "Architecture switches: model_variant=%s, use_prostate_prior=%s, use_clinical=%s",
            self.model_variant,
            self.use_prostate_prior,
            self.use_clinical,
        )

        self.test_configs = [
            {
                "source_size": -1,
                "target_size": -1,
                "epochs": 100,
                "lr": 0.0005,
                "bs": 32,
                "da_weight": 0.1,
            },
            {
                "source_size": -1,
                "target_size": -1,
                "epochs": 100,
                "lr": 0.0005,
                "bs": 32,
                "da_weight": 0.5,
            },
            {
                "source_size": -1,
                "target_size": -1,
                "epochs": 100,
                "lr": 0.0005,
                "bs": 32,
                "da_weight": 0.9,
            },
        ]

        self.quick_configs = [
            {
                "source_size": -1,
                "target_size": -1,
                "epochs": 1,
                "lr": 0.0005,
                "bs": 32,
                "da_weight": 0.5,
            },
            # {'source_size': 5, 'target_size': -1, 'epochs': 30, 'lr': 0.01, 'bs': 1, 'da_weight': 0.0},
            # {'source_size': 50, 'target_size': -1, 'epochs': 20, 'lr': 0.001, 'bs': 4, 'da_weight': 0.5},
        ]

    def setup_logging(self):
        log_file = (
            self.run_dir
            / f"1_cnn_uda_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def _validate_center_configs(self, center_pairs):
        required_centers = set()
        for source_center, target_center in center_pairs:
            required_centers.add(source_center)
            required_centers.add(target_center)

        missing = [
            center
            for center in sorted(required_centers)
            if center not in self.dataset_configs
        ]
        if missing:
            raise ValueError(
                f"Missing dataset config for centers: {missing}. "
                f"Configured centers: {sorted(self.dataset_configs.keys())}"
            )

    def _create_trainer(self, source_center, target_center, workdir):
        """Create a UDATrainer with the current runner configuration."""
        source_config = self.dataset_configs[source_center]
        target_config = self.dataset_configs[target_center]
        return UDATrainer(
            source_center=source_center,
            target_center=target_center,
            workdir=workdir,
            device=self.device,
            logger=self.logger,
            data_dir=str(source_config.preprocessed_dir),
            use_preprocessed=True,
            binary_classification=self.binary_classification,
            da_method=self.da_method,
            upper_bound=self.upper_bound,
            dropout_rate=self.dropout_rate,
            use_batchnorm=self.use_batchnorm,
            checkpoint_validator=self.checkpoint_validator,
            checkpoint_save_interval=self.checkpoint_save_interval,
            use_early_stopping=self.use_early_stopping,
            da_warmup_epochs=self.da_warmup_epochs,
            use_class_weights=self.use_class_weights,
            backbone=self.backbone,
            weight_decay=self.weight_decay,
            lr_scheduler=self.lr_scheduler,
            lr_step_size=self.lr_step_size,
            lr_gamma=self.lr_gamma,
            two_splits_source=self.two_splits_source,
            two_splits_target=self.two_splits_target,
            daarda_divergence=self.daarda_divergence,
            daarda_relax=self.daarda_relax,
            daarda_grad_penalty=self.daarda_grad_penalty,
            augmentation_config=self.augmentation_config,
            model_variant=self.model_variant,
            use_prostate_prior=self.use_prostate_prior,
            prostate_prior_type=self.prostate_prior_type,
            prostate_prior_source_dir=self.prostate_prior_source_dir,
            prostate_prior_target_mode=self.prostate_prior_target_mode,
            prostate_prior_target_dir=self.prostate_prior_target_dir,
            prostate_prior_cache_dir=self.prostate_prior_cache_dir,
            prostate_prior_strength=self.prostate_prior_strength,
            prostate_prior_conf_thresh=self.prostate_prior_conf_thresh,
            use_clinical=self.use_clinical,
            clinical_features=self.clinical_features,
            clinical_fusion=self.clinical_fusion,
            clinical_impute=self.clinical_impute,
            clinical_missing_indicators=self.clinical_missing_indicators,
            source_clinical_column_map=self.source_clinical_column_map,
            target_clinical_column_map=self.target_clinical_column_map,
            source_dataset_config=source_config,
            target_dataset_config=target_config,
        )

    def run_experiment(self, source_center, target_center, config):
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"UDA Experiment: {source_center} -> {target_center}")
        self.logger.info(
            f"Config: src_size={config['source_size']}, tgt_size={config['target_size']}, "
            f"epochs={config['epochs']}, lr={config['lr']}, bs={config['bs']}, da_weight={config['da_weight']}"
        )
        self.logger.info(
            "Variant: model_variant=%s, prior=%s, clinical=%s",
            self.model_variant,
            self.use_prostate_prior,
            self.use_clinical,
        )
        self.logger.info(f"{'=' * 80}")

        if self.target_cv_folds > 1:
            return self._run_experiment_target_cv(source_center, target_center, config)

        start_time = time.time()
        da_weight = config["da_weight"]
        workdir = (
            self.run_dir / f"{source_center}_to_{target_center}" / f"da_{da_weight}"
        )

        trainer = self._create_trainer(source_center, target_center, workdir)

        result = trainer.train(
            source_size=config["source_size"],
            target_size=config["target_size"],
            num_epochs=config["epochs"],
            batch_size=config["bs"],
            learning_rate=config["lr"],
            da_weight=config["da_weight"],
        )

        result["source_center"] = source_center
        result["target_center"] = target_center
        result["config"] = config
        result["training_time"] = time.time() - start_time
        return result

    def _run_experiment_target_cv(self, source_center, target_center, config):
        """Run K-fold cross-validation on the target domain."""
        n_folds = self.target_cv_folds
        start_time = time.time()
        da_weight = config["da_weight"]
        base_workdir = (
            self.run_dir / f"{source_center}_to_{target_center}" / f"da_{da_weight}"
        )

        self.logger.info(
            f"Target {n_folds}-fold CV enabled (overrides --two-splits-target)"
        )

        # Build target dataset to extract labels for stratified folding.
        # We only need labels here, so branch-specific inputs are not required.
        target_config = self.dataset_configs[target_center]
        target_dataset = ISUPCenterDataset(
            center=target_center,
            marksheet_path=str(target_config.marksheet_path),
            data_dir=str(target_config.preprocessed_dir),
            use_preprocessed=True,
            binary_classification=self.binary_classification,
            dataset_config=target_config,
        )
        target_labels = target_dataset.isup_labels
        n_target = len(target_labels)
        self.logger.info(f"Target dataset: {n_target} samples for {n_folds}-fold CV")

        _, label_counts = np.unique(target_labels, return_counts=True)
        min_count = label_counts.min()
        if min_count >= n_folds:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_iter = kfold.split(np.arange(n_target), target_labels)
            self.logger.info(f"Using StratifiedKFold (min class count={min_count})")
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_iter = kfold.split(np.arange(n_target))
            self.logger.warning(
                f"Cannot stratify: min class count={min_count} < {n_folds} folds. Using KFold."
            )

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_iter):
            self.logger.info(f"\n{'─' * 60}")
            self.logger.info(
                f"FOLD {fold_idx + 1}/{n_folds}: "
                f"target_train={len(train_idx)}, target_val={len(val_idx)}"
            )
            self.logger.info(f"{'─' * 60}")

            fold_workdir = base_workdir / f"fold_{fold_idx + 1}"
            torch.manual_seed(42)
            np.random.seed(42)

            trainer = self._create_trainer(source_center, target_center, fold_workdir)

            fold_result = trainer.train(
                source_size=config["source_size"],
                target_size=config["target_size"],
                num_epochs=config["epochs"],
                batch_size=config["bs"],
                learning_rate=config["lr"],
                da_weight=config["da_weight"],
                target_train_indices=train_idx.tolist(),
                target_val_indices=val_idx.tolist(),
            )
            fold_result["fold"] = fold_idx + 1
            fold_results.append(fold_result)

        result = self._aggregate_fold_results(fold_results)
        result["source_center"] = source_center
        result["target_center"] = target_center
        result["config"] = config
        result["training_time"] = time.time() - start_time
        result["target_cv_folds"] = n_folds
        result["cv_enabled"] = True
        return result

    @staticmethod
    def _aggregate_fold_results(fold_results):
        """Aggregate metrics across CV folds (mean ± std)."""
        metric_keys = [
            "final_target_test_accuracy",
            "final_target_test_balanced_accuracy",
            "final_target_test_auc",
            "final_target_test_sensitivity",
            "final_target_test_specificity",
            "final_target_test_macro_f1",
            "final_target_test_micro_f1",
            "final_source_test_accuracy",
            "final_source_test_balanced_accuracy",
            "final_source_test_auc",
            "final_source_test_sensitivity",
            "final_source_test_specificity",
        ]
        aggregated = {}
        for key in metric_keys:
            values = [result[key] for result in fold_results if key in result]
            if values:
                aggregated[key] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        first = fold_results[0]
        for key in [
            "source_train_size",
            "source_val_size",
            "source_test_size",
            "two_splits_source",
            "two_splits_target",
            "checkpoint_validator",
            "mode",
            "model_variant",
            "use_prostate_prior",
            "prostate_prior_type",
            "use_clinical",
            "clinical_feature_dim",
            "clinical_fusion",
            "input_channels",
        ]:
            if key in first:
                aggregated[key] = first[key]

        aggregated["target_train_size"] = int(
            np.mean([result.get("target_train_size", 0) for result in fold_results])
        )
        aggregated["target_val_size"] = int(
            np.mean([result.get("target_val_size", 0) for result in fold_results])
        )
        aggregated["target_test_size"] = int(
            np.mean([result.get("target_test_size", 0) for result in fold_results])
        )
        aggregated["fold_results"] = fold_results
        return aggregated

    def ensure_preprocessed_data(self, center_pairs):
        self.logger.info("Checking for preprocessed data...")
        seen_centers = set()

        for source_center, target_center in center_pairs:
            seen_centers.add(source_center)
            seen_centers.add(target_center)

        for center_name in sorted(seen_centers):
            config = self.dataset_configs[center_name]
            if check_preprocessed_data_exists_from_config(config):
                self.logger.info(
                    f"Preprocessed data found for {center_name} in {config.preprocessed_dir}"
                )
                continue

            self.logger.info(
                f"Preprocessed data missing for {center_name}. Starting preprocessing..."
            )
            preprocess_and_save_dataset_from_config(config)
            self.logger.info(f"Preprocessing complete for {center_name}")

    def run_all_experiments(
        self, quick=False, filter_center_pairs=None, filter_da_weights=None
    ):
        self.logger.info("Starting UDA experiments")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Results directory: {self.run_dir}")

        center_pairs = list(self.center_pairs)

        if filter_center_pairs:
            filtered_pairs = []
            for src, tgt in center_pairs:
                pair_name = f"{src}_to_{tgt}"
                if pair_name in filter_center_pairs:
                    filtered_pairs.append((src, tgt))
            center_pairs = filtered_pairs
            self.logger.info(
                f"Filtered center pairs: {[f'{s}_to_{t}' for s, t in center_pairs]}"
            )

        if not center_pairs:
            raise ValueError("No center pairs left to run after filtering.")

        self._validate_center_configs(center_pairs)
        self.ensure_preprocessed_data(center_pairs)

        configs = self.quick_configs if quick else self.test_configs
        if filter_da_weights:
            configs = [cfg for cfg in configs if cfg["da_weight"] in filter_da_weights]
            self.logger.info(
                f"Filtered DA weights: {[cfg['da_weight'] for cfg in configs]}"
            )

        if self.lower_bound:
            self.logger.info("Lower bound mode: forcing da_weight=0.0 for all configs.")
            configs = [{**cfg, "da_weight": 0.0} for cfg in configs]

        for source, target in center_pairs:
            for i, config in enumerate(configs):
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(
                    f"Progress: {source}->{target} Config {i + 1}/{len(configs)}"
                )
                self.logger.info(f"{'=' * 80}")

                try:
                    result = self.run_experiment(source, target, config)
                    self.results.append(result)
                    self.save_results()
                except Exception as exc:
                    self.logger.error(f"Error in experiment: {exc}")
                    import traceback

                    traceback.print_exc()
                    continue

        self.generate_summary()
        if self.plot_oracle:
            self.generate_oracle_plots()

        self.logger.info(
            f"\nAll experiments completed! Results saved to: {self.run_dir}"
        )

    def generate_oracle_plots(self):
        """Generate oracle selector comparison plots."""
        self.logger.info("\nGenerating Oracle Selector plots...")
        try:
            scripts_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(scripts_dir))
            from plot_oracle_selector import plot_oracle_comparison_single

            output_dir = self.run_dir / "oracle_plots"
            output_dir.mkdir(exist_ok=True)

            plot_oracle_comparison_single(self.run_dir, output_dir)
            self.logger.info(f"Oracle plots saved to: {output_dir}")
        except Exception as exc:
            self.logger.warning(f"Failed to generate oracle plots: {exc}")

    def save_results(self):
        results_file = self.run_dir / "results.json"
        with open(results_file, "w") as file:
            json.dump(self.results, file, indent=2, default=str)

    def generate_summary(self):
        if not self.results:
            return

        summary_file = self.run_dir / "summary.txt"
        with open(summary_file, "w") as file:
            file.write("UDA EXPERIMENT SUMMARY\n")
            file.write("=" * 120 + "\n\n")
            file.write(f"Completed: {datetime.now()}\n")
            file.write(f"Total experiments: {len(self.results)}\n\n")

            file.write("RESULTS:\n")
            file.write("-" * 120 + "\n")
            for i, result in enumerate(self.results):
                src_size = result.get("source_train_size", result.get("source_size", 0))
                is_cv = result.get("cv_enabled", False)

                def _fmt(key):
                    val = result.get(key, 0)
                    std_key = f"{key}_std"
                    if is_cv and std_key in result:
                        return f"{val:.1f}±{result[std_key]:.1f}%"
                    return f"{val:.1f}%"

                cv_tag = (
                    f" [{result.get('target_cv_folds', 1)}-fold CV]" if is_cv else ""
                )
                file.write(
                    f"{i + 1:2d}. {result.get('source_center', 'N/A')}->{result.get('target_center', 'N/A')}{cv_tag}: "
                    f"Size={src_size}, "
                    f"SrcTestBalAcc={_fmt('final_source_test_balanced_accuracy')}, "
                    f"SrcTestAUC={_fmt('final_source_test_auc')}, "
                    f"TgtTestAcc={_fmt('final_target_test_accuracy')}, "
                    f"TgtTestBalAcc={_fmt('final_target_test_balanced_accuracy')}, "
                    f"TgtTestAUC={_fmt('final_target_test_auc')}, "
                    f"Sens={_fmt('final_target_test_sensitivity')}, "
                    f"Spec={_fmt('final_target_test_specificity')}, "
                    f"F1={_fmt('final_target_test_macro_f1')}, "
                    f"Validator={result.get('checkpoint_validator', 'N/A')}, "
                    f"BestEpoch={result.get('best_epoch', 'N/A')}, "
                    f"Time={result.get('training_time', 0):.1f}s\n"
                )


def main():
    parser = argparse.ArgumentParser(description="UDA CNN Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Binary classification (csPCa vs no csPCa)",
    )
    parser.add_argument(
        "--da-method",
        type=str,
        default="coral",
        choices=[
            "coral",
            "entropy",
            "hybrid",
            "mmd",
            "dann",
            "mcd",
            "mcc",
            "bnm",
            "daarda",
            "none",
        ],
        help="Domain adaptation method: coral, entropy, hybrid, mmd, dann, mcd, mcc, bnm, daarda, none",
    )
    parser.add_argument(
        "--daarda-divergence",
        type=str,
        default="js_beta",
        choices=["js", "js_beta", "w_beta", "js_sort"],
        help="DAARDA divergence variant (used only when --da-method daarda)",
    )
    parser.add_argument(
        "--daarda-relax",
        type=float,
        default=1.0,
        help="DAARDA relaxation beta (used only when --da-method daarda)",
    )
    parser.add_argument(
        "--daarda-grad-penalty",
        type=float,
        default=0.0,
        help="DAARDA discriminator gradient penalty weight (used only when --da-method daarda)",
    )
    parser.add_argument(
        "--lower-bound",
        action="store_true",
        help="Run lower bound (source-only, no DA)",
    )
    parser.add_argument(
        "--upper-bound",
        action="store_true",
        help="Run upper bound (supervised on source+target)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)"
    )
    parser.add_argument(
        "--no-batchnorm", action="store_true", help="Disable batch normalization"
    )
    parser.add_argument(
        "--checkpoint-validator",
        type=str,
        default="source_val",
        choices=["source_val", "target_val"],
        help="Checkpoint selection strategy: source_val or target_val (oracle)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true", help="Disable early stopping"
    )
    parser.add_argument(
        "--da-warmup", type=int, default=0, help="DA warmup epochs (CE-only before DA)"
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class weights for imbalanced classes",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet10",
        choices=["resnet10", "resnet18", "resnet34", "resnet50"],
        help="Model backbone",
    )
    parser.add_argument(
        "--plot-oracle",
        action="store_true",
        help="Generate oracle selector plots after training",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=["none", "step", "cosine", "inv"],
        help="LR scheduler: none, step, cosine, inv",
    )
    parser.add_argument(
        "--lr-step-size", type=int, default=30, help="Step size for StepLR"
    )
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="Gamma for StepLR")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--workdir", type=str, default="workdir/uda", help="Working directory"
    )

    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="public",
        choices=["public", "public_to_uulm"],
        help="Dataset profile: current public setup or merged-public to UULM",
    )
    parser.add_argument(
        "--center-pairs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit center pairs SOURCE_to_TARGET (overrides profile defaults)",
    )

    parser.add_argument(
        "--public-marksheet",
        type=str,
        default="input/picai_labels/clinical_information/marksheet.csv",
    )
    parser.add_argument("--public-raw-dir", type=str, default="input/images")
    parser.add_argument(
        "--public-preprocessed-dir", type=str, default="input/images_preprocessed"
    )

    parser.add_argument("--uulm-center-name", type=str, default="UULM")
    parser.add_argument("--uulm-label-file", type=str, default="0ii/man.xlsx")
    parser.add_argument(
        "--uulm-use-dual-metadata",
        action="store_true",
        help="Load separate MRI-only and PET/MRI UULM metadata sheets together.",
    )
    parser.add_argument("--uulm-pet-label-file", type=str, default="0ii/pet.xlsx")
    parser.add_argument("--uulm-pet-sheet-name", type=str, default="Auswertung")
    parser.add_argument("--uulm-raw-dir", type=str, default="0ii/files/mri_data_robust")
    parser.add_argument(
        "--uulm-preprocessed-dir", type=str, default="0ii/files/registered"
    )
    parser.add_argument("--uulm-label-column", type=str, default="Gleason-Score gesamt")
    parser.add_argument("--uulm-patient-id-column", type=str, default="Patientennr")
    parser.add_argument("--uulm-study-id-column", type=str, default="Auftragsnr")
    parser.add_argument("--uulm-pet-label-column", type=str, default="Gleason")
    parser.add_argument("--uulm-pet-patient-id-column", type=str, default="PatientenID")
    parser.add_argument("--uulm-pet-study-id-column", type=str, default="AuftragsID")

    parser.add_argument(
        "--filter-center-pairs",
        type=str,
        nargs="+",
        default=None,
        help="Only run specified center pairs (e.g., RUMC_to_PCNN)",
    )
    parser.add_argument(
        "--filter-da-weights",
        type=float,
        nargs="+",
        default=None,
        help="Only run specified DA weights (e.g., 0.1 0.5)",
    )

    parser.add_argument(
        "--two-splits-source",
        action="store_true",
        help="Use 2-split for source (85%% train, 15%% eval) instead of 3-split",
    )
    parser.add_argument(
        "--two-splits-target",
        action="store_true",
        help="Use 2-split for target (85%% train, 15%% eval) instead of 3-split",
    )
    parser.add_argument(
        "--target-cv-folds",
        type=int,
        default=1,
        help="If >1, run target-domain K-fold CV (e.g., 5 for 5-fold). Overrides --two-splits-target.",
    )

    branch_group = parser.add_argument_group(
        "Optional Classification Branches",
        "Classification-first branches; all disabled by default for backward compatibility.",
    )
    branch_group.add_argument(
        "--model-variant",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "prostate_prior",
            "clinical",
            "prostate_clinical",
            "pet_fusion",
        ],
        help="High-level model variant preset.",
    )
    branch_group.add_argument(
        "--use-prostate-prior",
        action="store_true",
        help="Enable optional prostate-prior channels",
    )
    branch_group.add_argument(
        "--prostate-prior-type",
        type=str,
        default="whole_gland",
        choices=["whole_gland", "zonal", "both"],
        help="Prior channel format: whole gland, zonal, or both.",
    )
    branch_group.add_argument(
        "--prostate-prior-source",
        type=str,
        default="bosma22b",
        choices=["bosma22b", "guerbet23", "heviai23", "yuan23", "custom"],
        help="Source prior preset. Use custom with --prostate-prior-source-dir.",
    )
    branch_group.add_argument(
        "--prostate-prior-source-dir",
        type=str,
        default=None,
        help="Custom source prior directory",
    )
    branch_group.add_argument(
        "--prostate-prior-target",
        type=str,
        default="none",
        choices=["none", "pseudo"],
        help="Target prior mode: none or pseudo-mask directory.",
    )
    branch_group.add_argument(
        "--prostate-prior-target-dir",
        type=str,
        default=None,
        help="Target pseudo-prior directory",
    )
    branch_group.add_argument(
        "--prostate-prior-cache-dir",
        type=str,
        default=None,
        help="Optional cache dir for processed priors",
    )
    branch_group.add_argument(
        "--prostate-prior-strength",
        type=float,
        default=1.0,
        help="Scale factor for prior channels",
    )
    branch_group.add_argument(
        "--prostate-prior-conf-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for soft pseudo priors (target mode pseudo).",
    )

    branch_group.add_argument(
        "--use-clinical",
        action="store_true",
        help="Enable optional clinical metadata branch",
    )
    branch_group.add_argument(
        "--clinical-features",
        type=str,
        nargs="+",
        default=["psa", "psad", "prostate_volume", "age"],
        help="Clinical feature names to include (default: psa psad prostate_volume age).",
    )
    branch_group.add_argument(
        "--clinical-fusion",
        type=str,
        default="early",
        choices=["early", "late"],
        help="Clinical fusion strategy.",
    )
    branch_group.add_argument(
        "--clinical-impute",
        type=str,
        default="median",
        choices=["median", "constant"],
        help="Clinical imputation strategy.",
    )
    branch_group.add_argument(
        "--clinical-missing-indicators",
        action="store_true",
        help="Append binary missingness indicators to clinical features.",
    )
    branch_group.add_argument(
        "--source-clinical-map",
        type=str,
        nargs="+",
        default=None,
        help="Source feature-to-column map entries FEATURE=COLUMN (e.g., psa=PSA).",
    )
    branch_group.add_argument(
        "--target-clinical-map",
        type=str,
        nargs="+",
        default=None,
        help="Target feature-to-column map entries FEATURE=COLUMN (e.g., psa=tPSA).",
    )

    pet_group = parser.add_argument_group("PET Fusion (Planned)")
    pet_group.add_argument(
        "--use-pet",
        action="store_true",
        help="Reserved switch; PET fusion not implemented yet.",
    )
    pet_group.add_argument(
        "--pet-fusion",
        type=str,
        default="late",
        choices=["late"],
        help="Reserved PET fusion mode.",
    )
    pet_group.add_argument(
        "--pet-freeze-image-backbone",
        action="store_true",
        help="Reserved PET flag; MRI backbone freezing is not implemented yet.",
    )

    aug_group = parser.add_argument_group(
        "Data Augmentation", "Simple MRI augmentations, disabled by default."
    )
    aug_group.add_argument(
        "--aug-noise", action="store_true", help="Enable Gaussian noise"
    )
    aug_group.add_argument(
        "--aug-brightness", action="store_true", help="Enable brightness adjustment"
    )
    aug_group.add_argument(
        "--aug-contrast", action="store_true", help="Enable contrast adjustment"
    )
    aug_group.add_argument(
        "--aug-blur", action="store_true", help="Enable Gaussian blur"
    )
    aug_group.add_argument(
        "--aug-all", action="store_true", help="Enable all augmentations"
    )

    args = parser.parse_args()

    if args.target_cv_folds < 1:
        raise ValueError("--target-cv-folds must be >= 1")

    torch.manual_seed(42)
    np.random.seed(42)

    model_variant = args.model_variant.lower()
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported --model-variant '{args.model_variant}'.")

    if args.use_pet or model_variant == "pet_fusion":
        raise NotImplementedError(
            "PET fusion is planned but not implemented in this runner/trainer path yet."
        )

    use_prostate_prior = args.use_prostate_prior or model_variant in {
        "prostate_prior",
        "prostate_clinical",
    }
    use_clinical = args.use_clinical or model_variant in {
        "clinical",
        "prostate_clinical",
    }

    if not use_prostate_prior and args.prostate_prior_target != "none":
        raise ValueError(
            "--prostate-prior-target requires --use-prostate-prior or a prior model variant."
        )

    source_clinical_map = parse_feature_column_map(args.source_clinical_map)
    target_clinical_map = parse_feature_column_map(args.target_clinical_map)

    if use_clinical and len(args.clinical_features) == 0:
        raise ValueError("Clinical branch enabled, but --clinical-features is empty.")

    public_extra_columns = (
        resolve_feature_columns(args.clinical_features, source_clinical_map)
        if use_clinical
        else None
    )
    uulm_extra_columns = (
        resolve_feature_columns(args.clinical_features, target_clinical_map)
        if use_clinical
        else None
    )

    augmentation_config = None
    if (
        args.aug_all
        or args.aug_noise
        or args.aug_brightness
        or args.aug_contrast
        or args.aug_blur
    ):
        augmentation_config = AugmentationConfig(
            gaussian_noise=args.aug_all or args.aug_noise,
            brightness=args.aug_all or args.aug_brightness,
            contrast=args.aug_all or args.aug_contrast,
            blur=args.aug_all or args.aug_blur,
        )
        print(f"Data augmentation enabled: {augmentation_config.get_enabled_names()}")

    if args.dataset_profile == "public_to_uulm":
        if not Path(args.uulm_label_file).exists():
            raise FileNotFoundError(
                f"UULM MRI metadata file not found: {args.uulm_label_file}. "
                f"Set --uulm-label-file to the confidential MRI metadata sheet path."
            )
        if args.uulm_use_dual_metadata and not Path(args.uulm_pet_label_file).exists():
            raise FileNotFoundError(
                f"UULM PET metadata file not found: {args.uulm_pet_label_file}. "
                f"Set --uulm-pet-label-file to the confidential PET metadata sheet path."
            )

    prostate_prior_source_dir = None
    prostate_prior_target_dir = None
    if use_prostate_prior:
        prostate_prior_source_dir = resolve_prostate_prior_source_dir(
            args.prostate_prior_source,
            args.prostate_prior_source_dir,
        )
        if not prostate_prior_source_dir.exists():
            raise FileNotFoundError(
                f"Prostate prior source directory not found: {prostate_prior_source_dir}"
            )

        if args.prostate_prior_target == "pseudo":
            if not args.prostate_prior_target_dir:
                raise ValueError(
                    "--prostate-prior-target pseudo requires --prostate-prior-target-dir"
                )
            prostate_prior_target_dir = Path(args.prostate_prior_target_dir)

    dataset_configs = build_dataset_configs(
        args,
        public_extra_columns=public_extra_columns,
        uulm_extra_columns=uulm_extra_columns,
    )
    center_pairs = (
        parse_center_pairs(args.center_pairs)
        if args.center_pairs
        else get_default_center_pairs(args)
    )

    runner = UDARunner(
        dataset_configs=dataset_configs,
        center_pairs=center_pairs,
        base_workdir=args.workdir,
        gpu_id=args.gpu_id,
        binary_classification=args.binary,
        da_method=args.da_method,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        dropout_rate=args.dropout,
        use_batchnorm=not args.no_batchnorm,
        checkpoint_validator=args.checkpoint_validator,
        checkpoint_save_interval=args.checkpoint_interval,
        use_early_stopping=not args.no_early_stopping,
        da_warmup_epochs=args.da_warmup,
        use_class_weights=args.class_weights,
        backbone=args.backbone,
        plot_oracle=args.plot_oracle,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        two_splits_source=args.two_splits_source,
        two_splits_target=args.two_splits_target,
        daarda_divergence=args.daarda_divergence,
        daarda_relax=args.daarda_relax,
        daarda_grad_penalty=args.daarda_grad_penalty,
        augmentation_config=augmentation_config,
        target_cv_folds=args.target_cv_folds,
        model_variant=model_variant,
        use_prostate_prior=use_prostate_prior,
        prostate_prior_type=args.prostate_prior_type,
        prostate_prior_source_dir=str(prostate_prior_source_dir)
        if prostate_prior_source_dir
        else None,
        prostate_prior_target_mode=args.prostate_prior_target,
        prostate_prior_target_dir=str(prostate_prior_target_dir)
        if prostate_prior_target_dir
        else None,
        prostate_prior_cache_dir=args.prostate_prior_cache_dir,
        prostate_prior_strength=args.prostate_prior_strength,
        prostate_prior_conf_thresh=args.prostate_prior_conf_thresh,
        use_clinical=use_clinical,
        clinical_features=args.clinical_features,
        clinical_fusion=args.clinical_fusion,
        clinical_impute=args.clinical_impute,
        clinical_missing_indicators=args.clinical_missing_indicators,
        source_clinical_column_map=source_clinical_map,
        target_clinical_column_map=target_clinical_map,
    )
    runner.run_all_experiments(
        quick=args.quick,
        filter_center_pairs=args.filter_center_pairs,
        filter_da_weights=args.filter_da_weights,
    )


if __name__ == "__main__":
    main()
