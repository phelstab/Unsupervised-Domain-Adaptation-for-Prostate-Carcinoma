import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import create_model
from training_setup.data_generator import ISUPCenterDataset
from training_setup.loss_functions import (
    ISUPLoss,
    CORALLoss,
    EntropyLoss,
    MMDLoss,
    DANNLoss,
    MCDLoss,
    MCCLoss,
    BNMLoss,
    DAARDALoss,
)
from training_setup.domain_discriminator import DomainDiscriminator
from training_setup.label_analyzer import analyze_splits
from training_setup.augmentations import AugmentationConfig
from training_setup.dataset_config import DatasetConfig


def inv_lr_scheduler(optimizer, p, gamma, power, lr=0.001, weight_decay=0.0005):
    """Inverse learning rate scheduler (Ganin 2016 / DANN convention).

    Decays the learning rate as: lr = lr_base * (1 + gamma * p) ** (-power)
    where `p` is the NORMALISED training progress in [0, 1]
    (cumulative iteration count divided by the total number of iterations).

    This matches the formulation in:
      - Ganin et al. 2016, "Domain-Adversarial Training of Neural Networks"
      - Cui et al. 2020, BNM reference implementation
      https://github.com/cuishuhao/BNM/blob/master/DA/BNM/lr_schedule.py
    With gamma=10 and power=0.75 this decays from lr_base at p=0 to
    ~0.166 * lr_base at p=1.

    Args:
        optimizer: PyTorch optimizer with param groups having 'lr_mult' and 'decay_mult'
        p: Normalised training progress in [0, 1] (iter_num / total_iters)
        gamma: Controls decay speed (typical: 10 for normalised p)
        power: Controls decay curve shape (typical: 0.75)
        lr: Base learning rate
        weight_decay: Base weight decay

    Returns:
        optimizer: Updated optimizer (modifies in-place)
    """
    lr = lr * (1 + gamma * p) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_mult", 1.0)
        param_group["weight_decay"] = weight_decay * param_group.get("decay_mult", 1.0)
    return optimizer


class ValidatorType(Enum):
    """Checkpoint selection strategy."""

    SOURCE_VAL = "source_val"
    TARGET_VAL = "target_val"


@dataclass
class DataSplits:
    """Container for all data splits.

    Supports two splitting modes per domain (source/target):

    3-split mode (default): train/val/test (70%/15%/15%)
        - val: used for checkpoint selection during training
        - test: held-out final evaluation

    2-split mode: train/eval (85%/15%)
        - eval: used for BOTH checkpoint selection AND final evaluation
        - Recommended when data is limited (per supervisor advice)
        - Use --two-splits-source / --two-splits-target to enable
    """

    source_train: Subset
    source_val: Subset
    source_test: Optional[Subset]  # None in 2-split mode (use source_val as eval)
    target_train: Subset
    target_val: Subset
    target_test: Optional[Subset]  # None in 2-split mode (use target_val as eval)

    def __post_init__(self):
        assert len(self.source_train) > 0, "source_train must not be empty"
        assert len(self.source_val) > 0, "source_val must not be empty"
        assert len(self.target_train) > 0, "target_train must not be empty"
        assert len(self.target_val) > 0, "target_val must not be empty"
        # source_test and target_test can be None in 2-split mode


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    epoch: int
    model_state: dict
    source_val_bal_acc: float
    target_val_bal_acc: float
    path: Optional[Path] = None
    outputs_path: Optional[Path] = None
    aux_state: Optional[dict] = None  # For MCD/DANN classifier states


@dataclass
class SampleOutputs:
    """Outputs for a single sample - for post-hoc validator selection."""

    sample_ids: list
    logits: np.ndarray
    features: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray


class CheckpointManager:
    """Manages checkpoint saving and selection."""

    def __init__(
        self,
        workdir: Path,
        save_interval: int = 10,
        keep_all: bool = True,
        save_outputs: bool = True,
    ):
        self.workdir = workdir
        self.save_interval = save_interval
        self.keep_all = keep_all
        self.save_outputs = save_outputs
        self.checkpoints_dir = workdir / "checkpoints"
        self.outputs_dir = workdir / "outputs"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if save_outputs:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.all_checkpoints: list[CheckpointInfo] = []
        self.best_by_source: Optional[CheckpointInfo] = None
        self.best_by_target: Optional[CheckpointInfo] = None

    def should_save(self, epoch: int, num_epochs: int) -> bool:
        """Determine if checkpoint should be saved at this epoch."""
        if epoch < 10:
            return False
        if epoch == num_epochs - 1:
            return True
        if self.save_interval > 0 and epoch % self.save_interval == 0:
            return True
        return False

    def save_checkpoint(
        self,
        epoch: int,
        model_state: dict,
        source_val_bal_acc: float,
        target_val_bal_acc: float,
        outputs: Optional[Dict[str, SampleOutputs]] = None,
        aux_state: Optional[dict] = None,
    ) -> CheckpointInfo:
        """Save a checkpoint and update best trackers.

        Args:
            epoch: Current epoch number
            model_state: Model state dict
            source_val_bal_acc: Source validation balanced accuracy
            target_val_bal_acc: Target validation balanced accuracy
            outputs: Dict with keys like 'source_train', 'source_val', 'target_train', 'target_val'
                    containing SampleOutputs for each split (for post-hoc validator selection)
            aux_state: Optional dict with auxiliary state (e.g., MCD/DANN classifier weights)
        """
        checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:04d}.pt"
        outputs_path = (
            self.outputs_dir / f"epoch_{epoch:04d}_outputs.pt"
            if self.save_outputs
            else None
        )

        checkpoint_info = CheckpointInfo(
            epoch=epoch,
            model_state=deepcopy(model_state),
            source_val_bal_acc=source_val_bal_acc,
            target_val_bal_acc=target_val_bal_acc,
            path=checkpoint_path,
            outputs_path=outputs_path,
            aux_state=deepcopy(aux_state) if aux_state else None,
        )

        # Save combined state (model + aux) to checkpoint file
        save_dict = {"model": model_state}
        if aux_state:
            save_dict["aux"] = aux_state

        if self.keep_all:
            torch.save(save_dict, checkpoint_path)
            self.all_checkpoints.append(checkpoint_info)

        if self.save_outputs and outputs is not None:
            outputs_dict = {}
            for split_name, split_outputs in outputs.items():
                outputs_dict[split_name] = {
                    "sample_ids": split_outputs.sample_ids,
                    "logits": split_outputs.logits,
                    "features": split_outputs.features,
                    "predictions": split_outputs.predictions,
                    "labels": split_outputs.labels,
                    "probabilities": split_outputs.probabilities,
                }
            torch.save(outputs_dict, outputs_path)

        if (
            self.best_by_source is None
            or source_val_bal_acc > self.best_by_source.source_val_bal_acc
        ):
            self.best_by_source = checkpoint_info
            torch.save(save_dict, self.workdir / "best_by_source_val.pt")

        if (
            self.best_by_target is None
            or target_val_bal_acc > self.best_by_target.target_val_bal_acc
        ):
            self.best_by_target = checkpoint_info
            torch.save(save_dict, self.workdir / "best_by_target_val.pt")

        return checkpoint_info

    def get_best_checkpoint(self, validator: ValidatorType) -> Optional[CheckpointInfo]:
        """Get the best checkpoint based on validator type."""
        if validator == ValidatorType.SOURCE_VAL:
            return self.best_by_source
        else:
            return self.best_by_target

    def get_summary(self) -> dict:
        """Get summary of all checkpoints."""
        return {
            "total_checkpoints": len(self.all_checkpoints),
            "best_by_source": {
                "epoch": self.best_by_source.epoch if self.best_by_source else None,
                "source_val_bal_acc": self.best_by_source.source_val_bal_acc
                if self.best_by_source
                else None,
                "target_val_bal_acc": self.best_by_source.target_val_bal_acc
                if self.best_by_source
                else None,
            }
            if self.best_by_source
            else None,
            "best_by_target": {
                "epoch": self.best_by_target.epoch if self.best_by_target else None,
                "source_val_bal_acc": self.best_by_target.source_val_bal_acc
                if self.best_by_target
                else None,
                "target_val_bal_acc": self.best_by_target.target_val_bal_acc
                if self.best_by_target
                else None,
            }
            if self.best_by_target
            else None,
        }


class UDATrainer:
    """
    Unified Domain Adaptation Trainer supporting three modes:

    1. LOWER BOUND (baseline): Train on source only, test on target
       - da_method='none', upper_bound=False

    2. UDA: Train supervised on source + unsupervised DA on target
       - da_method in ['coral', 'entropy', 'hybrid'], upper_bound=False

    3. UPPER BOUND: Train supervised on source+target combined
       - upper_bound=True (da_method ignored)

    Features:
    - Source test evaluation
    - Target-based checkpoint selection (oracle upper bound)
    - Periodic checkpoint saving (every N epochs)
    """

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

    def __init__(
        self,
        source_center: str,
        target_center: str,
        workdir: str,
        device: torch.device,
        logger=None,
        data_dir: str = "input/images",
        use_preprocessed: bool = False,
        binary_classification: bool = False,
        da_method: str = "coral",
        upper_bound: bool = False,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
        checkpoint_validator: str = "source_val",
        checkpoint_save_interval: int = 10,
        use_early_stopping: bool = True,
        da_warmup_epochs: int = 0,
        use_class_weights: bool = False,
        backbone: str = "resnet10",
        weight_decay: float = 0.0,
        lr_scheduler: str = "none",
        lr_step_size: int = 30,
        lr_gamma: float = 0.1,
        two_splits_source: bool = False,
        two_splits_target: bool = False,
        daarda_divergence: str = "js_beta",
        daarda_relax: float = 1.0,
        daarda_grad_penalty: float = 0.0,
        augmentation_config: Optional[AugmentationConfig] = None,
        model_variant: str = "baseline",
        use_prostate_prior: bool = False,
        prostate_prior_type: str = "whole_gland",
        prostate_prior_source_dir: Optional[str] = None,
        prostate_prior_target_mode: str = "none",
        prostate_prior_target_dir: Optional[str] = None,
        prostate_prior_cache_dir: Optional[str] = None,
        prostate_prior_strength: float = 1.0,
        prostate_prior_conf_thresh: float = 0.5,
        use_clinical: bool = False,
        clinical_features: Optional[list[str]] = None,
        clinical_fusion: str = "early",
        clinical_impute: str = "median",
        clinical_missing_indicators: bool = False,
        source_clinical_column_map: Optional[dict[str, str]] = None,
        target_clinical_column_map: Optional[dict[str, str]] = None,
        source_dataset_config: Optional[DatasetConfig] = None,
        target_dataset_config: Optional[DatasetConfig] = None,
    ):

        self.source_center = source_center
        self.target_center = target_center
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.data_dir = data_dir
        self.use_preprocessed = use_preprocessed
        self.binary_classification = binary_classification
        self.num_classes = 2 if binary_classification else 6
        self.source_dataset_config = source_dataset_config
        self.target_dataset_config = target_dataset_config

        self.model_variant = model_variant.lower()
        if self.model_variant not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant '{model_variant}'. "
                f"Choose from {sorted(self.MODEL_VARIANTS)}."
            )

        if self.model_variant == "pet_fusion":
            raise NotImplementedError(
                "PET fusion is planned but not implemented in UDATrainer yet. "
                "Use model_variant='baseline' | 'prostate_prior' | 'clinical' | 'prostate_clinical'."
            )

        variant_uses_prostate_prior = self.model_variant in {
            "prostate_prior",
            "prostate_clinical",
        }
        variant_uses_clinical = self.model_variant in {"clinical", "prostate_clinical"}

        self.use_prostate_prior = use_prostate_prior or variant_uses_prostate_prior
        self.prostate_prior_type = prostate_prior_type
        self.prostate_prior_source_dir = (
            Path(prostate_prior_source_dir) if prostate_prior_source_dir else None
        )
        self.prostate_prior_target_mode = prostate_prior_target_mode
        self.prostate_prior_target_dir = (
            Path(prostate_prior_target_dir) if prostate_prior_target_dir else None
        )
        self.prostate_prior_cache_dir = (
            Path(prostate_prior_cache_dir) if prostate_prior_cache_dir else None
        )
        self.prostate_prior_strength = prostate_prior_strength
        self.prostate_prior_conf_thresh = prostate_prior_conf_thresh

        self.use_clinical = use_clinical or variant_uses_clinical
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
        self.clinical_feature_dim = 0

        self.input_channels = 3
        if self.use_prostate_prior:
            if self.prostate_prior_type == "whole_gland":
                self.input_channels += 1
            elif self.prostate_prior_type == "zonal":
                self.input_channels += 2
            elif self.prostate_prior_type == "both":
                self.input_channels += 3
            else:
                raise ValueError(
                    f"Unsupported prostate_prior_type '{self.prostate_prior_type}'. "
                    "Choose from 'whole_gland', 'zonal', 'both'."
                )

            if self.prostate_prior_source_dir is None:
                raise ValueError(
                    "use_prostate_prior=True requires prostate_prior_source_dir. "
                    "Provide a directory with source-side gland/zonal priors."
                )

        if self.prostate_prior_target_mode not in {"none", "pseudo"}:
            raise ValueError(
                f"Unsupported prostate_prior_target_mode '{self.prostate_prior_target_mode}'. "
                "Choose from 'none' or 'pseudo'."
            )

        if (
            self.use_prostate_prior
            and self.prostate_prior_target_mode == "pseudo"
            and self.prostate_prior_target_dir is None
        ):
            raise ValueError(
                "prostate_prior_target_mode='pseudo' requires prostate_prior_target_dir."
            )

        if self.use_clinical and self.clinical_fusion not in {"early", "late"}:
            raise ValueError(
                f"Unsupported clinical_fusion '{self.clinical_fusion}'. "
                "Choose from 'early' or 'late'."
            )

        if self.use_clinical and len(self.clinical_features) == 0:
            raise ValueError("use_clinical=True requires at least one feature name.")

        if variant_uses_prostate_prior and not use_prostate_prior:
            self.logger.info(
                "model_variant '%s' enables prostate priors.", self.model_variant
            )
        if variant_uses_clinical and not use_clinical:
            self.logger.info(
                "model_variant '%s' enables clinical fusion.", self.model_variant
            )

        # Augmentation config
        self.augmentation_config = (
            augmentation_config
            if augmentation_config and augmentation_config.any_enabled()
            else None
        )
        if self.augmentation_config is not None:
            self.logger.info(
                f"Data augmentation enabled (GPU path): {self.augmentation_config.get_enabled_names()}"
            )
        self.upper_bound = upper_bound
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.backbone = backbone.lower()

        self.checkpoint_validator = ValidatorType(checkpoint_validator)
        self.checkpoint_save_interval = checkpoint_save_interval
        self.use_early_stopping = use_early_stopping
        self.da_warmup_epochs = da_warmup_epochs
        self.use_class_weights = use_class_weights
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler.lower()
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.lr_base = None  # Will be set during optimizer creation
        self.global_iter = 0  # For inv scheduler (per-iteration counter)
        self.total_iters = 1  # For inv scheduler (set per-run before the epoch loop)
        self.daarda_divergence = daarda_divergence
        self.daarda_relax = daarda_relax
        self.daarda_grad_penalty = daarda_grad_penalty

        # Split mode: 2-split (train/eval) vs 3-split (train/val/test)
        # 2-split recommended when data is limited - merges val+test into single eval set
        self.two_splits_source = two_splits_source
        self.two_splits_target = two_splits_target
        if two_splits_source:
            self.logger.info("SOURCE: Using 2-split mode (85% train, 15% eval)")
        if two_splits_target:
            self.logger.info("TARGET: Using 2-split mode (85% train, 15% eval)")

        assert self.lr_scheduler_type in ["none", "step", "cosine", "inv"], (
            f"Invalid LR scheduler: {lr_scheduler}. Must be 'none', 'step', 'cosine', or 'inv'"
        )

        if upper_bound:
            self.da_method = "none"
            self.logger.info(
                "UPPER BOUND MODE: Training supervised on source+target combined"
            )
        else:
            self.da_method = da_method.lower()
            assert self.da_method in [
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
            ], (
                f"Invalid DA method: {da_method}. Must be 'coral', 'entropy', 'hybrid', 'mmd', 'dann', 'mcd', 'mcc', 'bnm', 'daarda', or 'none'"
            )

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.mcd_loss_fn = None  # Will be set during MCD training
        self.dann_loss_fn = None  # Will be set during DANN training
        self.daarda_loss_fn = None  # Will be set during DAARDA training
        self.daarda_d_optimizer = None  # Separate discriminator optimizer for DAARDA
        self._samples_used_for_training = {"source": 0, "target": 0}
        self._training_mode = None

    def _create_optimizer_and_scheduler(
        self, parameters, learning_rate: float, num_epochs: int
    ):
        """Create optimizer with weight decay and optional LR scheduler."""
        self.lr_base = learning_rate

        # For 'inv' scheduler, create parameter groups with lr_mult and decay_mult
        if self.lr_scheduler_type == "inv":
            # Convert parameters to list if needed
            param_list = (
                list(parameters) if not isinstance(parameters, list) else parameters
            )
            # All params use same multiplier (can be customized later for backbone vs classifier)
            param_groups = [{"params": param_list, "lr_mult": 1.0, "decay_mult": 1.0}]
            self.optimizer = optim.Adam(
                param_groups, lr=learning_rate, weight_decay=self.weight_decay
            )
            self.scheduler = "inv"  # String marker for per-iteration scheduling
            self.global_iter = 0
            self.total_iters = 1  # updated before the epoch loop starts
            self.logger.info(
                f"Using Inverse LR scheduler: gamma=10, power=0.75 (per-iteration, "
                "p = iter / total_iters, Ganin 2016 convention)"
            )
        else:
            self.optimizer = optim.Adam(
                parameters, lr=learning_rate, weight_decay=self.weight_decay
            )

            if self.lr_scheduler_type == "step":
                self.scheduler = StepLR(
                    self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
                )
                self.logger.info(
                    f"Using StepLR scheduler: step_size={self.lr_step_size}, gamma={self.lr_gamma}"
                )
            elif self.lr_scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
                )
                self.logger.info(
                    f"Using CosineAnnealingLR scheduler: T_max={num_epochs}"
                )
            else:
                self.scheduler = None

        if self.weight_decay > 0:
            self.logger.info(f"Using weight decay: {self.weight_decay}")

    def _clinical_columns_for_domain(self, column_map: dict[str, str]) -> list[str]:
        """Resolve marksheet columns needed for the configured clinical features."""
        columns: list[str] = []
        for feature_name in self.clinical_features:
            column = column_map.get(
                feature_name,
                self.CLINICAL_DEFAULT_COLUMN_MAP.get(feature_name, feature_name),
            )
            columns.append(column)
        return list(dict.fromkeys(columns))

    def _clone_config_with_extra_columns(
        self,
        config: Optional[DatasetConfig],
        extra_columns: Optional[list[str]] = None,
    ) -> Optional[DatasetConfig]:
        """Create an isolated DatasetConfig copy with merged extra metadata columns."""
        if config is None:
            return None

        cloned = DatasetConfig(**config.__dict__)
        merged = list(cloned.extra_columns or [])
        if extra_columns:
            merged.extend(extra_columns)
            merged = list(dict.fromkeys(merged))
        cloned.extra_columns = merged if merged else None
        return cloned

    def create_datasets(
        self,
        source_size: int,
        target_size: int,
        target_train_indices=None,
        target_val_indices=None,
    ) -> DataSplits:
        """Create all data splits including source_test."""
        source_config = self._clone_config_with_extra_columns(
            self.source_dataset_config,
            self._clinical_columns_for_domain(self.source_clinical_column_map)
            if self.use_clinical
            else None,
        )
        target_config = self._clone_config_with_extra_columns(
            self.target_dataset_config,
            self._clinical_columns_for_domain(self.target_clinical_column_map)
            if self.use_clinical
            else None,
        )

        source_marksheet = (
            str(source_config.marksheet_path)
            if source_config is not None
            else "input/picai_labels/clinical_information/marksheet.csv"
        )
        source_data_dir = (
            str(
                source_config.preprocessed_dir
                if self.use_preprocessed
                else source_config.data_dir
            )
            if source_config is not None
            else self.data_dir
        )

        target_marksheet = (
            str(target_config.marksheet_path)
            if target_config is not None
            else "input/picai_labels/clinical_information/marksheet.csv"
        )
        target_data_dir = (
            str(
                target_config.preprocessed_dir
                if self.use_preprocessed
                else target_config.data_dir
            )
            if target_config is not None
            else self.data_dir
        )

        source_prior_cache_dir = (
            self.prostate_prior_cache_dir / "source"
            if self.prostate_prior_cache_dir
            else None
        )
        target_prior_cache_dir = (
            self.prostate_prior_cache_dir / "target"
            if self.prostate_prior_cache_dir
            else None
        )

        source_dataset = ISUPCenterDataset(
            center=self.source_center,
            marksheet_path=source_marksheet,
            data_dir=source_data_dir,
            use_preprocessed=self.use_preprocessed,
            binary_classification=self.binary_classification,
            dataset_config=source_config,
            use_clinical=self.use_clinical,
            clinical_feature_names=self.clinical_features,
            clinical_column_map=self.source_clinical_column_map,
            clinical_impute=self.clinical_impute,
            clinical_missing_indicators=self.clinical_missing_indicators,
            use_prostate_prior=self.use_prostate_prior,
            prostate_prior_type=self.prostate_prior_type,
            prostate_prior_dir=self.prostate_prior_source_dir,
            prostate_prior_soft=False,
            prostate_prior_cache_dir=source_prior_cache_dir,
            prostate_prior_strength=self.prostate_prior_strength,
            prostate_prior_conf_thresh=self.prostate_prior_conf_thresh,
        )

        target_dataset = ISUPCenterDataset(
            center=self.target_center,
            marksheet_path=target_marksheet,
            data_dir=target_data_dir,
            use_preprocessed=self.use_preprocessed,
            binary_classification=self.binary_classification,
            dataset_config=target_config,
            use_clinical=self.use_clinical,
            clinical_feature_names=self.clinical_features,
            clinical_column_map=self.target_clinical_column_map,
            clinical_impute=self.clinical_impute,
            clinical_missing_indicators=self.clinical_missing_indicators,
            use_prostate_prior=self.use_prostate_prior,
            prostate_prior_type=self.prostate_prior_type,
            prostate_prior_dir=self.prostate_prior_target_dir
            if self.prostate_prior_target_mode == "pseudo"
            else None,
            prostate_prior_soft=self.prostate_prior_target_mode == "pseudo",
            prostate_prior_cache_dir=target_prior_cache_dir,
            prostate_prior_strength=self.prostate_prior_strength,
            prostate_prior_conf_thresh=self.prostate_prior_conf_thresh,
        )

        if self.use_clinical:
            source_dim = source_dataset.clinical_feature_dim
            target_dim = target_dataset.clinical_feature_dim
            if source_dim != target_dim:
                raise ValueError(
                    f"Clinical feature dimension mismatch: source={source_dim}, target={target_dim}. "
                    "Ensure both domains expose the same feature set and missing-indicator settings."
                )
            self.clinical_feature_dim = source_dim
            self.logger.info(
                "Clinical branch enabled: features=%s, dim=%d, fusion=%s, impute=%s, missing_indicators=%s",
                self.clinical_features,
                self.clinical_feature_dim,
                self.clinical_fusion,
                self.clinical_impute,
                self.clinical_missing_indicators,
            )
        else:
            self.clinical_feature_dim = 0

        if self.use_prostate_prior:
            self.logger.info(
                "Prostate prior enabled: type=%s, input_channels=%d, source_dir=%s, target_mode=%s",
                self.prostate_prior_type,
                self.input_channels,
                self.prostate_prior_source_dir,
                self.prostate_prior_target_mode,
            )

        if source_size != -1 and source_size < len(source_dataset):
            indices = np.random.choice(len(source_dataset), source_size, replace=False)
            source_dataset = Subset(source_dataset, indices.tolist())

        if target_size != -1 and target_size < len(target_dataset):
            indices = np.random.choice(len(target_dataset), target_size, replace=False)
            target_dataset = Subset(target_dataset, indices.tolist())

        # Single seeded RNG for all dataset splits — ensures reproducibility
        # while avoiding systematic label bias from sequential patient ordering
        split_rng = np.random.RandomState(42)

        # Split source: 2-split (85/15) or 3-split (70/15/15)
        if self.two_splits_source:
            source_splits = self._split_dataset_two(
                source_dataset, train=0.85, eval_=0.15, name="source", rng=split_rng
            )
        else:
            source_splits = self._split_dataset(
                source_dataset,
                train=0.7,
                val=0.15,
                test=0.15,
                name="source",
                rng=split_rng,
            )

        # Split target: external fold indices, 2-split (85/15), or 3-split (70/15/15)
        if target_train_indices is not None and target_val_indices is not None:
            target_splits = {
                "train": Subset(target_dataset, target_train_indices),
                "val": Subset(target_dataset, target_val_indices),
            }
        elif self.two_splits_target:
            target_splits = self._split_dataset_two(
                target_dataset, train=0.85, eval_=0.15, name="target", rng=split_rng
            )
        else:
            target_splits = self._split_dataset(
                target_dataset,
                train=0.7,
                val=0.15,
                test=0.15,
                name="target",
                rng=split_rng,
            )

        return DataSplits(
            source_train=source_splits["train"],
            source_val=source_splits["val"],
            source_test=source_splits.get("test"),  # None in 2-split mode
            target_train=target_splits["train"],
            target_val=target_splits["val"],
            target_test=target_splits.get("test"),  # None in 2-split mode
        )

    def _split_dataset(
        self,
        dataset,
        train: float,
        val: float,
        test: float,
        name: str,
        rng: np.random.RandomState = None,
    ) -> dict:
        """Split a dataset into train/val/test with shuffled indices.

        Args:
            rng: Seeded RandomState for reproducible shuffling. If None, creates one with seed 42.
        """
        total = len(dataset)

        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.permutation(total).tolist()

        if total <= 5:
            return {
                "train": Subset(dataset, indices[: max(1, total - 2)]),
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

    def _split_dataset_two(
        self,
        dataset,
        train: float,
        eval_: float,
        name: str,
        rng: np.random.RandomState = None,
    ) -> dict:
        """Split dataset into train/eval (2-split mode) with shuffled indices.

        Use this when data is limited - the eval set serves as both validation
        (for checkpoint selection) AND test set (for final evaluation).

        Args:
            rng: Seeded RandomState for reproducible shuffling. If None, creates one with seed 42.
        """
        total = len(dataset)

        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.permutation(total).tolist()

        if total <= 3:
            return {
                "train": Subset(dataset, indices[: max(1, total - 1)]),
                "val": Subset(dataset, [indices[total - 1]]),
            }

        train_end = int(train * total)
        train_end = max(1, train_end)

        return {
            "train": Subset(dataset, indices[:train_end]),
            "val": Subset(dataset, indices[train_end:]),
            # Note: no 'test' key - val serves as both val and test
        }

    def _extract_labels_from_subset(self, subset: Subset) -> np.ndarray:
        """Extract labels from a Subset without loading MRI data."""
        dataset = subset.dataset
        indices = list(subset.indices)
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset
        return np.array([dataset.isup_labels[i] for i in indices])

    def _compute_class_weights(self, subset: Subset) -> torch.Tensor:
        """Compute inverse-frequency class weights from source training labels."""
        labels = self._extract_labels_from_subset(subset)
        counts = np.bincount(labels, minlength=self.num_classes)
        weights = np.zeros(self.num_classes, dtype=np.float32)
        nonzero = counts > 0
        weights[nonzero] = len(labels) / (self.num_classes * counts[nonzero])
        self.logger.info(f"Source train class counts: {counts.tolist()}")
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _clinical_tensor_from_batch(self, batch: dict) -> Optional[torch.Tensor]:
        """Extract optional clinical tensor and move to active device."""
        if not self.use_clinical:
            return None

        clinical_tensor = batch.get("clinical_features")
        if clinical_tensor is None:
            raise KeyError(
                "Clinical branch is enabled, but batch is missing 'clinical_features'. "
                "Check dataset creation and model_variant/use_clinical flags."
            )
        return clinical_tensor.to(self.device)

    def _augment_batch_gpu(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply MRI intensity augmentations directly on GPU for a batch.

        Augmentations are applied only to the MRI channels (first 3 channels),
        while optional non-image channels (e.g. prostate priors) are left unchanged.
        """
        cfg = self.augmentation_config
        if cfg is None or batch.ndim != 5:
            return batch

        image_channels = min(3, batch.shape[1])
        if image_channels <= 0:
            return batch

        mri = batch[:, :image_channels]
        augmented = mri
        batch_size = mri.shape[0]
        spatial_shape = tuple(mri.shape[2:])

        def _sample_mask(prob: float) -> Optional[torch.Tensor]:
            if prob <= 0:
                return None
            mask = torch.rand(batch_size, device=batch.device) < prob
            if not torch.any(mask):
                return None
            return mask.view(batch_size, 1, 1, 1, 1)

        def _ensure_writable() -> None:
            nonlocal augmented
            if augmented is mri:
                augmented = mri.clone()

        def _gaussian_kernel1d(sigma: float) -> torch.Tensor:
            sigma = max(sigma, 1e-6)
            # Match scipy.ndimage.gaussian_filter default truncate=4.0.
            radius = max(1, int(4.0 * sigma + 0.5))
            x = torch.arange(
                -radius, radius + 1, device=batch.device, dtype=augmented.dtype
            )
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel

        def _gaussian_blur_3d(data_5d: torch.Tensor, sigma: float) -> torch.Tensor:
            sigma = max(float(sigma), 1e-6)
            sigma_z = max(sigma * 0.5, 1e-6)
            sigma_xy = sigma

            channels = data_5d.shape[1]
            out = data_5d

            kz = _gaussian_kernel1d(sigma_z)
            ky = _gaussian_kernel1d(sigma_xy)
            kx = _gaussian_kernel1d(sigma_xy)

            pad_z = kz.numel() // 2
            pad_y = ky.numel() // 2
            pad_x = kx.numel() // 2

            wz = kz.view(1, 1, -1, 1, 1).repeat(channels, 1, 1, 1, 1)
            wy = ky.view(1, 1, 1, -1, 1).repeat(channels, 1, 1, 1, 1)
            wx = kx.view(1, 1, 1, 1, -1).repeat(channels, 1, 1, 1, 1)

            out = F.pad(out, (0, 0, 0, 0, pad_z, pad_z), mode="reflect")
            out = F.conv3d(out, wz, groups=channels)

            out = F.pad(out, (0, 0, pad_y, pad_y, 0, 0), mode="reflect")
            out = F.conv3d(out, wy, groups=channels)

            out = F.pad(out, (pad_x, pad_x, 0, 0, 0, 0), mode="reflect")
            out = F.conv3d(out, wx, groups=channels)

            return out

        if cfg.gaussian_noise:
            mask = _sample_mask(float(cfg.gaussian_noise_prob))
            if mask is not None:
                _ensure_writable()
                std = float(cfg.gaussian_noise_std)
                if cfg.per_channel:
                    noise = torch.randn_like(augmented) * std
                else:
                    noise = (
                        torch.randn(
                            (batch_size, image_channels, *spatial_shape),
                            device=batch.device,
                            dtype=augmented.dtype,
                        )
                        * std
                    )
                augmented = augmented + noise * mask.to(dtype=augmented.dtype)

        if cfg.brightness:
            mask = _sample_mask(float(cfg.brightness_prob))
            if mask is not None:
                _ensure_writable()
                low, high = cfg.brightness_range
                if cfg.per_channel:
                    shift = torch.empty(
                        (batch_size, image_channels, 1, 1, 1),
                        device=batch.device,
                        dtype=augmented.dtype,
                    ).uniform_(float(low), float(high))
                else:
                    shift = torch.empty(
                        (batch_size, 1, 1, 1, 1),
                        device=batch.device,
                        dtype=augmented.dtype,
                    ).uniform_(float(low), float(high))
                augmented = augmented + shift * mask.to(dtype=augmented.dtype)

        if cfg.contrast:
            mask = _sample_mask(float(cfg.contrast_prob))
            if mask is not None:
                _ensure_writable()
                low, high = cfg.contrast_range
                if cfg.per_channel:
                    scale = torch.empty(
                        (batch_size, image_channels, 1, 1, 1),
                        device=batch.device,
                        dtype=augmented.dtype,
                    ).uniform_(float(low), float(high))
                else:
                    scale = torch.empty(
                        (batch_size, 1, 1, 1, 1),
                        device=batch.device,
                        dtype=augmented.dtype,
                    ).uniform_(float(low), float(high))

                mean = augmented.mean(dim=(2, 3, 4), keepdim=True)
                contrasted = mean + scale * (augmented - mean)
                augmented = torch.where(mask, contrasted, augmented)

        if cfg.blur:
            mask = _sample_mask(float(cfg.blur_prob))
            if mask is not None:
                _ensure_writable()
                low, high = cfg.blur_sigma_range
                active_indices = torch.where(mask.view(batch_size))[0].tolist()
                for idx in active_indices:
                    sigma = float(
                        torch.empty(1, device=batch.device)
                        .uniform_(float(low), float(high))
                        .item()
                    )
                    augmented[idx : idx + 1] = _gaussian_blur_3d(
                        augmented[idx : idx + 1], sigma
                    )

        if augmented is mri:
            return batch

        if image_channels == batch.shape[1]:
            return augmented

        out = batch.clone()
        out[:, :image_channels] = augmented
        return out

    def _forward_model(
        self, data: torch.Tensor, clinical_features: Optional[torch.Tensor] = None
    ) -> dict:
        """Forward wrapper keeping baseline path untouched when clinical is disabled."""
        if self.use_clinical:
            return self.model(data, clinical_features=clinical_features)
        return self.model(data)

    def train(
        self,
        source_size: int,
        target_size: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        da_weight: float,
        target_train_indices=None,
        target_val_indices=None,
    ) -> dict:
        """Main training entry point."""
        self.logger.info(f"Creating datasets...")
        self.logger.info(f"Checkpoint validator: {self.checkpoint_validator.value}")
        self.logger.info(
            f"Checkpoint save interval: every {self.checkpoint_save_interval} epochs"
        )

        splits = self.create_datasets(
            source_size,
            target_size,
            target_train_indices=target_train_indices,
            target_val_indices=target_val_indices,
        )

        # Log split sizes (test is None in 2-split mode)
        src_test_str = (
            str(len(splits.source_test)) if splits.source_test else "N/A (2-split)"
        )
        tgt_test_str = (
            str(len(splits.target_test)) if splits.target_test else "N/A (2-split)"
        )
        self.logger.info(
            f"Source: train={len(splits.source_train)}, val={len(splits.source_val)}, test={src_test_str}"
        )
        self.logger.info(
            f"Target: train={len(splits.target_train)}, val={len(splits.target_val)}, test={tgt_test_str}"
        )

        analyze_splits(
            splits,
            binary_classification=self.binary_classification,
            source_center=self.source_center,
            target_center=self.target_center,
        )

        if self.upper_bound:
            return self._train_upper_bound(
                splits, num_epochs, batch_size, learning_rate
            )
        else:
            return self._train_uda(
                splits, num_epochs, batch_size, learning_rate, da_weight
            )

    def _create_model(self):
        """Create and return the model."""
        model = create_model(
            backbone=self.backbone,
            num_channels=self.input_channels,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            use_batchnorm=self.use_batchnorm,
            clinical_feature_dim=self.clinical_feature_dim,
            clinical_fusion=self.clinical_fusion,
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            "Model config: backbone=%s, variant=%s, channels=%d, clinical_dim=%d, fusion=%s, dropout=%s, batchnorm=%s",
            self.backbone,
            self.model_variant,
            self.input_channels,
            self.clinical_feature_dim,
            self.clinical_fusion if self.use_clinical else "disabled",
            self.dropout_rate,
            self.use_batchnorm,
        )
        self.logger.info(
            f"Model params: total={total_params:,}, trainable={trainable_params:,}"
        )
        return model

    def _train_upper_bound(
        self, splits: DataSplits, num_epochs: int, batch_size: int, learning_rate: float
    ) -> dict:
        """UPPER BOUND: Fully supervised training on source+target combined."""
        self._training_mode = "upper_bound"

        # Keep DataLoader datasets unaugmented and apply augmentations on-GPU in the train step.
        combined_train = torch.utils.data.ConcatDataset(
            [splits.source_train, splits.target_train]
        )
        self._samples_used_for_training["source"] = len(splits.source_train)
        self._samples_used_for_training["target"] = len(splits.target_train)

        self.logger.info(
            f"UPPER BOUND: Combined train = {len(splits.source_train)} (source) + {len(splits.target_train)} (target) = {len(combined_train)}"
        )

        assert self._samples_used_for_training["target"] > 0, (
            "Target samples must be included!"
        )
        assert self._samples_used_for_training["source"] > 0, (
            "Source samples must be included!"
        )

        # num_workers=0: dataset is pre-cached in RAM, so workers only duplicate memory on Windows
        _nw = 0
        train_loader = DataLoader(
            combined_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=True,
        )
        source_train_loader = DataLoader(
            splits.source_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        source_val_loader = DataLoader(
            splits.source_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        target_train_loader = DataLoader(
            splits.target_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        target_val_loader = DataLoader(
            splits.target_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )

        # In 2-split mode, use val set as test set (None -> use val loader)
        source_test_loader = (
            DataLoader(
                splits.source_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_nw,
                pin_memory=True,
            )
            if splits.source_test
            else source_val_loader
        )
        target_test_loader = (
            DataLoader(
                splits.target_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_nw,
                pin_memory=True,
            )
            if splits.target_test
            else target_val_loader
        )

        output_loaders = {
            "source_train": source_train_loader,
            "source_val": source_val_loader,
            "target_train": target_train_loader,
            "target_val": target_val_loader,
        }

        self.model = self._create_model()
        self._create_optimizer_and_scheduler(
            self.model.parameters(), learning_rate, num_epochs
        )
        class_weights = None
        if self.use_class_weights:
            class_weights = self._compute_class_weights(splits.source_train)
            self.logger.info(f"Using dynamic class weights: {class_weights.tolist()}")
        class_loss_fn = ISUPLoss(
            num_classes=self.num_classes, class_weights=class_weights
        )

        checkpoint_manager = CheckpointManager(
            self.workdir,
            save_interval=self.checkpoint_save_interval,
            keep_all=True,
            save_outputs=True,
        )

        writer = SummaryWriter(self.workdir / "tensorboard")
        history = self._init_history()

        patience, min_epochs = 20, 20
        epochs_no_improve = 0

        # Ganin 2016 inv-decay scheduler needs the total number of iterations
        # to normalise progress p = iter / total_iters into [0, 1].
        if self.scheduler == "inv":
            self.total_iters = max(1, num_epochs * len(train_loader))
            self.global_iter = 0

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch_supervised(train_loader, class_loss_fn)
            source_val_metrics = self.validate(
                source_val_loader, class_loss_fn, is_target=False
            )
            target_val_metrics = self.validate(
                target_val_loader, class_loss_fn, is_target=True
            )

            # Step per-epoch schedulers (not inv, which is per-iteration)
            if self.scheduler is not None and self.scheduler != "inv":
                self.scheduler.step()
            if self.optimizer is not None:
                writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            self._log_metrics(
                writer, epoch, train_metrics, source_val_metrics, target_val_metrics
            )
            self._update_history(
                history, train_metrics, source_val_metrics, target_val_metrics
            )

            source_bal_acc = source_val_metrics.get("balanced_accuracy", 0.0)
            target_bal_acc = target_val_metrics.get("balanced_accuracy", 0.0)

            should_save = checkpoint_manager.should_save(epoch, num_epochs)
            is_new_best = False
            if not should_save and epoch >= 10:
                old_best = checkpoint_manager.get_best_checkpoint(
                    self.checkpoint_validator
                )
                old_best_val = (
                    old_best.target_val_bal_acc
                    if self.checkpoint_validator == ValidatorType.TARGET_VAL
                    else old_best.source_val_bal_acc
                    if old_best
                    else -1
                )
                current_val = (
                    target_bal_acc
                    if self.checkpoint_validator == ValidatorType.TARGET_VAL
                    else source_bal_acc
                )
                is_new_best = current_val > old_best_val

            if should_save or is_new_best:
                outputs = self.collect_all_outputs(output_loaders)
                # Supervised training: no aux_state (no MCD/DANN classifiers)
                checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model.state_dict(),
                    source_bal_acc,
                    target_bal_acc,
                    outputs,
                    aux_state=None,
                )

            current_metric = (
                target_bal_acc
                if self.checkpoint_validator == ValidatorType.TARGET_VAL
                else source_bal_acc
            )
            best_ckpt = checkpoint_manager.get_best_checkpoint(
                self.checkpoint_validator
            )
            best_metric = (
                best_ckpt.target_val_bal_acc
                if self.checkpoint_validator == ValidatorType.TARGET_VAL
                else best_ckpt.source_val_bal_acc
                if best_ckpt
                else -1
            )

            if current_metric > best_metric - 0.001:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self._log_epoch_summary(
                    epoch,
                    num_epochs,
                    train_metrics,
                    source_val_metrics,
                    target_val_metrics,
                )

            if (
                self.use_early_stopping
                and epoch >= min_epochs
                and epochs_no_improve >= patience
            ):
                self.logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch: {best_ckpt.epoch if best_ckpt else 'N/A'}"
                )
                break

        writer.close()

        best_ckpt = checkpoint_manager.get_best_checkpoint(self.checkpoint_validator)
        if best_ckpt:
            self.model.load_state_dict(best_ckpt.model_state)
            self.logger.info(
                f"Loaded best checkpoint from epoch {best_ckpt.epoch} (validator={self.checkpoint_validator.value})"
            )

        results = self._final_evaluation(
            source_test_loader,
            target_test_loader,
            class_loss_fn,
            splits,
            history,
            checkpoint_manager,
        )
        results["mode"] = "upper_bound"
        results["combined_train_size"] = len(combined_train)

        return results

    def _train_uda(
        self,
        splits: DataSplits,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        da_weight: float,
    ) -> dict:
        """Standard UDA training (lower bound when da_method='none')."""
        self._training_mode = "lower_bound" if self.da_method == "none" else "uda"

        self._samples_used_for_training["source"] = len(splits.source_train)
        self._samples_used_for_training["target"] = (
            0 if self.da_method == "none" else len(splits.target_train)
        )

        if self.da_method == "none":
            assert da_weight < 0.001, (
                f"da_weight must be ~0 for lower bound, got {da_weight}"
            )

        # Keep DataLoader datasets unaugmented and apply augmentations on-GPU in the train step.
        source_train = splits.source_train
        target_train = splits.target_train

        # num_workers=0: dataset is pre-cached in RAM, so workers only duplicate memory on Windows
        _nw = 0
        source_train_loader = DataLoader(
            source_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=True,
        )
        target_train_loader = DataLoader(
            target_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=True,
        )
        source_train_loader_eval = DataLoader(
            splits.source_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        source_val_loader = DataLoader(
            splits.source_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        target_train_loader_eval = DataLoader(
            splits.target_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )
        target_val_loader = DataLoader(
            splits.target_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=True,
        )

        # In 2-split mode, use val set as test set (None -> use val loader)
        source_test_loader = (
            DataLoader(
                splits.source_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_nw,
                pin_memory=True,
            )
            if splits.source_test
            else source_val_loader
        )
        target_test_loader = (
            DataLoader(
                splits.target_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_nw,
                pin_memory=True,
            )
            if splits.target_test
            else target_val_loader
        )

        output_loaders = {
            "source_train": source_train_loader_eval,
            "source_val": source_val_loader,
            "target_train": target_train_loader_eval,
            "target_val": target_val_loader,
        }

        self.logger.info(f"DA method: {self.da_method.upper()}, da_weight: {da_weight}")
        if self.da_warmup_epochs > 0:
            self.logger.info(
                f"DA warmup: {self.da_warmup_epochs} epochs (CE-only), then scheduled da_weight"
            )

        self.model = self._create_model()

        class_weights = None
        if self.use_class_weights:
            class_weights = self._compute_class_weights(splits.source_train)
            self.logger.info(f"Using dynamic class weights: {class_weights.tolist()}")

        class_loss_fn = ISUPLoss(
            num_classes=self.num_classes, class_weights=class_weights
        )
        coral_loss_fn = CORALLoss()
        entropy_loss_fn = EntropyLoss()
        mmd_loss_fn = MMDLoss()
        mcc_loss_fn = MCCLoss(temperature=2.5)  # MCC uses target logits only
        bnm_loss_fn = (
            BNMLoss()
        )  # BNM uses target logits only (Information Maximization category)
        dann_loss_fn = (
            DANNLoss(feature_dim=512).to(self.device)
            if self.da_method == "dann"
            else None
        )
        mcd_loss_fn = (
            MCDLoss(feature_dim=512, num_classes=self.num_classes).to(self.device)
            if self.da_method == "mcd"
            else None
        )
        daarda_loss_fn = (
            DAARDALoss(
                feature_dim=512,
                divergence=self.daarda_divergence,
                relax=self.daarda_relax,
                grad_penalty=self.daarda_grad_penalty,
            ).to(self.device)
            if self.da_method == "daarda"
            else None
        )
        self.mcd_loss_fn = mcd_loss_fn  # Store for use in validate/collect_outputs
        self.dann_loss_fn = dann_loss_fn  # Store for checkpoint saving
        self.daarda_loss_fn = daarda_loss_fn  # Store for checkpoint saving

        if dann_loss_fn is not None:
            self._create_optimizer_and_scheduler(
                list(self.model.parameters()) + list(dann_loss_fn.parameters()),
                learning_rate,
                num_epochs,
            )
        elif mcd_loss_fn is not None:
            self._create_optimizer_and_scheduler(
                list(self.model.parameters()) + list(mcd_loss_fn.parameters()),
                learning_rate,
                num_epochs,
            )
        elif daarda_loss_fn is not None:
            # Feature extractor/classifier optimizer
            self._create_optimizer_and_scheduler(
                self.model.parameters(), learning_rate, num_epochs
            )
            # Domain discriminator optimizer (separate adversarial step)
            self.daarda_d_optimizer = optim.Adam(
                daarda_loss_fn.domain_discriminator.parameters(),
                lr=learning_rate,
                weight_decay=self.weight_decay,
            )
            self.logger.info(
                f"DAARDA config: divergence={self.daarda_divergence}, "
                f"relax={self.daarda_relax}, grad_penalty={self.daarda_grad_penalty}"
            )
        else:
            self._create_optimizer_and_scheduler(
                self.model.parameters(), learning_rate, num_epochs
            )
            self.daarda_d_optimizer = None

        checkpoint_manager = CheckpointManager(
            self.workdir,
            save_interval=self.checkpoint_save_interval,
            keep_all=True,
            save_outputs=True,
        )

        writer = SummaryWriter(self.workdir / "tensorboard")
        history = self._init_history()

        patience, min_epochs = 20, 20
        epochs_no_improve = 0

        # Ganin 2016 inv-decay scheduler needs the total number of iterations
        # to normalise progress p = iter / total_iters into [0, 1].
        if self.scheduler == "inv":
            self.total_iters = max(1, num_epochs * len(source_train_loader))
            self.global_iter = 0

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch_uda(
                source_train_loader,
                target_train_loader,
                class_loss_fn,
                coral_loss_fn,
                entropy_loss_fn,
                mmd_loss_fn,
                dann_loss_fn,
                da_weight,
                epoch,
                num_epochs,
                mcd_loss_fn=mcd_loss_fn,
                mcc_loss_fn=mcc_loss_fn,
                bnm_loss_fn=bnm_loss_fn,
                daarda_loss_fn=daarda_loss_fn,
            )
            source_val_metrics = self.validate(
                source_val_loader, class_loss_fn, is_target=False
            )
            target_val_metrics = self.validate(
                target_val_loader, class_loss_fn, is_target=True
            )

            # Step per-epoch schedulers (not inv, which is per-iteration)
            if self.scheduler is not None and self.scheduler != "inv":
                self.scheduler.step()
            if self.optimizer is not None:
                writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            self._log_metrics(
                writer, epoch, train_metrics, source_val_metrics, target_val_metrics
            )
            self._update_history(
                history, train_metrics, source_val_metrics, target_val_metrics
            )

            source_bal_acc = source_val_metrics.get("balanced_accuracy", 0.0)
            target_bal_acc = target_val_metrics.get("balanced_accuracy", 0.0)

            should_save = checkpoint_manager.should_save(epoch, num_epochs)
            is_new_best = False
            if not should_save and epoch >= 10:
                old_best = checkpoint_manager.get_best_checkpoint(
                    self.checkpoint_validator
                )
                old_best_val = (
                    old_best.target_val_bal_acc
                    if self.checkpoint_validator == ValidatorType.TARGET_VAL
                    else old_best.source_val_bal_acc
                    if old_best
                    else -1
                )
                current_val = (
                    target_bal_acc
                    if self.checkpoint_validator == ValidatorType.TARGET_VAL
                    else source_bal_acc
                )
                is_new_best = current_val > old_best_val

            if should_save or is_new_best:
                outputs = self.collect_all_outputs(output_loaders)
                # Build auxiliary state for MCD/DANN classifiers
                aux_state = None
                if self.mcd_loss_fn is not None:
                    aux_state = {"mcd": self.mcd_loss_fn.state_dict()}
                if self.dann_loss_fn is not None:
                    aux_state = aux_state or {}
                    aux_state["dann"] = self.dann_loss_fn.state_dict()
                if self.daarda_loss_fn is not None:
                    aux_state = aux_state or {}
                    aux_state["daarda"] = self.daarda_loss_fn.state_dict()
                checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model.state_dict(),
                    source_bal_acc,
                    target_bal_acc,
                    outputs,
                    aux_state,
                )

            current_metric = (
                target_bal_acc
                if self.checkpoint_validator == ValidatorType.TARGET_VAL
                else source_bal_acc
            )
            best_ckpt = checkpoint_manager.get_best_checkpoint(
                self.checkpoint_validator
            )
            best_metric = (
                best_ckpt.target_val_bal_acc
                if self.checkpoint_validator == ValidatorType.TARGET_VAL
                else best_ckpt.source_val_bal_acc
                if best_ckpt
                else -1
            )

            if current_metric > best_metric - 0.001:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self._log_epoch_summary(
                    epoch,
                    num_epochs,
                    train_metrics,
                    source_val_metrics,
                    target_val_metrics,
                )

            if (
                self.use_early_stopping
                and epoch >= min_epochs
                and epochs_no_improve >= patience
            ):
                self.logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch: {best_ckpt.epoch if best_ckpt else 'N/A'}"
                )
                break

        writer.close()

        best_ckpt = checkpoint_manager.get_best_checkpoint(self.checkpoint_validator)
        if best_ckpt:
            self.model.load_state_dict(best_ckpt.model_state)
            # Restore MCD classifier state if available
            if (
                best_ckpt.aux_state is not None
                and "mcd" in best_ckpt.aux_state
                and self.mcd_loss_fn is not None
            ):
                self.mcd_loss_fn.load_state_dict(best_ckpt.aux_state["mcd"])
                self.logger.info(f"Loaded MCD classifiers from epoch {best_ckpt.epoch}")
            # Restore DANN domain discriminator state if available
            if (
                best_ckpt.aux_state is not None
                and "dann" in best_ckpt.aux_state
                and self.dann_loss_fn is not None
            ):
                self.dann_loss_fn.load_state_dict(best_ckpt.aux_state["dann"])
                self.logger.info(
                    f"Loaded DANN discriminator from epoch {best_ckpt.epoch}"
                )
            if (
                best_ckpt.aux_state is not None
                and "daarda" in best_ckpt.aux_state
                and self.daarda_loss_fn is not None
            ):
                self.daarda_loss_fn.load_state_dict(best_ckpt.aux_state["daarda"])
                self.logger.info(
                    f"Loaded DAARDA discriminator from epoch {best_ckpt.epoch}"
                )
            self.logger.info(
                f"Loaded best checkpoint from epoch {best_ckpt.epoch} (validator={self.checkpoint_validator.value})"
            )

        results = self._final_evaluation(
            source_test_loader,
            target_test_loader,
            class_loss_fn,
            splits,
            history,
            checkpoint_manager,
        )
        results["mode"] = self._training_mode

        return results

    def _train_epoch_supervised(self, loader, loss_fn) -> dict:
        """Simple supervised training epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            data = batch["data"].to(self.device)
            data = self._augment_batch_gpu(data)
            labels = batch["isup_grade"].to(self.device)
            clinical_features = self._clinical_tensor_from_batch(batch)

            self.optimizer.zero_grad()

            # Inverse LR scheduler update (per iteration), same as UDA training
            if self.scheduler == "inv":
                p = self.global_iter / max(1, self.total_iters)
                inv_lr_scheduler(
                    self.optimizer,
                    p,
                    gamma=10,
                    power=0.75,
                    lr=self.lr_base,
                    weight_decay=self.weight_decay,
                )
                self.global_iter += 1

            outputs = self._forward_model(data, clinical_features)
            loss = loss_fn(outputs["classification"], labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = outputs["classification"].argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(loader),
            "accuracy": 100.0 * correct / total if total > 0 else 0,
        }

    def _compute_scheduled_da_weight(
        self, da_weight_max: float, epoch: int, num_epochs: int
    ) -> float:
        """Compute scheduled da_weight with warmup and gradual increase.

        Schedule:
        - Epochs 0 to warmup: da_weight = 0 (CE-only training)
        - Epochs warmup to end: da_weight gradually increases from 0 to da_weight_max
          using sigmoid schedule: λ(p) = λ_max * (2 / (1 + exp(-10*p)) - 1)
        """
        if epoch < self.da_warmup_epochs:
            return 0.0

        # Progress after warmup (0 to 1)
        remaining_epochs = num_epochs - self.da_warmup_epochs
        if remaining_epochs <= 0:
            return da_weight_max

        p = (epoch - self.da_warmup_epochs) / remaining_epochs
        # Sigmoid schedule (same as DANN alpha)
        scheduled_weight = da_weight_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        return scheduled_weight

    def _train_epoch_uda(
        self,
        source_loader,
        target_loader,
        class_loss_fn,
        coral_loss_fn,
        entropy_loss_fn,
        mmd_loss_fn,
        dann_loss_fn,
        da_weight,
        epoch=0,
        num_epochs=100,
        mcd_loss_fn=None,
        mcc_loss_fn=None,
        bnm_loss_fn=None,
        daarda_loss_fn=None,
    ) -> dict:
        """UDA training epoch with domain adaptation."""
        self.model.train()

        # Compute effective da_weight with warmup and scheduling
        if self.da_warmup_epochs > 0:
            effective_da_weight = self._compute_scheduled_da_weight(
                da_weight, epoch, num_epochs
            )
        else:
            effective_da_weight = da_weight

        if dann_loss_fn is not None:
            dann_loss_fn.train()
            p = epoch / num_epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            dann_loss_fn.set_alpha(alpha, device=self.device)

        if mcd_loss_fn is not None:
            mcd_loss_fn.train()
        if daarda_loss_fn is not None:
            daarda_loss_fn.train()

        total_loss, total_cls_loss, total_da_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0

        target_iter = iter(target_loader)

        for batch_idx, source_batch in enumerate(source_loader):
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            source_data = source_batch["data"].to(self.device)
            source_labels = source_batch["isup_grade"].to(self.device)
            target_data = target_batch["data"].to(self.device)
            source_data = self._augment_batch_gpu(source_data)
            target_data = self._augment_batch_gpu(target_data)
            source_clinical = self._clinical_tensor_from_batch(source_batch)
            target_clinical = self._clinical_tensor_from_batch(target_batch)

            self.optimizer.zero_grad()

            # Inverse LR scheduler update (per iteration)
            if self.scheduler == "inv":
                p = self.global_iter / max(1, self.total_iters)
                inv_lr_scheduler(
                    self.optimizer,
                    p,
                    gamma=10,
                    power=0.75,
                    lr=self.lr_base,
                    weight_decay=self.weight_decay,
                )
                self.global_iter += 1

            source_outputs = self._forward_model(source_data, source_clinical)
            target_outputs = self._forward_model(target_data, target_clinical)

            if batch_idx == 0:
                assert source_outputs["classification"].shape[1] == self.num_classes

            cls_loss = class_loss_fn(source_outputs["classification"], source_labels)

            da_loss = torch.tensor(0.0).to(self.device)

            if self.da_method == "mcd" and mcd_loss_fn is not None:
                # MCD training - always use MCD classifiers (even during warmup)
                logits1, logits2 = mcd_loss_fn(source_outputs["features"])
                cls_loss = mcd_loss_fn.classification_loss(
                    logits1, logits2, source_labels
                )

                if effective_da_weight > 0:
                    # Full MCD 3-step training procedure
                    # Step A: Train feature extractor + classifiers to minimize source classification loss
                    self.optimizer.zero_grad()
                    # Add entropy regularization on target
                    tgt_logits1, tgt_logits2 = mcd_loss_fn(target_outputs["features"])
                    ent_loss = 0.01 * (
                        mcd_loss_fn.entropy_loss(tgt_logits1)
                        + mcd_loss_fn.entropy_loss(tgt_logits2)
                    )
                    step_a_loss = cls_loss + ent_loss
                    step_a_loss.backward()
                    self.optimizer.step()

                    # Step B: Train classifiers to maximize discrepancy (freeze feature extractor)
                    for param in self.model.parameters():
                        param.requires_grad = False
                    for param in mcd_loss_fn.parameters():
                        param.requires_grad = True

                    self.optimizer.zero_grad()
                    with torch.no_grad():
                        source_features = self._forward_model(
                            source_data, source_clinical
                        )["features"]
                        target_features = self._forward_model(
                            target_data, target_clinical
                        )["features"]

                    logits1, logits2 = mcd_loss_fn(source_features)
                    tgt_logits1, tgt_logits2 = mcd_loss_fn(target_features)
                    cls_loss_b = mcd_loss_fn.classification_loss(
                        logits1, logits2, source_labels
                    )
                    discrepancy = mcd_loss_fn.discrepancy(tgt_logits1, tgt_logits2)
                    ent_loss_b = 0.01 * (
                        mcd_loss_fn.entropy_loss(tgt_logits1)
                        + mcd_loss_fn.entropy_loss(tgt_logits2)
                    )
                    step_b_loss = (
                        cls_loss_b + ent_loss_b - effective_da_weight * discrepancy
                    )
                    step_b_loss.backward()
                    self.optimizer.step()

                    # Step C: Train feature extractor to minimize discrepancy (freeze classifiers)
                    for param in self.model.parameters():
                        param.requires_grad = True
                    for param in mcd_loss_fn.parameters():
                        param.requires_grad = False

                    for _ in range(4):  # Multiple generator steps per iteration
                        self.optimizer.zero_grad()
                        target_outputs_c = self._forward_model(
                            target_data, target_clinical
                        )
                        tgt_logits1, tgt_logits2 = mcd_loss_fn(
                            target_outputs_c["features"]
                        )
                        discrepancy = mcd_loss_fn.discrepancy(tgt_logits1, tgt_logits2)
                        step_c_loss = effective_da_weight * discrepancy
                        step_c_loss.backward()
                        self.optimizer.step()

                    # Restore gradients for all parameters
                    for param in mcd_loss_fn.parameters():
                        param.requires_grad = True

                    # Re-compute fresh logits for accurate logging (after all 3 steps)
                    with torch.no_grad():
                        fresh_source_outputs = self._forward_model(
                            source_data, source_clinical
                        )
                        logits1, logits2 = mcd_loss_fn(fresh_source_outputs["features"])
                        cls_loss = mcd_loss_fn.classification_loss(
                            logits1, logits2, source_labels
                        )

                    da_loss = discrepancy.detach()
                    loss = cls_loss.detach() + effective_da_weight * da_loss
                else:
                    # MCD warmup: train G + MCD classifiers on source classification only (no Steps B/C)
                    self.optimizer.zero_grad()
                    cls_loss.backward()
                    self.optimizer.step()
                    loss = cls_loss.detach()

                # Use averaged MCD classifier output for accuracy (consistent with evaluation)
                avg_logits = (logits1 + logits2) / 2
                predicted = avg_logits.argmax(1)
            elif self.da_method == "daarda" and daarda_loss_fn is not None:
                if effective_da_weight > 0:
                    da_loss = daarda_loss_fn.feature_alignment_loss(
                        source_outputs["features"], target_outputs["features"]
                    )

                loss = cls_loss + effective_da_weight * da_loss
                loss.backward()
                self.optimizer.step()

                if effective_da_weight > 0 and self.daarda_d_optimizer is not None:
                    with torch.no_grad():
                        source_features_d = self._forward_model(
                            source_data, source_clinical
                        )["features"]
                        target_features_d = self._forward_model(
                            target_data, target_clinical
                        )["features"]
                    self.daarda_d_optimizer.zero_grad()
                    d_loss = daarda_loss_fn.discriminator_loss(
                        source_features_d, target_features_d
                    )
                    d_loss.backward()
                    self.daarda_d_optimizer.step()

                predicted = source_outputs["classification"].argmax(1)
            else:
                # Standard training for other DA methods
                if effective_da_weight > 0:
                    # Prediction-level DA losses (MCC, BNM, entropy) use
                    # image-only logits so auxiliary modalities (clinical,
                    # future PET) never pollute the adaptation signal.
                    if self.da_method == "coral":
                        da_loss = coral_loss_fn(
                            source_outputs["features"], target_outputs["features"]
                        )
                    elif self.da_method == "entropy":
                        da_loss = entropy_loss_fn(
                            target_outputs["image_classification"]
                        )
                    elif self.da_method == "hybrid":
                        c_loss = coral_loss_fn(
                            source_outputs["features"], target_outputs["features"]
                        )
                        e_loss = entropy_loss_fn(target_outputs["image_classification"])
                        da_loss = (c_loss + e_loss) / 2.0
                    elif self.da_method == "mmd":
                        da_loss = mmd_loss_fn(
                            source_outputs["features"], target_outputs["features"]
                        )
                    elif self.da_method == "dann":
                        da_loss = dann_loss_fn(
                            source_outputs["features"], target_outputs["features"]
                        )
                    elif self.da_method == "mcc":
                        da_loss = mcc_loss_fn(target_outputs["image_classification"])
                    elif self.da_method == "bnm":
                        da_loss = bnm_loss_fn(target_outputs["image_classification"])

                loss = cls_loss + effective_da_weight * da_loss
                loss.backward()
                self.optimizer.step()

                predicted = source_outputs["classification"].argmax(1)

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_da_loss += (
                da_loss.item() if isinstance(da_loss, torch.Tensor) else da_loss
            )

            correct += (predicted == source_labels).sum().item()
            total += source_labels.size(0)

        return {
            "loss": total_loss / len(source_loader),
            "cls_loss": total_cls_loss / len(source_loader),
            "da_loss": total_da_loss / len(source_loader),
            "accuracy": 100.0 * correct / total if total > 0 else 0,
        }

    def collect_outputs(self, loader) -> SampleOutputs:
        """Collect all outputs from a loader for post-hoc validator selection."""
        self.model.eval()
        if hasattr(self, "mcd_loss_fn") and self.mcd_loss_fn is not None:
            self.mcd_loss_fn.eval()
        all_ids, all_logits, all_features = [], [], []
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                data = batch["data"].to(self.device)
                labels = batch["isup_grade"].to(self.device)
                sample_ids = batch.get("patient_id", list(range(len(labels))))
                clinical_features = self._clinical_tensor_from_batch(batch)

                outputs = self._forward_model(data, clinical_features)
                features = outputs["features"]

                # For MCD, use the MCD classifiers instead of model's built-in classifier
                if hasattr(self, "mcd_loss_fn") and self.mcd_loss_fn is not None:
                    logits1, logits2 = self.mcd_loss_fn(features)
                    # Average predictions from both classifiers for evaluation
                    logits = (logits1 + logits2) / 2
                else:
                    logits = outputs["classification"]

                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(1)

                all_ids.extend(
                    sample_ids
                    if isinstance(sample_ids, list)
                    else sample_ids.tolist()
                    if hasattr(sample_ids, "tolist")
                    else list(sample_ids)
                )
                all_logits.append(logits.cpu().numpy())
                all_features.append(features.cpu().numpy())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.append(probs.cpu().numpy())

        return SampleOutputs(
            sample_ids=all_ids,
            logits=np.concatenate(all_logits, axis=0) if all_logits else np.array([]),
            features=np.concatenate(all_features, axis=0)
            if all_features
            else np.array([]),
            predictions=np.array(all_preds),
            labels=np.array(all_labels),
            probabilities=np.concatenate(all_probs, axis=0)
            if all_probs
            else np.array([]),
        )

    def collect_all_outputs(
        self, loaders: Dict[str, DataLoader]
    ) -> Dict[str, SampleOutputs]:
        """Collect outputs from all loaders."""
        return {name: self.collect_outputs(loader) for name, loader in loaders.items()}

    def validate(self, loader, loss_fn, is_target: bool = False) -> dict:
        """Validate model and compute all metrics."""
        self.model.eval()
        if hasattr(self, "mcd_loss_fn") and self.mcd_loss_fn is not None:
            self.mcd_loss_fn.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                data = batch["data"].to(self.device)
                labels = batch["isup_grade"].to(self.device)
                clinical_features = self._clinical_tensor_from_batch(batch)

                outputs = self._forward_model(data, clinical_features)

                # For MCD, use the MCD classifiers instead of model's built-in classifier
                if hasattr(self, "mcd_loss_fn") and self.mcd_loss_fn is not None:
                    features = outputs["features"]
                    logits1, logits2 = self.mcd_loss_fn(features)
                    # Average predictions from both classifiers for evaluation
                    logits = (logits1 + logits2) / 2
                else:
                    logits = outputs["classification"]

                probs = torch.softmax(logits, dim=1)

                if not is_target:
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item()

                predicted = probs.argmax(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.append(probs.cpu().numpy())

        if total == 0:
            return self._empty_metrics()

        return self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.concatenate(all_probs, axis=0) if all_probs else None,
            total_loss / len(loader) if not is_target else 0.0,
            100.0 * correct / total,
        )

    def _compute_metrics(self, y_true, y_pred, y_prob, loss, accuracy) -> dict:
        """Compute all evaluation metrics."""
        labels_full = list(range(self.num_classes))

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100
        macro_precision = (
            precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
        )
        micro_precision = (
            precision_score(y_true, y_pred, average="micro", zero_division=0) * 100
        )
        macro_recall = (
            recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        )
        micro_recall = (
            recall_score(y_true, y_pred, average="micro", zero_division=0) * 100
        )
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100

        cm = confusion_matrix(y_true, y_pred, labels=labels_full)
        sensitivity, specificity = self._compute_sens_spec(cm)

        auc = self._compute_auc(y_true, y_prob)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "micro_precision": micro_precision,
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,
            "auc": auc,
            "balanced_accuracy": balanced_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def _compute_sens_spec(self, cm) -> tuple:
        """Compute sensitivity and specificity from confusion matrix."""
        total_samples = cm.sum()

        if self.num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = (tp / (tp + fn) if (tp + fn) > 0 else 0.0) * 100
            specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0.0) * 100
        else:
            per_class_sens, per_class_spec = [], []
            for i in range(self.num_classes):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = total_samples - tp - fn - fp

                per_class_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                per_class_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

            sensitivity = float(np.mean(per_class_sens)) * 100
            specificity = float(np.mean(per_class_spec)) * 100

        return sensitivity, specificity

    def _compute_auc(self, y_true, y_prob) -> float:
        """Compute AUC score."""
        unique_classes = np.unique(y_true)

        if y_prob is None or len(unique_classes) <= 1:
            return 0.0

        try:
            if self.num_classes == 2:
                return roc_auc_score(y_true, y_prob[:, 1]) * 100
            else:
                return (
                    roc_auc_score(
                        y_true,
                        y_prob[:, unique_classes],
                        multi_class="ovr",
                        average="macro",
                        labels=unique_classes,
                    )
                    * 100
                )
        except ValueError:
            return 0.0

    def _empty_metrics(self) -> dict:
        """Return empty metrics dict."""
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "micro_precision": 0.0,
            "macro_recall": 0.0,
            "micro_recall": 0.0,
            "auc": 0.0,
            "balanced_accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    def _init_history(self) -> dict:
        """Initialize training history."""
        return {
            "train_loss": [],
            "train_acc": [],
            "source_val_acc": [],
            "source_val_bal_acc": [],
            "target_val_acc": [],
            "target_val_bal_acc": [],
        }

    def _update_history(
        self, history, train_metrics, source_val_metrics, target_val_metrics
    ):
        """Update training history."""
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["source_val_acc"].append(source_val_metrics["accuracy"])
        history["source_val_bal_acc"].append(
            source_val_metrics.get("balanced_accuracy", 0)
        )
        history["target_val_acc"].append(target_val_metrics["accuracy"])
        history["target_val_bal_acc"].append(
            target_val_metrics.get("balanced_accuracy", 0)
        )

    def _log_metrics(
        self, writer, epoch, train_metrics, source_val_metrics, target_val_metrics
    ):
        """Log metrics to TensorBoard."""
        for k, v in train_metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in source_val_metrics.items():
            writer.add_scalar(f"SourceVal/{k}", v, epoch)
        for k, v in target_val_metrics.items():
            writer.add_scalar(f"TargetVal/{k}", v, epoch)

    def _log_epoch_summary(
        self, epoch, num_epochs, train_metrics, source_val_metrics, target_val_metrics
    ):
        """Log epoch summary."""
        self.logger.info(
            f"Epoch {epoch:3d}/{num_epochs}: "
            f"TrainL={train_metrics['loss']:.4f}, TrainA={train_metrics['accuracy']:.1f}% | "
            f"SrcValA={source_val_metrics['accuracy']:.1f}%, SrcBalAcc={source_val_metrics.get('balanced_accuracy', 0):.1f}%, SrcAUC={source_val_metrics.get('auc', 0):.1f}% | "
            f"TgtValA={target_val_metrics['accuracy']:.1f}%, TgtBalAcc={target_val_metrics.get('balanced_accuracy', 0):.1f}%, TgtAUC={target_val_metrics.get('auc', 0):.1f}%"
        )

    def _final_evaluation(
        self,
        source_test_loader,
        target_test_loader,
        loss_fn,
        splits: DataSplits,
        history: dict,
        checkpoint_manager: CheckpointManager,
    ) -> dict:
        """Final evaluation on both source and target test sets."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL EVALUATION on held-out test sets")
        self.logger.info("=" * 80)

        source_test_metrics = self.validate(
            source_test_loader, loss_fn, is_target=False
        )
        target_test_metrics = self.validate(target_test_loader, loss_fn, is_target=True)

        self.logger.info(f"\nSOURCE TEST Results:")
        self.logger.info(f"  Accuracy: {source_test_metrics['accuracy']:.2f}%")
        self.logger.info(
            f"  Balanced Accuracy: {source_test_metrics.get('balanced_accuracy', 0):.2f}%"
        )
        self.logger.info(f"  AUC: {source_test_metrics.get('auc', 0):.2f}%")
        self.logger.info(
            f"  Sensitivity: {source_test_metrics.get('sensitivity', 0):.2f}%"
        )
        self.logger.info(
            f"  Specificity: {source_test_metrics.get('specificity', 0):.2f}%"
        )

        self.logger.info(f"\nTARGET TEST Results:")
        self.logger.info(f"  Accuracy: {target_test_metrics['accuracy']:.2f}%")
        self.logger.info(
            f"  Balanced Accuracy: {target_test_metrics.get('balanced_accuracy', 0):.2f}%"
        )
        self.logger.info(f"  AUC: {target_test_metrics.get('auc', 0):.2f}%")
        self.logger.info(
            f"  Sensitivity: {target_test_metrics.get('sensitivity', 0):.2f}%"
        )
        self.logger.info(
            f"  Specificity: {target_test_metrics.get('specificity', 0):.2f}%"
        )

        ckpt_summary = checkpoint_manager.get_summary()
        self.logger.info(f"\nCheckpoint Summary: {ckpt_summary}")

        best_ckpt = checkpoint_manager.get_best_checkpoint(self.checkpoint_validator)

        # Report sizes (in 2-split mode, test_size = val_size since val is used as test)
        source_test_size = (
            len(splits.source_test) if splits.source_test else len(splits.source_val)
        )
        target_test_size = (
            len(splits.target_test) if splits.target_test else len(splits.target_val)
        )

        return {
            "source_train_size": len(splits.source_train),
            "source_val_size": len(splits.source_val),
            "source_test_size": source_test_size,
            "target_train_size": len(splits.target_train),
            "target_val_size": len(splits.target_val),
            "target_test_size": target_test_size,
            "two_splits_source": splits.source_test is None,
            "two_splits_target": splits.target_test is None,
            "model_variant": self.model_variant,
            "use_prostate_prior": self.use_prostate_prior,
            "prostate_prior_type": self.prostate_prior_type
            if self.use_prostate_prior
            else None,
            "prostate_prior_target_mode": self.prostate_prior_target_mode
            if self.use_prostate_prior
            else "none",
            "input_channels": self.input_channels,
            "use_clinical": self.use_clinical,
            "clinical_features": list(self.clinical_features)
            if self.use_clinical
            else [],
            "clinical_feature_dim": self.clinical_feature_dim,
            "clinical_fusion": self.clinical_fusion if self.use_clinical else None,
            "final_source_test_accuracy": source_test_metrics["accuracy"],
            "final_source_test_balanced_accuracy": source_test_metrics.get(
                "balanced_accuracy", 0
            ),
            "final_source_test_auc": source_test_metrics.get("auc", 0),
            "final_source_test_sensitivity": source_test_metrics.get("sensitivity", 0),
            "final_source_test_specificity": source_test_metrics.get("specificity", 0),
            "final_target_test_accuracy": target_test_metrics["accuracy"],
            "final_target_test_balanced_accuracy": target_test_metrics.get(
                "balanced_accuracy", 0
            ),
            "final_target_test_auc": target_test_metrics.get("auc", 0),
            "final_target_test_sensitivity": target_test_metrics.get("sensitivity", 0),
            "final_target_test_specificity": target_test_metrics.get("specificity", 0),
            "final_target_test_macro_f1": target_test_metrics.get("macro_f1", 0),
            "final_target_test_micro_f1": target_test_metrics.get("micro_f1", 0),
            "checkpoint_validator": self.checkpoint_validator.value,
            "best_epoch": best_ckpt.epoch if best_ckpt else None,
            "best_source_val_bal_acc": best_ckpt.source_val_bal_acc
            if best_ckpt
            else None,
            "best_target_val_bal_acc": best_ckpt.target_val_bal_acc
            if best_ckpt
            else None,
            "checkpoint_summary": ckpt_summary,
            "history": history,
        }
