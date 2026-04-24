import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

import SimpleITK as sitk
from picai_prep.preprocessing import Sample, PreprocessingSettings

from training_setup.dataset_config import (
    DatasetConfig,
    find_preprocessed_file,
    load_dataset_metadata,
    normalize_identifier,
    resolve_scan_paths,
    split_center_expression,
)

logger = logging.getLogger(__name__)


class AugmentedSubset(Dataset):
    """Wrapper that applies augmentation to a Subset.

    Used to apply training augmentations only to training splits,
    not to validation/test splits.
    """

    def __init__(self, subset: Subset, transform=None, image_channels: int = 3):
        self.subset = subset
        self.transform = transform
        self.image_channels = image_channels

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]

        if self.transform is not None:
            # Apply intensity augmentation only to MRI channels.
            # Extra channels (e.g., prostate priors) must remain untouched.
            data_np = sample["data"].numpy()
            if data_np.shape[0] <= self.image_channels:
                data_np = self.transform(data_np)
            else:
                augmented = data_np.copy()
                augmented[: self.image_channels] = self.transform(
                    data_np[: self.image_channels]
                )
                data_np = augmented
            sample["data"] = torch.from_numpy(data_np).float()

        return sample


class ISUPCenterDataset(Dataset):
    DEFAULT_CLINICAL_COLUMNS = {
        "psa": "psa",
        "psad": "psad",
        "prostate_volume": "prostate_volume",
        "age": "patient_age",
    }

    def __init__(
        self,
        center,
        marksheet_path,
        data_dir,
        transform=None,
        use_preprocessed=False,
        binary_classification=False,
        dataset_config: DatasetConfig = None,
        use_clinical: bool = False,
        clinical_feature_names: Optional[list[str]] = None,
        clinical_column_map: Optional[dict[str, str]] = None,
        clinical_impute: str = "median",
        clinical_missing_indicators: bool = False,
        use_prostate_prior: bool = False,
        prostate_prior_type: str = "whole_gland",
        prostate_prior_dir: Optional[str | Path] = None,
        prostate_prior_soft: bool = False,
        prostate_prior_cache_dir: Optional[str | Path] = None,
        prostate_prior_strength: float = 1.0,
        prostate_prior_conf_thresh: float = 0.5,
    ):
        self.center = center
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        self.binary_classification = binary_classification

        self.use_clinical = use_clinical
        self.clinical_feature_names = clinical_feature_names or [
            "psa",
            "psad",
            "prostate_volume",
            "age",
        ]
        self.clinical_column_map = clinical_column_map or {}
        self.clinical_impute = clinical_impute
        self.clinical_missing_indicators = clinical_missing_indicators
        self.clinical_features = None
        self.clinical_feature_dim = 0

        self.use_prostate_prior = use_prostate_prior
        self.prostate_prior_type = prostate_prior_type
        self.prostate_prior_dir = (
            Path(prostate_prior_dir) if prostate_prior_dir else None
        )
        self.prostate_prior_soft = prostate_prior_soft
        self.prostate_prior_cache_dir = (
            Path(prostate_prior_cache_dir) if prostate_prior_cache_dir else None
        )
        self.prostate_prior_strength = float(prostate_prior_strength)
        self.prostate_prior_conf_thresh = float(prostate_prior_conf_thresh)
        self.prostate_prior_channels = 0
        self._missing_prior_logged: set[str] = set()
        self._invalid_prior_logged: set[str] = set()
        self._zero_prior_logged: set[str] = set()

        if dataset_config is None:
            dataset_config = DatasetConfig(
                center_alias=str(center),
                marksheet_path=marksheet_path,
                data_dir=data_dir,
                preprocessed_dir=data_dir,
                table_format="csv",
                table_skiprows=0,
                center_column="center",
                center_filter=None,
                patient_id_column="patient_id",
                study_id_column="study_id",
                label_column="case_ISUP",
                label_mode="isup",
                sequence_strategy="public",
            )
        else:
            dataset_config = DatasetConfig(**dataset_config.__dict__)

        if self.use_preprocessed:
            dataset_config.preprocessed_dir = self.data_dir
        else:
            dataset_config.data_dir = self.data_dir

        self.dataset_config = dataset_config

        df = load_dataset_metadata(self.dataset_config)
        selected_centers = split_center_expression(center)
        self.df = df[df["center"].isin(selected_centers)].reset_index(drop=True)

        if self.df.empty:
            available = sorted(df["center"].unique().tolist())
            raise ValueError(
                f"No samples found for center expression '{center}'. "
                f"Available centers: {available}"
            )

        self.patient_ids = self.df["patient_id"].tolist()
        self.study_ids = self.df["study_id"].tolist()
        isup_labels = self.df["case_ISUP"].values

        # Binary: ISUP 0-1 (no csPCa) = 0, ISUP >=2 (csPCa) = 1
        if self.binary_classification:
            self.isup_labels = (isup_labels >= 2).astype(int)
        else:
            self.isup_labels = isup_labels

        if self.use_clinical:
            self.clinical_features = self._prepare_clinical_features()
            self.clinical_feature_dim = self.clinical_features.shape[1]

        if self.use_prostate_prior:
            if self.prostate_prior_type not in {"whole_gland", "zonal", "both"}:
                raise ValueError(
                    f"Invalid prostate_prior_type '{self.prostate_prior_type}'. "
                    "Choose from 'whole_gland', 'zonal', or 'both'."
                )
            self.prostate_prior_channels = self._expected_prostate_prior_channels()
            if self.prostate_prior_cache_dir is not None:
                self.prostate_prior_cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        study_id = self.study_ids[idx]
        isup_grade = self.isup_labels[idx]

        if idx in self._data_cache:
            data = self._data_cache[idx].copy()
        elif self.use_preprocessed:
            data = self.load_preprocessed(patient_id, study_id)
        else:
            data = self.load_mri_sequences(patient_id, study_id)

        if self.use_prostate_prior:
            prior = self.load_prostate_prior(patient_id, study_id)
            if prior is None:
                prior_key = f"{patient_id}_{study_id}"
                if (
                    self.prostate_prior_dir is not None
                    and prior_key not in self._missing_prior_logged
                    and prior_key not in self._invalid_prior_logged
                    and prior_key not in self._zero_prior_logged
                ):
                    logger.warning(
                        "Using zero prostate prior for %s because no valid prior tensor was returned.",
                        prior_key,
                    )
                    self._zero_prior_logged.add(prior_key)
                prior = np.zeros(
                    (self.prostate_prior_channels, *data.shape[1:]), dtype=np.float32
                )
            data = np.concatenate([data, prior], axis=0)

        if self.transform:
            data = self.transform(data)

        sample = {
            "data": torch.from_numpy(data).float(),
            "isup_grade": torch.tensor(isup_grade, dtype=torch.long),
            "patient_id": patient_id,
        }

        if self.use_clinical and self.clinical_features is not None:
            sample["clinical_features"] = torch.from_numpy(
                self.clinical_features[idx]
            ).float()

        return sample

    def load_mri_sequences(self, patient_id, study_id):
        patient_id = normalize_identifier(patient_id)
        study_id = normalize_identifier(study_id)

        scan_paths = resolve_scan_paths(self.dataset_config, patient_id, study_id)
        if scan_paths is None:
            raise FileNotFoundError(
                f"MRI files not found for patient={patient_id}, study={study_id} in {self.dataset_config.data_dir}"
            )

        # Load sequences as SimpleITK images
        scans = [sitk.ReadImage(str(path)) for path in scan_paths]

        for i, scan in enumerate(scans):
            arr = sitk.GetArrayFromImage(scan)
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected 3D MRI data, got {arr.ndim}D for {scan_paths[i]}\n"
                    f"Shape: {arr.shape}"
                )

        # CRITICAL: Use picai_prep registration to align ADC and HBV to T2W space
        preprocessing_settings = PreprocessingSettings(
            matrix_size=[20, 256, 256],
            spacing=None,
            scan_interpolator=sitk.sitkBSpline,
        )

        sample = Sample(
            scans=scans,
            settings=preprocessing_settings,
            name=f"{patient_id}_{study_id}",
        )

        # Registration: resample ADC and HBV to T2W, then crop/pad to 20x256x256
        sample.resample_to_first_scan()
        sample.centre_crop_or_pad()
        sample.align_physical_metadata()

        # Convert registered sequences to numpy and normalize
        data_list = []
        for scan in sample.scans:
            arr = sitk.GetArrayFromImage(scan)
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            data_list.append(arr)

        data = np.stack(data_list, axis=0)
        return data

    def load_preprocessed(self, patient_id, study_id):
        patient_id = normalize_identifier(patient_id)
        study_id = normalize_identifier(study_id)
        preprocessed_path = find_preprocessed_file(self.data_dir, patient_id, study_id)

        if preprocessed_path is None:
            raise FileNotFoundError(
                f"Preprocessed file not found for patient={patient_id}, study={study_id} in {self.data_dir}"
            )

        data = np.load(preprocessed_path).astype(np.float32)
        return data

    def _expected_prostate_prior_channels(self) -> int:
        if self.prostate_prior_type == "whole_gland":
            return 1
        if self.prostate_prior_type == "zonal":
            return 2
        return 3  # both

    def _cache_path_for_prior(self, patient_id: str, study_id: str) -> Optional[Path]:
        if self.prostate_prior_cache_dir is None:
            return None

        prefix = "soft" if self.prostate_prior_soft else "hard"
        prior_subdir = (
            self.prostate_prior_cache_dir / f"{prefix}_{self.prostate_prior_type}"
        )
        prior_subdir.mkdir(parents=True, exist_ok=True)
        return prior_subdir / f"{patient_id}_{study_id}.npy"

    def _prior_from_label_map(self, arr: np.ndarray) -> np.ndarray:
        if self.prostate_prior_type == "whole_gland":
            return (arr > 0).astype(np.float32)[None, ...]

        zonal = np.stack(
            [
                (arr == 1).astype(np.float32),
                (arr == 2).astype(np.float32),
            ],
            axis=0,
        )

        if self.prostate_prior_type == "zonal":
            return zonal

        whole = (arr > 0).astype(np.float32)[None, ...]
        return np.concatenate([whole, zonal], axis=0)

    def _format_loaded_prior_array(self, arr: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if self.prostate_prior_soft:
                if self.prostate_prior_type == "whole_gland":
                    return arr.astype(np.float32)[None, ...]
                if self.prostate_prior_type == "zonal":
                    return None
                return np.concatenate(
                    [
                        arr.astype(np.float32)[None, ...],
                        np.zeros((2, *arr.shape), dtype=np.float32),
                    ],
                    axis=0,
                )
            return self._prior_from_label_map(arr)

        if arr.ndim != 4:
            return None

        if arr.shape[0] in {1, 2, 3}:
            channel_first = arr
        elif arr.shape[-1] in {1, 2, 3}:
            channel_first = np.moveaxis(arr, -1, 0)
        else:
            return None

        channel_first = channel_first.astype(np.float32)
        if self.prostate_prior_type == "whole_gland":
            return channel_first[:1]
        if self.prostate_prior_type == "zonal":
            if channel_first.shape[0] >= 2:
                return channel_first[:2]
            return None

        if channel_first.shape[0] >= 3:
            return channel_first[:3]
        if channel_first.shape[0] == 2:
            whole = np.clip(channel_first.sum(axis=0, keepdims=True), 0.0, 1.0)
            return np.concatenate([whole, channel_first], axis=0)
        if channel_first.shape[0] == 1:
            whole = channel_first[:1]
            return np.concatenate(
                [whole, np.zeros((2, *whole.shape[1:]), dtype=np.float32)], axis=0
            )
        return None

    def _preprocess_prior_image(self, mask_path: Path) -> Optional[np.ndarray]:
        try:
            mask_img = sitk.ReadImage(str(mask_path))
        except Exception as exc:
            logger.warning("Failed to read prostate prior mask %s: %s", mask_path, exc)
            return None

        settings = PreprocessingSettings(
            matrix_size=[20, 256, 256],
            spacing=None,
            scan_interpolator=sitk.sitkNearestNeighbor,
        )

        sample = Sample(
            scans=[mask_img],
            settings=settings,
            name=mask_path.stem,
        )
        sample.centre_crop_or_pad()
        sample.align_physical_metadata()

        arr = sitk.GetArrayFromImage(sample.scans[0])
        if self.prostate_prior_soft:
            return self._format_loaded_prior_array(arr)
        return self._prior_from_label_map(arr)

    def load_prostate_prior(self, patient_id, study_id) -> Optional[np.ndarray]:
        patient_id = normalize_identifier(patient_id)
        study_id = normalize_identifier(study_id)

        cache_path = self._cache_path_for_prior(patient_id, study_id)
        key = f"{patient_id}_{study_id}"
        if cache_path is not None and cache_path.exists():
            prior = np.load(cache_path).astype(np.float32)
        else:
            if self.prostate_prior_dir is None:
                return None

            npy_path = self.prostate_prior_dir / f"{key}.npy"
            nii_path = self.prostate_prior_dir / f"{key}.nii.gz"
            mha_path = self.prostate_prior_dir / f"{key}.mha"
            source_path: Optional[Path] = None

            if npy_path.exists():
                source_path = npy_path
                prior = self._format_loaded_prior_array(np.load(npy_path))
            elif nii_path.exists():
                source_path = nii_path
                prior = self._preprocess_prior_image(nii_path)
            elif mha_path.exists():
                source_path = mha_path
                prior = self._preprocess_prior_image(mha_path)
            else:
                if key not in self._missing_prior_logged:
                    logger.warning(
                        "Prostate prior missing for %s in %s. Using zero prior.",
                        key,
                        self.prostate_prior_dir,
                    )
                    self._missing_prior_logged.add(key)
                return None

            if prior is None:
                if key not in self._invalid_prior_logged:
                    logger.warning(
                        "Prostate prior for %s at %s could not be interpreted for type '%s'. Using zero prior.",
                        key,
                        source_path,
                        self.prostate_prior_type,
                    )
                    self._invalid_prior_logged.add(key)
                return None

            if cache_path is not None:
                np.save(cache_path, prior.astype(np.float32))

        if prior.shape[0] != self.prostate_prior_channels:
            if key not in self._invalid_prior_logged:
                logger.warning(
                    "Prostate prior channel mismatch for %s: got %d channels, expected %d. Using zero prior.",
                    key,
                    int(prior.shape[0]),
                    int(self.prostate_prior_channels),
                )
                self._invalid_prior_logged.add(key)
            return None

        if self.prostate_prior_soft and self.prostate_prior_conf_thresh > 0:
            prior = np.where(prior >= self.prostate_prior_conf_thresh, prior, 0.0)

        return prior.astype(np.float32) * self.prostate_prior_strength

    def _resolve_clinical_column(self, feature_name: str) -> str:
        if feature_name in self.clinical_column_map:
            return self.clinical_column_map[feature_name]
        return self.DEFAULT_CLINICAL_COLUMNS.get(feature_name, feature_name)

    def _prepare_clinical_features(self) -> np.ndarray:
        if self.clinical_impute not in {"median", "constant"}:
            raise ValueError(f"Unsupported clinical_impute '{self.clinical_impute}'.")

        matrices = []
        missing_indicators = []

        for feature_name in self.clinical_feature_names:
            column = self._resolve_clinical_column(feature_name)
            if column in self.df.columns:
                values = pd.to_numeric(self.df[column], errors="coerce").astype(float)
            else:
                values = pd.Series(np.nan, index=self.df.index, dtype=float)

            if feature_name == "psad" and values.isna().all():
                psa_col = self._resolve_clinical_column("psa")
                volume_col = self._resolve_clinical_column("prostate_volume")
                if psa_col in self.df.columns and volume_col in self.df.columns:
                    psa_values = pd.to_numeric(
                        self.df[psa_col], errors="coerce"
                    ).astype(float)
                    volume_values = pd.to_numeric(
                        self.df[volume_col], errors="coerce"
                    ).astype(float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        values = psa_values / volume_values.replace(0.0, np.nan)

            if feature_name in {"psa", "psad", "prostate_volume"}:
                values = np.log1p(values.clip(lower=0.0))

            missing = values.isna()
            if self.clinical_impute == "median":
                fill_value = values.median(skipna=True)
                if pd.isna(fill_value):
                    fill_value = 0.0
            else:
                fill_value = 0.0

            values = values.fillna(fill_value)
            mean = values.mean()
            std = values.std()
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0

            standardized = ((values - mean) / std).to_numpy(dtype=np.float32)
            matrices.append(standardized)
            missing_indicators.append(missing.to_numpy(dtype=np.float32))

        if not matrices:
            return np.zeros((len(self.df), 0), dtype=np.float32)

        clinical_matrix = np.stack(matrices, axis=1)
        if self.clinical_missing_indicators:
            missing_matrix = np.stack(missing_indicators, axis=1)
            clinical_matrix = np.concatenate([clinical_matrix, missing_matrix], axis=1)

        return clinical_matrix.astype(np.float32)
