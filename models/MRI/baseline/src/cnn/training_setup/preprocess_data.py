import numpy as np
from pathlib import Path

import SimpleITK as sitk
from picai_prep.preprocessing import Sample, PreprocessingSettings
from tqdm import tqdm

from training_setup.dataset_config import (
    DatasetConfig,
    find_preprocessed_file,
    load_dataset_metadata,
    normalize_identifier,
    resolve_scan_paths,
)


def preprocess_and_save_dataset_from_config(config: DatasetConfig):
    """Preprocess one dataset according to dataset configuration."""
    output_dir = Path(config.preprocessed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset_metadata(config)

    preprocessing_settings = PreprocessingSettings(
        matrix_size=[20, 256, 256],
        spacing=None,
        scan_interpolator=sitk.sitkBSpline,
    )

    print(f"Preprocessing {len(df)} samples for {config.center_alias} to {output_dir}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {config.center_alias}"):
        patient_id = normalize_identifier(row["patient_id"])
        study_id = normalize_identifier(row["study_id"])

        output_patient_dir = output_dir / patient_id
        output_patient_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_patient_dir / f"{patient_id}_{study_id}_registered.npy"
        if output_path.exists():
            continue

        scan_paths = resolve_scan_paths(config, patient_id, study_id)
        if scan_paths is None:
            print(f"\nWarning: Missing MRI sequences for patient={patient_id}, study={study_id}")
            continue

        scans = []
        skip = False
        for path in scan_paths:
            img = sitk.ReadImage(str(path))
            n_comp = img.GetNumberOfComponentsPerPixel()
            if n_comp > 1:
                print(f"\nWarning: Skipping patient={patient_id}, study={study_id} — "
                      f"vector pixel type ({n_comp} components) in {path.name}")
                skip = True
                break
            if img.GetPixelID() != sitk.sitkFloat32:
                img = sitk.Cast(img, sitk.sitkFloat32)
            scans.append(img)
        if skip:
            continue

        sample = Sample(
            scans=scans,
            settings=preprocessing_settings,
            name=f"{patient_id}_{study_id}",
        )

        sample.resample_to_first_scan()
        sample.centre_crop_or_pad()
        sample.align_physical_metadata()

        data_list = []
        for scan in sample.scans:
            arr = sitk.GetArrayFromImage(scan)
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            data_list.append(arr)

        data = np.stack(data_list, axis=0)
        np.save(output_path, data)

    print(f"\nPreprocessing complete. Data saved to {output_dir}")


def check_preprocessed_data_exists_from_config(config: DatasetConfig) -> bool:
    """Check whether all samples in metadata already have registered tensors."""
    preprocessed_dir = Path(config.preprocessed_dir)
    if not preprocessed_dir.exists():
        return False

    df = load_dataset_metadata(config)
    for _, row in df.iterrows():
        patient_id = normalize_identifier(row["patient_id"])
        study_id = normalize_identifier(row["study_id"])
        if find_preprocessed_file(preprocessed_dir, patient_id, study_id) is None:
            return False
    return True


def preprocess_and_save_dataset(marksheet_path, raw_data_dir, output_dir, centers=None):
    """Backward-compatible wrapper for PI-CAI public datasets."""
    config = DatasetConfig(
        center_alias="public",
        marksheet_path=marksheet_path,
        data_dir=raw_data_dir,
        preprocessed_dir=output_dir,
        table_format="csv",
        table_skiprows=0,
        center_column="center",
        center_filter=list(centers) if centers else None,
        patient_id_column="patient_id",
        study_id_column="study_id",
        label_column="case_ISUP",
        label_mode="isup",
        sequence_strategy="public",
    )
    preprocess_and_save_dataset_from_config(config)


def check_preprocessed_data_exists(marksheet_path, preprocessed_dir, centers=None):
    """Backward-compatible wrapper for PI-CAI public datasets."""
    config = DatasetConfig(
        center_alias="public",
        marksheet_path=marksheet_path,
        data_dir="input/images",
        preprocessed_dir=preprocessed_dir,
        table_format="csv",
        table_skiprows=0,
        center_column="center",
        center_filter=list(centers) if centers else None,
        patient_id_column="patient_id",
        study_id_column="study_id",
        label_column="case_ISUP",
        label_mode="isup",
        sequence_strategy="public",
    )
    return check_preprocessed_data_exists_from_config(config)
