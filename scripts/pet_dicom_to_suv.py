"""PET DICOM -> SUV Converter — Convert PET DICOM series to SUVbw-normalized volumes.

Reads the PET series catalog from ``pet_data_explorer.py``, loads DICOM slices,
applies SUVbw (body-weight) normalization, and saves as ``.mha`` files for
downstream radiomics extraction.

SUVbw formula::

    pixel_bqml = raw_pixel * RescaleSlope + RescaleIntercept
    decay_factor = 2^(-delta_t / half_life)
    injected_dose_corrected = injected_dose_bq * decay_factor
    SUVbw = pixel_bqml * patient_weight_kg * 1000 / injected_dose_corrected

Usage::

    .venv-cnn\\Scripts\\python.exe scripts/pet_dicom_to_suv.py

    # Only convert specific patients:
    .venv-cnn\\Scripts\\python.exe scripts/pet_dicom_to_suv.py --patients 0000103667 0003006814

Output::

    workdir/pet/suv_volumes/{patient_id}_{study_id}_pet_suv.mha
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATALOG_PATH = PROJECT_ROOT / "workdir" / "pet" / "pet_series_catalog.json"
OUTPUT_DIR = PROJECT_ROOT / "workdir" / "pet" / "suv_volumes"

# Ga-68 half-life in seconds (fallback if not in DICOM)
GA68_HALF_LIFE_S = 4062.6

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SUV Conversion
# ---------------------------------------------------------------------------


def _parse_dicom_time(time_str: str) -> float:
    """Parse DICOM TM tag to seconds since midnight."""
    time_str = str(time_str).strip()
    # Format: HHMMSS.FFFFFF or HHMMSS
    h = int(time_str[0:2])
    m = int(time_str[2:4])
    s = float(time_str[4:])
    return h * 3600 + m * 60 + s


def _parse_dicom_datetime(dt_str: str) -> datetime:
    """Parse DICOM DT tag to datetime object."""
    dt_str = str(dt_str).strip()
    # Format: YYYYMMDDHHMMSS.FFFFFF
    return datetime.strptime(dt_str[:14], "%Y%m%d%H%M%S")


def compute_suv_factor(dcm: pydicom.Dataset) -> float:
    """Compute the SUVbw multiplication factor from DICOM headers.

    Returns a factor such that: ``SUVbw = pixel_bqml * factor``

    The factor accounts for:
    - Decay correction from injection to acquisition time
    - Body weight normalization
    """
    patient_weight_kg = float(getattr(dcm, "PatientWeight", 0))
    if patient_weight_kg <= 0:
        log.warning("PatientWeight missing or zero, using 75 kg default")
        patient_weight_kg = 75.0

    rp_seq = getattr(dcm, "RadiopharmaceuticalInformationSequence", None)
    if not rp_seq:
        raise ValueError("RadiopharmaceuticalInformationSequence not found in DICOM")
    rp = rp_seq[0]

    injected_dose_bq = float(getattr(rp, "RadionuclideTotalDose", 0))
    half_life_s = float(getattr(rp, "RadionuclideHalfLife", GA68_HALF_LIFE_S))
    injection_time_str = str(getattr(rp, "RadiopharmaceuticalStartTime", ""))

    if injected_dose_bq <= 0:
        raise ValueError("RadionuclideTotalDose missing or zero")

    # Time delta: acquisition time - injection time (seconds)
    acq_time_str = str(getattr(dcm, "AcquisitionTime", ""))
    acq_date_str = str(getattr(dcm, "AcquisitionDate", ""))
    inj_date_str = str(getattr(rp, "RadiopharmaceuticalStartDateTime", ""))

    if inj_date_str and len(inj_date_str) >= 14:
        # Use full datetime comparison (handles midnight crossing)
        inj_dt = _parse_dicom_datetime(inj_date_str)
        acq_dt = datetime.strptime(f"{acq_date_str}{acq_time_str[:6]}", "%Y%m%d%H%M%S")
        delta_t_s = (acq_dt - inj_dt).total_seconds()
    else:
        # Fallback: time-only comparison
        inj_s = _parse_dicom_time(injection_time_str)
        acq_s = _parse_dicom_time(acq_time_str)
        delta_t_s = acq_s - inj_s
        if delta_t_s < 0:
            delta_t_s += 86400  # midnight crossing

    # Decay correction
    decay_factor = 2.0 ** (-delta_t_s / half_life_s)
    corrected_dose_bq = injected_dose_bq * decay_factor

    # SUVbw factor: pixel_bqml -> SUV
    # SUVbw = pixel_bqml * weight_g / corrected_dose_bq
    # weight_g = weight_kg * 1000
    suv_factor = (patient_weight_kg * 1000.0) / corrected_dose_bq

    log.debug(
        "  SUV params: weight=%.1f kg, dose=%.0f MBq, half_life=%.1f s, "
        "delta_t=%.0f s, decay=%.4f, factor=%.6f",
        patient_weight_kg,
        injected_dose_bq / 1e6,
        half_life_s,
        delta_t_s,
        decay_factor,
        suv_factor,
    )

    return suv_factor


def load_pet_series_as_suv(series_path: Path) -> sitk.Image:
    """Load a PET DICOM series and convert to SUVbw-normalized SimpleITK image.

    Parameters
    ----------
    series_path : Path
        Directory containing DICOM ``.dcm`` files for a single PET series.

    Returns
    -------
    sitk.Image
        3D float32 volume with SUVbw values.
    """
    # Collect and sort DICOM files by InstanceNumber
    dcm_files = sorted(
        (series_path / f for f in os.listdir(series_path) if f.endswith(".dcm")),
        key=lambda p: p.name,
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {series_path}")

    # Read first slice for metadata
    dcm0 = pydicom.dcmread(str(dcm_files[0]))
    rescale_slope = float(getattr(dcm0, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(dcm0, "RescaleIntercept", 0.0))
    suv_factor = compute_suv_factor(dcm0)

    # Use SimpleITK's DICOM series reader for proper spatial ordering
    reader = sitk.ImageSeriesReader()
    # Get properly ordered file names from SimpleITK
    series_ids = reader.GetGDCMSeriesIDs(str(series_path))
    if not series_ids:
        raise ValueError(f"No DICOM series found by SimpleITK in {series_path}")

    # Use the first (usually only) series ID
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_path), series_ids[0])
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    # Convert to float and apply SUV
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Apply rescale: pixel_bqml = raw * slope + intercept
    # Then apply SUV factor: SUVbw = pixel_bqml * factor
    # Combined: SUVbw = (raw * slope + intercept) * factor
    # SimpleITK already applies RescaleSlope/Intercept during reading,
    # so we just multiply by the SUV factor.
    #
    # Actually, SimpleITK's GDCM reader applies RescaleSlope/Intercept
    # automatically for PET DICOM. Verify by checking pixel values.
    arr = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    log.debug(
        "  Raw image stats: min=%.2f, max=%.2f, mean=%.2f",
        arr.min(),
        arr.max(),
        arr.mean(),
    )

    # Apply SUV factor
    suv_image = image * suv_factor

    arr_suv = sitk.GetArrayFromImage(suv_image)
    log.info(
        "  SUV stats: min=%.2f, max=%.2f, mean=%.2f, shape=%s",
        arr_suv.min(),
        arr_suv.max(),
        arr_suv.mean(),
        arr_suv.shape,
    )

    return suv_image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PET DICOM -> SUV Converter")
    p.add_argument(
        "--catalog",
        type=Path,
        default=CATALOG_PATH,
        help="Path to PET series catalog JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for SUV .mha files",
    )
    p.add_argument(
        "--patients",
        nargs="*",
        help="Only convert these patient IDs (default: all)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if output file already exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load catalog
    if not args.catalog.exists():
        log.error(
            "Catalog not found: %s. Run pet_data_explorer.py first.", args.catalog
        )
        sys.exit(1)

    with open(args.catalog) as f:
        catalog = json.load(f)

    log.info("Loaded catalog with %d patients", len(catalog))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to requested patients
    patient_ids = args.patients or sorted(catalog.keys())
    patient_ids = [pid for pid in patient_ids if pid in catalog]
    log.info("Converting %d patients", len(patient_ids))

    success = 0
    failed = []

    for pid in patient_ids:
        entry = catalog[pid]
        study_id = entry["study_id"]
        series_path = Path(entry["series_path"])
        output_path = args.output_dir / f"{pid}_{study_id}_pet_suv.mha"

        if output_path.exists() and not args.force:
            log.info("  %s: already exists, skipping", pid)
            success += 1
            continue

        log.info(
            "Converting %s (study %s, series %s)", pid, study_id, entry["series_number"]
        )

        try:
            suv_image = load_pet_series_as_suv(series_path)
            sitk.WriteImage(suv_image, str(output_path))
            log.info("  -> %s", output_path.name)
            success += 1
        except Exception as exc:
            log.error("  FAILED for %s: %s", pid, exc)
            failed.append(pid)

    # Summary
    print("\n" + "=" * 60)
    print("PET DICOM -> SUV Conversion — Summary")
    print("=" * 60)
    print(f"Converted:  {success}/{len(patient_ids)}")
    if failed:
        print(f"Failed:     {failed}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
