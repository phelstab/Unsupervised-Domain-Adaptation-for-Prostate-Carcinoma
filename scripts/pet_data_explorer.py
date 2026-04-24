"""PET DICOM Explorer — Identify and catalog PET AC series for each patient.

Scans ``0ii/files/pet/`` and produces a JSON catalog mapping each patient to
its Pelvis PET AC series location, image geometry, and SUV conversion
parameters.  This catalog is consumed by ``pet_dicom_to_suv.py`` and
``c1_pet_radiomics_fusion.py``.

Usage::

    .venv-cnn\\Scripts\\python.exe scripts/pet_data_explorer.py

Output::

    workdir/pet/pet_series_catalog.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pydicom

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PET_DATA_DIR = PROJECT_ROOT / "0ii" / "files" / "pet"
OUTPUT_DIR = PROJECT_ROOT / "workdir" / "pet"
CATALOG_PATH = OUTPUT_DIR / "pet_series_catalog.json"

TARGET_SERIES_DESC = "Pelvis_PetAcquisition_AC Images"
TARGET_MODALITY = "PT"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dicom_datetime(date_str: str, time_str: str) -> str:
    """Return ISO-8601 string from DICOM DA + TM tags."""
    date_str = str(date_str).strip()
    time_str = str(time_str).strip().split(".")[0]  # drop fractional seconds
    try:
        dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        return dt.isoformat()
    except ValueError:
        return f"{date_str}T{time_str}"


def _extract_series_info(series_path: Path) -> dict | None:
    """Read one DICOM from *series_path* and return metadata dict, or None."""
    dcm_files = sorted(f for f in os.listdir(series_path) if f.endswith(".dcm"))
    if not dcm_files:
        return None

    try:
        dcm = pydicom.dcmread(str(series_path / dcm_files[0]), stop_before_pixels=True)
    except Exception as exc:
        log.warning("Failed to read %s: %s", series_path / dcm_files[0], exc)
        return None

    modality = getattr(dcm, "Modality", "")
    description = getattr(dcm, "SeriesDescription", "")

    if modality != TARGET_MODALITY:
        return None
    if TARGET_SERIES_DESC not in description:
        return None

    # --- SUV conversion parameters ---
    rp_seq = getattr(dcm, "RadiopharmaceuticalInformationSequence", None)
    suv_params: dict = {}
    if rp_seq:
        rp = rp_seq[0]
        suv_params["radionuclide_total_dose_bq"] = float(
            getattr(rp, "RadionuclideTotalDose", 0)
        )
        suv_params["radionuclide_half_life_s"] = float(
            getattr(rp, "RadionuclideHalfLife", 4062.6)
        )
        suv_params["radiopharmaceutical"] = str(getattr(rp, "Radiopharmaceutical", ""))
        suv_params["injection_time"] = str(
            getattr(rp, "RadiopharmaceuticalStartTime", "")
        )
        suv_params["injection_datetime"] = str(
            getattr(rp, "RadiopharmaceuticalStartDateTime", "")
        )

    patient_weight_kg = float(getattr(dcm, "PatientWeight", 0))

    return {
        "series_path": str(series_path),
        "series_description": description,
        "series_number": str(series_path.name),
        "num_slices": len(dcm_files),
        "rows": int(dcm.Rows),
        "columns": int(dcm.Columns),
        "pixel_spacing_mm": [float(x) for x in dcm.PixelSpacing],
        "slice_thickness_mm": float(getattr(dcm, "SliceThickness", 0)),
        "rescale_slope": float(getattr(dcm, "RescaleSlope", 1)),
        "rescale_intercept": float(getattr(dcm, "RescaleIntercept", 0)),
        "units": str(getattr(dcm, "Units", "")),
        "patient_weight_kg": patient_weight_kg,
        "acquisition_datetime": _parse_dicom_datetime(
            getattr(dcm, "AcquisitionDate", ""),
            getattr(dcm, "AcquisitionTime", ""),
        ),
        "decay_correction": str(getattr(dcm, "DecayCorrection", "")),
        "suv_params": suv_params,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_catalog() -> dict:
    """Scan all patients and return a catalog dict."""
    if not PET_DATA_DIR.is_dir():
        log.error("PET data directory not found: %s", PET_DATA_DIR)
        sys.exit(1)

    catalog: dict = {}
    patients = sorted(
        d for d in os.listdir(PET_DATA_DIR) if (PET_DATA_DIR / d).is_dir()
    )
    log.info("Scanning %d patient directories in %s", len(patients), PET_DATA_DIR)

    for pid in patients:
        pid_path = PET_DATA_DIR / pid
        for study in sorted(os.listdir(pid_path)):
            study_path = pid_path / study
            if not study_path.is_dir():
                continue
            for series in sorted(os.listdir(study_path)):
                series_path = study_path / series
                if not series_path.is_dir():
                    continue
                info = _extract_series_info(series_path)
                if info is not None:
                    info["patient_id"] = pid
                    info["study_id"] = study
                    catalog[pid] = info
                    log.info(
                        "  %s / %s / %s -> %s (%d slices)",
                        pid,
                        study,
                        series,
                        info["series_description"],
                        info["num_slices"],
                    )
                    break  # one match per patient is enough
            if pid in catalog:
                break

    log.info(
        "Found Pelvis PET AC series for %d / %d patients", len(catalog), len(patients)
    )

    missing = [p for p in patients if p not in catalog]
    if missing:
        log.warning("Missing Pelvis PET AC for: %s", missing)

    return catalog


def main() -> None:
    catalog = build_catalog()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
    log.info("Catalog written to %s", CATALOG_PATH)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PET DICOM Explorer — Summary")
    print("=" * 60)
    print(
        f"Patients scanned:  {len([d for d in os.listdir(PET_DATA_DIR) if (PET_DATA_DIR / d).is_dir()])}"
    )
    print(f"Pelvis PET AC found: {len(catalog)}")
    print(
        f"Consistent slices:   {len(set(v['num_slices'] for v in catalog.values())) == 1}"
    )
    if catalog:
        first = next(iter(catalog.values()))
        print(
            f"Image size:          {first['rows']} x {first['columns']} x {first['num_slices']}"
        )
        print(f"Pixel spacing:       {first['pixel_spacing_mm']} mm")
        print(f"Slice thickness:     {first['slice_thickness_mm']} mm")
        print(f"Units:               {first['units']}")
    print(f"Catalog path:        {CATALOG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
