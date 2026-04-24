from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Optional, Any
from collections.abc import Mapping
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class MetadataSourceConfig:
    """Configuration for one metadata source contributing to a dataset.

    Used for composite datasets such as the in-house UULM cohort where MRI-only
    and PET/MRI cases live in separate spreadsheets but should be exposed as one
    canonical metadata table to the training pipeline.
    """

    source_name: str
    marksheet_path: str | Path
    table_format: str = "excel"  # csv|excel
    table_skiprows: int = 0
    sheet_name: Optional[str] = None
    patient_id_column: str = "patient_id"
    study_id_column: str = "study_id"
    label_column: str = "case_ISUP"
    label_mode: str = "isup"  # isup|binary_negative|binary_positive
    patient_id_pad_width: Optional[int] = None
    study_id_pad_width: Optional[int] = None
    extra_column_map: Optional[dict[str, str | list[str]]] = None
    static_columns: Optional[dict[str, Any]] = None
    required: bool = True

    def __post_init__(self):
        self.marksheet_path = Path(self.marksheet_path)


@dataclass
class DatasetConfig:
    """Configuration for dataset metadata and MRI file layout."""

    center_alias: str
    marksheet_path: str | Path
    data_dir: str | Path
    preprocessed_dir: str | Path
    table_format: str = "csv"  # csv|excel
    table_skiprows: int = 0
    center_column: Optional[str] = "center"
    center_filter: Optional[list[str]] = None
    patient_id_column: str = "patient_id"
    study_id_column: str = "study_id"
    label_column: str = "case_ISUP"
    label_mode: str = "isup"  # isup|binary_negative
    sequence_strategy: str = "public"  # public|uulm
    temporary_skip_missing_scans: bool = False
    extra_columns: Optional[list[str]] = None
    metadata_sources: Optional[list[MetadataSourceConfig]] = None

    def __post_init__(self):
        self.marksheet_path = Path(self.marksheet_path)
        self.data_dir = Path(self.data_dir)
        self.preprocessed_dir = Path(self.preprocessed_dir)
        if self.metadata_sources:
            normalized_sources: list[MetadataSourceConfig] = []
            for source in self.metadata_sources:
                if isinstance(source, MetadataSourceConfig):
                    normalized_sources.append(source)
                elif isinstance(source, Mapping):
                    normalized_sources.append(MetadataSourceConfig(**dict(source)))
                elif is_dataclass(source):
                    normalized_sources.append(MetadataSourceConfig(**asdict(source)))
                elif hasattr(source, "__dict__"):
                    normalized_sources.append(
                        MetadataSourceConfig(**dict(source.__dict__))
                    )
                else:
                    raise TypeError(
                        "metadata_sources entries must be MetadataSourceConfig instances, mappings, "
                        f"or dataclass-like objects. Got: {type(source)!r}"
                    )
            self.metadata_sources = normalized_sources


def split_center_expression(center_expression: str) -> list[str]:
    """Support merged center expressions like 'RUMC+PCNN+ZGT'."""
    return [
        center.strip() for center in str(center_expression).split("+") if center.strip()
    ]


def normalize_identifier(value) -> str:
    """Normalize patient/study IDs while preserving useful strings."""
    if pd.isna(value):
        raise ValueError("Identifier value is NaN")

    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _to_bool(value) -> bool:
    """Parse mixed-type boolean columns from clinical sheets."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(int(value))

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "ja", "negativ", "negative"}:
        return True
    if text in {"0", "false", "f", "no", "n", "nein", "positiv", "positive", ""}:
        return False

    raise ValueError(f"Could not parse boolean value '{value}'")


def _read_metadata_table(
    path: Path,
    table_format: str,
    table_skiprows: int = 0,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Read metadata table from CSV/Excel with optional sheet selection."""
    table_format = table_format.lower()
    if table_format == "csv":
        return pd.read_csv(path, skiprows=table_skiprows)
    if table_format in {"xlsx", "xls", "excel"}:
        try:
            read_sheet_name = 0 if sheet_name is None else sheet_name
            return pd.read_excel(
                path, skiprows=table_skiprows, sheet_name=read_sheet_name
            )
        except ImportError as exc:
            raise ImportError(
                "Reading Excel metadata requires 'openpyxl'. Install it in .venv-cnn, "
                "for example: .venv-cnn\\Scripts\\python.exe -m pip install openpyxl"
            ) from exc
    raise ValueError(f"Unsupported table_format '{table_format}'")


def _normalize_series_identifier(
    series: pd.Series, pad_width: Optional[int] = None
) -> pd.Series:
    """Normalize identifier column and optionally zero-pad numeric IDs."""
    normalized = series.apply(normalize_identifier)
    if pad_width is not None:
        normalized = normalized.apply(
            lambda value: (
                value.zfill(pad_width)
                if isinstance(value, str) and value.isdigit()
                else value
            )
        )
    return normalized


# Gleason score -> ISUP Grade Group.  csPCa = ISUP >= 2  <=>  Gleason >= 3+4=7.
_GLEASON_TO_GG: dict[str, int] = {
    "6": 1,
    "3+3=6": 1,
    "7a": 2,
    "3+4=7": 2,
    "7b": 3,
    "4+3=7": 3,
    "8": 4,
    "8a": 4,
    "3+5=8": 4,
    "4+4=8": 4,
    "5+3=8": 4,
    "9": 5,
    "9a": 5,
    "4+5=9": 5,
    "9b": 5,
    "5+4=9": 5,
    "10": 5,
    "5+5=10": 5,
}


def _gleason_to_isup(value) -> int:
    """Map a raw Gleason score to a canonical case_ISUP value (0 or 2).

    Returns 0 (not csPCa) for NaN, Gleason 6 / GG1, or non-cancer findings
    (e.g. "high grade PIN").  Returns 2 (csPCa) for Gleason >= 7a / GG >= 2.
    The value 2 is chosen so that the existing ``(isup_labels >= 2)`` binary
    threshold in ``data_generator.py`` works unchanged.
    """
    if pd.isna(value) or str(value).strip() == "":
        return 0

    raw = str(value).strip().lower()

    if "pin" in raw or "asap" in raw:
        return 0

    gg = _GLEASON_TO_GG.get(raw)
    if gg is not None:
        return 2 if gg >= 2 else 0

    # Fallback: try numeric parse
    try:
        num = int(float(raw))
        return 2 if num >= 7 else 0
    except (ValueError, TypeError):
        logger.warning("Unrecognized Gleason value %r – treating as non-csPCa", value)
        return 0


def _convert_labels(raw_labels: pd.Series, label_mode: str) -> np.ndarray:
    """Convert raw source-specific labels to canonical case_ISUP values."""
    if label_mode == "isup":
        labels = pd.to_numeric(raw_labels, errors="coerce")
        if labels.isna().any():
            bad = int(labels.isna().sum())
            raise ValueError(f"Found {bad} non-numeric ISUP labels in metadata source")
        return labels.astype(int).to_numpy()

    if label_mode == "binary_negative":
        is_negative = raw_labels.apply(_to_bool)
        binary_label = (~is_negative).astype(int)
        return np.where(binary_label == 1, 2, 0).astype(int)

    if label_mode == "binary_positive":
        is_positive = raw_labels.apply(_to_bool)
        binary_label = is_positive.astype(int)
        return np.where(binary_label == 1, 2, 0).astype(int)

    if label_mode == "gleason":
        return raw_labels.apply(_gleason_to_isup).to_numpy().astype(int)

    raise ValueError(f"Unsupported label_mode '{label_mode}'")


def _resolve_extra_column(df: pd.DataFrame, spec: str | list[str]) -> pd.Series:
    """Resolve canonical column from one or more source columns with fallbacks."""
    candidates = [spec] if isinstance(spec, str) else list(spec)
    values: Optional[pd.Series] = None
    for candidate in candidates:
        if candidate in df.columns:
            candidate_values = df[candidate].copy()
            if values is None:
                values = candidate_values
            else:
                values = values.where(values.notna(), candidate_values)
    if values is None:
        return pd.Series(np.nan, index=df.index, dtype=object)
    return values


def _load_single_metadata_source(
    source: MetadataSourceConfig,
    center_alias: str,
    requested_extra_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load one metadata source and convert it to canonical output columns."""
    if source.required and not source.marksheet_path.exists():
        raise FileNotFoundError(
            f"Required metadata source not found: {source.marksheet_path}"
        )
    if not source.marksheet_path.exists():
        logger.warning("Optional metadata source missing: %s", source.marksheet_path)
        return pd.DataFrame(columns=["patient_id", "study_id", "center", "case_ISUP"])

    df = _read_metadata_table(
        source.marksheet_path,
        source.table_format,
        table_skiprows=source.table_skiprows,
        sheet_name=source.sheet_name,
    )

    required = [source.patient_id_column, source.study_id_column, source.label_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in metadata source {source.source_name} ({source.marksheet_path}). "
            f"Available columns: {list(df.columns)}"
        )

    for column in required:
        if df[column].dtype == object:
            df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)

    # For label_mode="gleason", _gleason_to_isup() handles NaN labels
    # correctly (NaN / missing Gleason -> 0 = non-csPCa, i.e. biopsy-
    # negative).  Only require non-NaN patient and study identifiers so
    # that biopsy-negative cases are kept in the dataset.
    nan_required = [source.patient_id_column, source.study_id_column]
    if source.label_mode != "gleason":
        nan_required.append(source.label_column)

    drop_mask = df[nan_required].isna().any(axis=1)
    if drop_mask.any():
        logger.warning(
            "Dropped %d rows with missing %s in metadata source %s",
            int(drop_mask.sum()),
            "/".join(nan_required),
            source.source_name,
        )
        df = df.loc[~drop_mask].reset_index(drop=True)

    patient_ids = _normalize_series_identifier(
        df[source.patient_id_column], source.patient_id_pad_width
    )
    study_ids = _normalize_series_identifier(
        df[source.study_id_column], source.study_id_pad_width
    )
    case_isup = _convert_labels(df[source.label_column], source.label_mode)

    out = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "study_id": study_ids,
            "center": [center_alias] * len(df),
            "case_ISUP": case_isup,
        }
    )

    static_columns = dict(source.static_columns or {})
    static_columns.setdefault("metadata_source", source.source_name)
    for column, value in static_columns.items():
        out[column] = value

    extra_columns = list(dict.fromkeys(requested_extra_columns or []))
    source_extra_map = source.extra_column_map or {}
    for column in extra_columns:
        if column in out.columns:
            continue
        if column in source_extra_map:
            out[column] = _resolve_extra_column(
                df, source_extra_map[column]
            ).reset_index(drop=True)
        else:
            out[column] = np.nan

    duplicated = out.duplicated(subset=["patient_id", "study_id"], keep=False)
    if duplicated.any():
        logger.warning(
            "Deduplicating repeated patient/study rows within metadata source '%s'.",
            source.source_name,
        )
        out = _merge_duplicate_metadata_rows(out)

    return out.reset_index(drop=True)


def _merge_duplicate_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge cross-source duplicates while checking label consistency."""
    if (
        df.empty
        or not df.duplicated(subset=["patient_id", "study_id"], keep=False).any()
    ):
        return df.reset_index(drop=True)

    merged_rows = []
    for _, group in df.groupby(["patient_id", "study_id"], sort=False, dropna=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue

        labels = group["case_ISUP"].dropna().unique().tolist()
        if len(labels) > 1:
            sources = group.get(
                "metadata_source", pd.Series(index=group.index, dtype=object)
            ).tolist()
            raise ValueError(
                "Conflicting labels for duplicated patient/study pair "
                f"{group.iloc[0]['patient_id']}_{group.iloc[0]['study_id']} across sources {sources}: {labels}"
            )

        merged = group.iloc[0].copy()
        if "metadata_source" in group.columns:
            merged["metadata_source"] = "+".join(
                sorted(set(group["metadata_source"].astype(str).tolist()))
            )
        if "pet_available" in group.columns:
            merged["pet_available"] = (
                pd.to_numeric(group["pet_available"], errors="coerce").fillna(0).max()
            )

        for column in group.columns:
            if column in {
                "patient_id",
                "study_id",
                "center",
                "case_ISUP",
                "metadata_source",
                "pet_available",
            }:
                continue
            non_null = group[column].dropna()
            if not non_null.empty:
                merged[column] = non_null.iloc[0]

        merged_rows.append(merged.to_dict())

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def _load_single_table_dataset_metadata(config: DatasetConfig) -> pd.DataFrame:
    """Existing single-table metadata loader path."""
    df = _read_metadata_table(
        config.marksheet_path, config.table_format, table_skiprows=config.table_skiprows
    )

    required = [config.patient_id_column, config.study_id_column, config.label_column]
    if config.center_column:
        required.append(config.center_column)

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {config.marksheet_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if config.center_filter and config.center_column:
        df = df[df[config.center_column].isin(config.center_filter)]

    before_len = len(df)
    for column in [
        config.patient_id_column,
        config.study_id_column,
        config.label_column,
    ]:
        if df[column].dtype == object:
            df[column] = df[column].replace(r"^\s*$", np.nan, regex=True)
    drop_mask = (
        df[[config.patient_id_column, config.study_id_column, config.label_column]]
        .isna()
        .any(axis=1)
    )
    if drop_mask.any():
        dropped_df = df.loc[drop_mask]
        for idx, row in dropped_df.iterrows():
            logger.warning(
                "  Dropped row %s: %s=%s, %s=%s, %s=%s",
                idx,
                config.patient_id_column,
                row.get(config.patient_id_column, "<NA>"),
                config.study_id_column,
                row.get(config.study_id_column, "<NA>"),
                config.label_column,
                row.get(config.label_column, "<NA>"),
            )
        logger.warning(
            "Dropped %d rows with missing patient/study/label in %s",
            int(drop_mask.sum()),
            config.marksheet_path,
        )
    df = df.loc[~drop_mask].reset_index(drop=True)

    patient_ids = df[config.patient_id_column].apply(normalize_identifier)
    study_ids = df[config.study_id_column].apply(normalize_identifier)

    if config.label_mode == "isup":
        labels = pd.to_numeric(df[config.label_column], errors="coerce")
        if labels.isna().any():
            bad = int(labels.isna().sum())
            raise ValueError(
                f"Found {bad} non-numeric ISUP labels in column '{config.label_column}'"
            )
        case_isup = labels.astype(int)
    elif config.label_mode == "binary_negative":
        is_negative = df[config.label_column].apply(_to_bool)
        binary_label = (~is_negative).astype(int)
        case_isup = np.where(binary_label == 1, 2, 0).astype(int)
    else:
        raise ValueError(f"Unsupported label_mode '{config.label_mode}'")

    if config.center_column:
        centers = df[config.center_column].astype(str)
    else:
        centers = pd.Series([config.center_alias] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "study_id": study_ids,
            "center": centers,
            "case_ISUP": case_isup,
        }
    )

    extra_columns = []
    if config.extra_columns:
        extra_columns = list(dict.fromkeys(config.extra_columns))

    for column in extra_columns:
        if column in out.columns:
            continue
        if column in df.columns:
            out[column] = df[column].reset_index(drop=True)
        else:
            logger.warning(
                "Extra column '%s' not found in %s. Filling with NaN.",
                column,
                config.marksheet_path,
            )
            out[column] = np.nan

    return out.reset_index(drop=True)


def _load_multi_source_dataset_metadata(config: DatasetConfig) -> pd.DataFrame:
    """Load a dataset composed of multiple metadata files into one canonical table."""
    if not config.metadata_sources:
        raise ValueError(
            "metadata_sources must be provided for multi-source metadata loading"
        )

    requested_extra_columns = list(dict.fromkeys(config.extra_columns or []))
    for source in config.metadata_sources:
        for column in (source.extra_column_map or {}).keys():
            if column not in requested_extra_columns:
                requested_extra_columns.append(column)

    frames = [
        _load_single_metadata_source(
            source,
            center_alias=config.center_alias,
            requested_extra_columns=requested_extra_columns,
        )
        for source in config.metadata_sources
    ]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _merge_duplicate_metadata_rows(out)

    if out.empty:
        raise ValueError(
            f"No rows available for dataset '{config.center_alias}' from metadata sources "
            f"{[str(source.marksheet_path) for source in config.metadata_sources]}"
        )

    return out.reset_index(drop=True)


def load_dataset_metadata(config: DatasetConfig) -> pd.DataFrame:
    """Load metadata and normalize to expected columns.

    Output columns: patient_id, study_id, center, case_ISUP (+ optional extra columns)
    """
    if config.metadata_sources:
        out = _load_multi_source_dataset_metadata(config)
    else:
        out = _load_single_table_dataset_metadata(config)

    if config.temporary_skip_missing_scans:
        # TEMPORARY: skip metadata rows without available MRI triplets until
        # the ground-truth spreadsheet is fully aligned with on-disk scans.
        valid_mask = []
        use_preprocessed_presence = Path(config.preprocessed_dir).exists()
        for patient_id, study_id in zip(out["patient_id"], out["study_id"]):
            has_sample = False
            if use_preprocessed_presence:
                # Fast path for training with preprocessed tensors: avoid expensive
                # recursive glob over raw MRI tree for every metadata row.
                has_sample = (
                    find_preprocessed_file(
                        config.preprocessed_dir, patient_id, study_id
                    )
                    is not None
                )
            if not has_sample:
                # Fallback to raw scan lookup so newly added metadata rows are not
                # dropped just because the preprocessed root folder already exists.
                has_sample = (
                    resolve_scan_paths(config, patient_id, study_id) is not None
                )
            valid_mask.append(has_sample)

        valid_mask = np.array(valid_mask, dtype=bool)
        missing_scans = int((~valid_mask).sum())
        if missing_scans > 0:
            skipped_patients = out.loc[~valid_mask, "patient_id"].tolist()
            skipped_studies = out.loc[~valid_mask, "study_id"].tolist()
            for pid, sid in zip(skipped_patients, skipped_studies):
                logger.warning("  No MRI found for patient=%s, study=%s", pid, sid)
            logger.warning(
                "TEMPORARY FILTER: skipped %d metadata rows without matching MRI sequences in %s",
                missing_scans,
                config.data_dir,
            )

        out = out.loc[valid_mask].reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"No rows available for dataset '{config.center_alias}' from {config.marksheet_path}"
        )

    return out.reset_index(drop=True)


def identifier_candidates(identifier: str) -> list[str]:
    """Generate common ID variants used in file naming."""
    identifier = normalize_identifier(identifier)
    candidates = [identifier]

    if identifier.isdigit():
        as_int = int(identifier)
        candidates.extend(
            [
                str(as_int),
                f"{as_int:05d}",
                f"{as_int:08d}",
                f"{as_int:010d}",
            ]
        )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def find_preprocessed_file(
    preprocessed_dir: Path, patient_id: str, study_id: str
) -> Optional[Path]:
    """Locate a preprocessed registered tensor file with robust ID matching."""
    preprocessed_dir = Path(preprocessed_dir)
    patient_variants = identifier_candidates(patient_id)
    study_variants = identifier_candidates(study_id)

    for patient_variant in patient_variants:
        patient_dir = preprocessed_dir / patient_variant
        for study_variant in study_variants:
            candidate = (
                patient_dir / f"{patient_variant}_{study_variant}_registered.npy"
            )
            if candidate.exists():
                return candidate
    return None


def resolve_scan_paths(
    config: DatasetConfig, patient_id: str, study_id: str
) -> Optional[list[Path]]:
    """Resolve T2W/ADC/HBV scan paths for one patient/study pair."""
    sequences = ["t2w", "adc", "hbv"]
    patient_variants = identifier_candidates(patient_id)
    study_variants = identifier_candidates(study_id)

    def _exact_public_layout() -> Optional[list[Path]]:
        for patient_variant in patient_variants:
            for study_variant in study_variants:
                patient_dir = config.data_dir / patient_variant
                candidate_paths = [
                    patient_dir / f"{patient_variant}_{study_variant}_{sequence}.mha"
                    for sequence in sequences
                ]
                if all(path.exists() for path in candidate_paths):
                    return candidate_paths
        return None

    exact_paths = _exact_public_layout()
    if exact_paths is not None:
        return exact_paths

    if config.sequence_strategy != "uulm":
        return None

    # Fallback for in-house data layout variations: recursive search by case/study prefix.
    resolved_paths = []
    for sequence in sequences:
        found = None
        for patient_variant in patient_variants:
            for study_variant in study_variants:
                pattern = f"**/{patient_variant}_{study_variant}_*{sequence}*.mha"
                matches = sorted(config.data_dir.glob(pattern))
                if matches:
                    found = matches[0]
                    break
            if found is not None:
                break

        if found is None:
            return None
        resolved_paths.append(found)

    return resolved_paths
