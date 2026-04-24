"""Per-validator A4 checkpoint selection for the H4 matrix.

For each of the 6 unsupervised/oracle validators defined in the thesis
(Section 2 "Validators" + Section 3.6), this module picks one A4 checkpoint
per fold from the full pool

    H_A4 = 7 methods x 3 DA weights x 46 epochs x 3 folds

and extracts the per-patient out-of-fold (OOF) P(csPCa) score vector over the
25-patient PET/MRI subset.  The result is one score vector per validator,
used by the B1/C1 PET sidecar classifiers in Experiment III (H4).

Design principles
-----------------
- **Fold-aware OOF**: each UULM target patient is in target_val for exactly
  one fold. For each validator V we pick ``h_V,f = argmax_h V(h)`` within the
  pool restricted to fold ``f`` (7 methods x 3 DA weights x ~46 epochs).
  The patient's score is the target_val softmax probability from that fold.
  This matches the A-series OOF protocol in mri_score_loader.py and is
  theoretically clean: the UDA encoder never saw the patient's MRI as
  unlabeled target-train data in the chosen fold.
- **No cross-fold leakage**: one patient --> one fold --> one score. No
  averaging, no pooling across folds during selection.
- **Cache one file per validator**: ``workdir/pet/mri_scores_per_validator/
  {validator}.json`` with ``{patient_id: score}``, plus a sidecar
  ``selection.json`` recording which (method, da_weight, epoch) was picked
  per fold for auditability.

References
----------
- Thesis Sec. 3.6 (Experiment III: PET-Enriched Extensions)
- validator/validators.py (shared validator scoring logic)
- Musgrave et al. 2022 (unsupervised model-selection reality check)
- Demircioglu 2021 (feature-selection leakage in radiomics CV)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validator.models import AlgorithmId, CheckpointData  # noqa: E402
from validator.validators import (  # noqa: E402
    CorrCValidator,
    EntropyValidator,
    InfoMaxValidator,
    OracleValidator,
    SNDValidator,
    SrcAccValidator,
    Validator,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (thesis-consistent; can be overridden by callers)
# ---------------------------------------------------------------------------
A4_COLLECTION = (
    PROJECT_ROOT / "runs" / "UULM_binary_class_3CV_RESNET10_GLAND_META"
)
TARGET_PAIR = "RUMC+PCNN+ZGT_to_UULM"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "workdir" / "pet" / "mri_scores_per_validator"

# The six validators defined in Section 2 "Validators" and used in Sec. 3.6.
# Key = short name used in file paths and the H4 matrix CSV.
VALIDATOR_REGISTRY: dict[str, Validator] = {
    "oracle": OracleValidator(),
    "src_acc": SrcAccValidator(),
    "entropy": EntropyValidator(),
    "infomax": InfoMaxValidator(),
    "corr_c": CorrCValidator(),
    "snd": SNDValidator(),
}


_ALGO_PATTERN = re.compile(
    r"_(mmd|dann|daarda|mcd|mcc|bnm|hybrid|coral|entropy)_", flags=re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


@dataclass
class CandidateFile:
    """One candidate A4 checkpoint output on disk."""

    output_path: Path
    algorithm: str
    da_weight: float
    fold: int
    epoch: int


def _parse_algorithm(run_name: str) -> str | None:
    match = _ALGO_PATTERN.search(f"_{run_name.lower()}_")
    return match.group(1).lower() if match else None


def _parse_epoch(filename: str) -> int | None:
    match = re.search(r"epoch_(\d+)_outputs\.pt$", filename)
    return int(match.group(1)) if match else None


def enumerate_a4_candidates(collection: Path = A4_COLLECTION) -> list[CandidateFile]:
    """List every (method, da_weight, fold, epoch) output file under the A4 collection.

    The expected directory layout is::

        {collection}/{run_name}/RUMC+PCNN+ZGT_to_UULM/da_{w}/fold_{i}/outputs/epoch_*.pt

    ``run_name`` is parsed for the algorithm name; ``da_{w}`` for the DA weight;
    ``fold_{i}`` for the fold index; and ``epoch_{e}_outputs.pt`` for the epoch.
    Any file not matching the pattern is skipped with a debug log.
    """
    candidates: list[CandidateFile] = []
    if not collection.is_dir():
        raise FileNotFoundError(f"A4 collection not found: {collection}")

    for run_dir in sorted(collection.iterdir()):
        if not run_dir.is_dir():
            continue
        algorithm = _parse_algorithm(run_dir.name)
        if algorithm is None:
            log.debug("skip %s: algorithm token not found", run_dir.name)
            continue

        domain_dir = run_dir / TARGET_PAIR
        if not domain_dir.is_dir():
            continue

        for da_dir in sorted(domain_dir.glob("da_*")):
            m = re.match(r"da_([0-9.]+)$", da_dir.name)
            if not m:
                continue
            da_weight = float(m.group(1))

            for fold_dir in sorted(da_dir.glob("fold_*")):
                m = re.match(r"fold_(\d+)$", fold_dir.name)
                if not m:
                    continue
                fold = int(m.group(1))

                outputs_dir = fold_dir / "outputs"
                if not outputs_dir.is_dir():
                    continue
                for output_path in sorted(outputs_dir.glob("epoch_*_outputs.pt")):
                    if output_path.name.endswith(".bak_biopsie"):
                        continue  # historical backup files, skip
                    epoch = _parse_epoch(output_path.name)
                    if epoch is None:
                        continue
                    candidates.append(
                        CandidateFile(
                            output_path=output_path,
                            algorithm=algorithm,
                            da_weight=da_weight,
                            fold=fold,
                            epoch=epoch,
                        )
                    )

    log.info(
        "A4 pool: %d candidate output files across %d runs",
        len(candidates),
        len({c.algorithm for c in candidates}),
    )
    return candidates


# ---------------------------------------------------------------------------
# Loading one candidate into CheckpointData (+ sample_ids)
# ---------------------------------------------------------------------------


@dataclass
class LoadedCandidate:
    """CheckpointData plus target_val sample ids and probabilities."""

    checkpoint: CheckpointData
    target_val_sample_ids: list[str]
    target_val_probs: np.ndarray
    source: CandidateFile


def load_candidate(cand: CandidateFile) -> LoadedCandidate | None:
    """Load a single epoch output file and build a CheckpointData instance.

    Returns ``None`` if the file is missing any of the fields required by the
    validators (features, probabilities, labels).
    """
    import torch  # local import keeps the extraction utility lightweight

    try:
        data = torch.load(str(cand.output_path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to load %s: %s", cand.output_path, exc)
        return None

    source_val = data.get("source_val", {}) or {}
    target_val = data.get("target_val", {}) or {}

    target_features = np.asarray(target_val.get("features", []))
    target_probs = np.asarray(target_val.get("probabilities", []))
    target_labels = np.asarray(target_val.get("labels", []))
    target_predictions = np.asarray(target_val.get("predictions", []))
    source_probs = np.asarray(source_val.get("probabilities", []))
    source_labels = np.asarray(source_val.get("labels", []))

    if target_features.size == 0 or target_probs.size == 0:
        return None
    if target_labels.size == 0:
        return None
    if target_predictions.size == 0:
        target_predictions = target_probs.argmax(axis=1)

    sample_ids_raw = target_val.get("sample_ids") or []
    target_val_sample_ids = [str(s) for s in sample_ids_raw]

    checkpoint = CheckpointData(
        run_name=cand.output_path.parents[3].name,
        algorithm_id=AlgorithmId(algorithm=cand.algorithm),
        da_weight=cand.da_weight,
        epoch=cand.epoch,
        target_pair=TARGET_PAIR,
        source_probs=source_probs,
        source_labels=source_labels,
        target_features=target_features,
        target_probs=target_probs,
        target_labels=target_labels,
        target_predictions=target_predictions,
        source_val_bal_acc=0.0,
        target_val_bal_acc=0.0,
        fold=cand.fold,
    )
    return LoadedCandidate(
        checkpoint=checkpoint,
        target_val_sample_ids=target_val_sample_ids,
        target_val_probs=target_probs,
        source=cand,
    )


# ---------------------------------------------------------------------------
# Per-validator, per-fold selection
# ---------------------------------------------------------------------------


def _normalize_patient_id(sid: str) -> str:
    """Match mri_score_loader.load_mri_scores_from_outputs.

    Patient IDs are stored as 10-digit zero-padded strings everywhere else in
    the pipeline (pet.xlsx, pet volumes, B1/C1 runners).  Sample ids from the
    target_val dict may be integer-like strings or already padded; normalise
    to the canonical form.
    """
    s = str(sid).strip()
    if s.isdigit():
        return s.zfill(10)
    return s.zfill(10)


@dataclass
class FoldSelection:
    fold: int
    algorithm: str
    da_weight: float
    epoch: int
    score: float
    target_val_bal_acc: float
    n_patients: int


@dataclass
class ValidatorSelectionResult:
    validator: str
    fold_selections: list[FoldSelection] = field(default_factory=list)
    scores_by_patient: dict[str, float] = field(default_factory=dict)


def select_best_per_fold(
    candidates: list[CandidateFile],
    validator: Validator,
    pet_patient_ids: set[str] | None = None,
    *,
    verbose: bool = True,
) -> ValidatorSelectionResult:
    """For each fold, pick the candidate maximising ``validator.score``.

    Then extract the PET-subset patients' target_val probabilities from that
    winning checkpoint.

    Parameters
    ----------
    candidates
        Full A4 pool from :func:`enumerate_a4_candidates`.
    validator
        One of the six validator instances in :data:`VALIDATOR_REGISTRY`.
    pet_patient_ids
        If given, only patients in this set are written to the final score
        vector.  Typically the 26 / 25 PET cohort IDs.
    """
    from validator.validators import target_balanced_accuracy  # local import

    folds = sorted({c.fold for c in candidates})
    result = ValidatorSelectionResult(validator=getattr(validator, "name", type(validator).__name__))

    for fold in folds:
        fold_candidates = [c for c in candidates if c.fold == fold]
        best: LoadedCandidate | None = None
        best_score = -float("inf")

        for cand in fold_candidates:
            loaded = load_candidate(cand)
            if loaded is None:
                continue
            score = validator.score(loaded.checkpoint)
            if score > best_score:
                best_score = score
                best = loaded

        if best is None:
            log.warning(
                "validator=%s fold=%d: no valid candidates",
                result.validator,
                fold,
            )
            continue

        fold_bal_acc = target_balanced_accuracy(best.checkpoint)
        result.fold_selections.append(
            FoldSelection(
                fold=fold,
                algorithm=best.source.algorithm,
                da_weight=best.source.da_weight,
                epoch=best.source.epoch,
                score=float(best_score),
                target_val_bal_acc=float(fold_bal_acc),
                n_patients=len(best.target_val_sample_ids),
            )
        )
        if verbose:
            log.info(
                "  [%s] fold=%d  best=%s  da=%.1f  epoch=%03d  V=%.4f  target_bal_acc=%.3f",
                result.validator,
                fold,
                best.source.algorithm.upper(),
                best.source.da_weight,
                best.source.epoch,
                best_score,
                fold_bal_acc,
            )

        # Extract per-patient OOF score from this fold's target_val
        for sid, probs in zip(
            best.target_val_sample_ids, best.target_val_probs, strict=False
        ):
            pid = _normalize_patient_id(sid)
            if pet_patient_ids is not None and pid not in pet_patient_ids:
                continue
            # probs is (K,), with K=2 for binary csPCa classification.
            # Positive-class probability is the last column, matching Stage A.
            p_positive = float(np.asarray(probs)[-1])
            if pid in result.scores_by_patient:
                log.warning(
                    "duplicate patient %s in target_val across folds (validator=%s)",
                    pid,
                    result.validator,
                )
            result.scores_by_patient[pid] = p_positive

    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_all_validators(
    candidates: list[CandidateFile],
    pet_patient_ids: set[str] | None = None,
    validators: dict[str, Validator] | None = None,
    verbose: bool = True,
) -> dict[str, ValidatorSelectionResult]:
    """Select one checkpoint per fold for every validator.

    Returns a dict ``{short_name: ValidatorSelectionResult}``.
    """
    validators = validators or VALIDATOR_REGISTRY
    results: dict[str, ValidatorSelectionResult] = {}
    for short_name, validator in validators.items():
        if verbose:
            log.info("Selecting checkpoints for validator=%s", short_name)
        results[short_name] = select_best_per_fold(
            candidates,
            validator,
            pet_patient_ids=pet_patient_ids,
            verbose=verbose,
        )
    return results


def save_selection_cache(
    results: dict[str, ValidatorSelectionResult],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Persist per-validator score vectors + audit records.

    For each validator creates two files under ``cache_dir``:
      - ``{short_name}.json``      : ``{patient_id: score}`` (the OOF vector).
      - ``{short_name}.selection.json`` : per-fold (method, da_weight, epoch).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    for short_name, res in results.items():
        scores_path = cache_dir / f"{short_name}.json"
        with open(scores_path, "w") as f:
            json.dump(res.scores_by_patient, f, indent=2, sort_keys=True)

        selection_path = cache_dir / f"{short_name}.selection.json"
        selection_payload = {
            "validator": res.validator,
            "fold_selections": [
                {
                    "fold": fs.fold,
                    "algorithm": fs.algorithm,
                    "da_weight": fs.da_weight,
                    "epoch": fs.epoch,
                    "score": fs.score,
                    "target_val_bal_acc": fs.target_val_bal_acc,
                    "n_patients": fs.n_patients,
                }
                for fs in res.fold_selections
            ],
        }
        with open(selection_path, "w") as f:
            json.dump(selection_payload, f, indent=2)

        log.info(
            "Cached %s: %d patients -> %s",
            short_name,
            len(res.scores_by_patient),
            scores_path,
        )


def load_scores(short_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, float]:
    """Return the cached ``{patient_id: score}`` map for *short_name*."""
    path = cache_dir / f"{short_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached scores for validator={short_name} at {path}.  "
            "Run scripts/a4_pool_score_extraction.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_selection(short_name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Return the audit record for *short_name* (per-fold selections)."""
    path = cache_dir / f"{short_name}.selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No selection cache for validator={short_name} at {path}."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _read_pet_patient_ids() -> set[str]:
    """Best-effort load of the 25/26 PET patient IDs for filtering score vectors.

    If ``0ii/pet.xlsx`` cannot be read (e.g. openpyxl missing), we return an
    empty set -- the extraction then writes the full UULM target_val score
    for every patient the validator-selected checkpoint covers.  The B1/C1
    runners downstream intersect with the pet.xlsx cohort, so this is safe.
    """
    import pandas as pd

    pet_xlsx = PROJECT_ROOT / "0ii" / "pet.xlsx"
    try:
        df = pd.read_excel(pet_xlsx, sheet_name="Auswertung")
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot read %s (%s); returning empty pet-id set", pet_xlsx, exc)
        return set()
    ids = {
        str(int(x)).zfill(10)
        for x in df["PatientenID"].dropna().tolist()
    }
    log.info("PET cohort: %d unique patient IDs in %s", len(ids), pet_xlsx.name)
    return ids


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Select one A4 checkpoint per fold per validator and cache the "
            "resulting PET-subset OOF scores for downstream B1/C1 LOOCV fusion."
        )
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=A4_COLLECTION,
        help="Root directory of the A4 run collection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Where to write {validator}.json and {validator}.selection.json.",
    )
    parser.add_argument(
        "--skip-pet-filter",
        action="store_true",
        help=(
            "If set, do not filter score vectors to the pet.xlsx cohort.  "
            "The B1/C1 runners will still intersect downstream."
        ),
    )
    parser.add_argument(
        "--validators",
        nargs="+",
        choices=sorted(VALIDATOR_REGISTRY.keys()),
        default=None,
        help="Subset of validators to process (default: all six).",
    )
    args = parser.parse_args()

    _configure_logging()

    candidates = enumerate_a4_candidates(args.collection)
    if not candidates:
        log.error("No candidates found; aborting.")
        return 1

    pet_patient_ids = None if args.skip_pet_filter else _read_pet_patient_ids()
    validators = (
        VALIDATOR_REGISTRY
        if args.validators is None
        else {k: VALIDATOR_REGISTRY[k] for k in args.validators}
    )

    results = run_all_validators(
        candidates,
        pet_patient_ids=pet_patient_ids or None,
        validators=validators,
        verbose=True,
    )
    save_selection_cache(results, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
