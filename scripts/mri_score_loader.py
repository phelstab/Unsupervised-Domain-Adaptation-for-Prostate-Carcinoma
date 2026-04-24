"""Shared MRI score loading utilities for B1/C1 post-hoc sidecar classifiers.

Provides functions to load pre-computed MRI P(csPCa) predictions from:
  - A full experiment run directory (averaging across folds)
  - A single ``epoch_*_outputs.pt`` file
  - A saved checkpoint (re-running inference)
  - Dummy scores for dry-run testing

These are used by both B1 (PET metadata sidecar) and C1 (PET radiomics fusion).
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "MRI" / "baseline" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths used by checkpoint inference (Mode 3)
# ---------------------------------------------------------------------------
PET_XLSX = PROJECT_ROOT / "0ii" / "pet.xlsx"
PET_SHEET = "Auswertung"
MAN_XLSX = PROJECT_ROOT / "0ii" / "man.xlsx"
COL_PATIENT_ID = "PatientenID"
COL_IPSA = "iPSA"
COL_UPSA = "uPSA"
COL_PROSTATE_VOL = "Prostatavolumen (ml)"
COL_AGE = "Alter zum Zeitpunkt der Untersuchung"


# ---------------------------------------------------------------------------
# Epoch selection
# ---------------------------------------------------------------------------


def select_best_epoch(output_files: list[Path], validator: str) -> Path:
    """Select the best epoch file based on a validator metric.

    For ``source_val``: pick epoch with highest source_val balanced accuracy.
    For ``oracle``: pick epoch with highest target_val balanced accuracy.
    """
    import torch

    best_score = -1.0
    best_file = output_files[-1]  # fallback to last

    for fpath in output_files:
        data = torch.load(str(fpath), map_location="cpu", weights_only=False)

        if validator == "source_val":
            split = data.get("source_val")
        elif validator == "oracle":
            split = data.get("target_val")
        else:
            split = data.get("source_val")

        if split is None:
            continue

        preds = split["predictions"]
        labels = split["labels"]
        if not isinstance(preds, np.ndarray):
            preds = preds.numpy()
        if not isinstance(labels, np.ndarray):
            labels = labels.numpy()

        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_file = fpath

    return best_file


# ---------------------------------------------------------------------------
# Mode 1: Load from run directory (preferred)
# ---------------------------------------------------------------------------


def load_mri_scores_from_run_dir(
    run_dir: Path,
    da_weight: str = "0.9",
    epoch: str | None = None,
    validator: str = "source_val",
) -> dict[str, float]:
    """Load MRI scores from a full run directory using out-of-fold predictions.

    The run directory structure is::

        <run_dir>/RUMC+PCNN+ZGT_to_UULM/da_{weight}/fold_{i}/outputs/epoch_*_outputs.pt

    For each fold, selects the best epoch by *validator* (default: source_val,
    which picks the epoch with highest source validation balanced accuracy),
    then collects each patient's **out-of-fold (OOF) prediction only** — i.e.
    the ``target_val`` split of the fold where that patient was held out from
    UDA training.

    .. note:: Prior to this fix, scores were averaged across *both*
       ``target_train`` and ``target_val`` splits for every fold, meaning each
       patient's final score was a mix of in-fold predictions (where the UDA
       model had adapted its feature extractor to that patient's MRI as
       unlabeled data) and the single held-out prediction.  With K=3 folds
       this produced scores that were 67 % in-fold / 33 % out-of-fold —
       an optimistic bias analogous to the train-set leakage described in
       stacked generalization (Wolpert 1992, Breiman 1996).

       Demircioğlu (2021) "Measuring the bias of incorrect application of
       feature selection when using cross-validation in radiomics" (Insights
       into Imaging, doi:10.1186/s13244-021-01105-5) showed that information
       leaking across CV folds inflates AUC by 0.05–0.20 in radiomics
       pipelines, especially at small sample sizes.  Samala et al. (2021)
       "Risks of feature leakage and sample size dependencies in deep feature
       extraction" (Medical Physics, doi:10.1002/mp.14781) demonstrated the
       same effect when deep-learning features are extracted from images that
       the model was exposed to during training.

       The correct protocol is to use **only** the ``target_val`` prediction
       for each patient — the single fold where the patient was unseen during
       UDA adaptation.  This yields one clean OOF score per patient, which is
       then passed downstream to the B1/C1 sidecar classifiers.

    Parameters
    ----------
    run_dir : Path
        Root of a single experiment run (e.g., ``20260324_..._dann_...``).
    da_weight : str
        DA loss weight subdirectory (e.g., "0.1", "0.5", "0.9").
    epoch : str or None
        If given, use this specific epoch (e.g., "0099") instead of validator
        selection.  ``"last"`` uses the last saved epoch.
    validator : str
        Which validator to use for epoch selection.  "source_val" picks the
        epoch with highest source validation balanced accuracy.  "oracle"
        picks the epoch with highest target validation balanced accuracy.
    """
    import torch

    domain_dir = run_dir / "RUMC+PCNN+ZGT_to_UULM" / f"da_{da_weight}"
    if not domain_dir.exists():
        candidates = list((run_dir / "RUMC+PCNN+ZGT_to_UULM").glob(f"da_{da_weight}*"))
        if candidates:
            domain_dir = candidates[0]
        else:
            raise FileNotFoundError(f"DA weight directory not found: {domain_dir}")

    fold_dirs = sorted(domain_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories in {domain_dir}")

    log.info(
        "Loading MRI scores from %s (da_%s, %d folds, validator=%s)",
        run_dir.name,
        da_weight,
        len(fold_dirs),
        validator,
    )

    per_patient_scores: dict[str, list[float]] = defaultdict(list)

    for fold_dir in fold_dirs:
        outputs_dir = fold_dir / "outputs"
        if not outputs_dir.exists():
            log.warning("No outputs dir in %s", fold_dir)
            continue

        output_files = sorted(outputs_dir.glob("epoch_*_outputs.pt"))
        if not output_files:
            log.warning("No output files in %s", outputs_dir)
            continue

        # Select which epoch to use
        if epoch == "last" or (epoch is None and validator == "last"):
            selected_file = output_files[-1]
        elif epoch is not None:
            target = f"epoch_{epoch.zfill(4)}_outputs.pt"
            selected_file = outputs_dir / target
            if not selected_file.exists():
                log.warning("Epoch %s not found in %s, using last", epoch, outputs_dir)
                selected_file = output_files[-1]
        else:
            selected_file = select_best_epoch(output_files, validator)

        log.info("  %s: using %s", fold_dir.name, selected_file.name)
        data = torch.load(str(selected_file), map_location="cpu", weights_only=False)

        # Collect ONLY out-of-fold (target_val) predictions.
        #
        # In K-fold CV each patient appears in target_val for exactly one fold
        # (held out from UDA training) and in target_train for the remaining
        # K-1 folds (used as unlabeled data for domain alignment).
        #
        # Using only target_val avoids the optimistic bias that results from
        # including predictions where the UDA feature extractor was adapted to
        # the patient's MRI during training.  See docstring references:
        #   - Demircioğlu 2021 (Insights Imaging, doi:10.1186/s13244-021-01105-5)
        #   - Samala et al. 2021 (Med Phys, doi:10.1002/mp.14781)
        #   - Wolpert 1992 (stacked generalization)
        split_name = "target_val"
        if split_name not in data:
            log.warning("No '%s' split in %s", split_name, selected_file.name)
            continue
        split = data[split_name]
        sample_ids = split["sample_ids"]
        probs = split["probabilities"]
        if not isinstance(probs, np.ndarray):
            probs = probs.numpy()
        p_positive = probs[:, -1]
        for sid, p in zip(sample_ids, p_positive):
            per_patient_scores[sid].append(float(p))

    # With K-fold CV each patient has exactly one OOF score (from the single
    # fold where they were in target_val).  No averaging needed, but we keep
    # np.mean for robustness in case a patient appears in multiple val folds
    # due to non-standard splitting.
    scores: dict[str, float] = {}
    for pid, fold_scores in per_patient_scores.items():
        scores[pid] = float(np.mean(fold_scores))

    log.info(
        "Loaded OOF MRI scores for %d target patients from %d folds",
        len(scores),
        len(fold_dirs),
    )
    return scores


# ---------------------------------------------------------------------------
# Mode 2: Load from a single epoch outputs file
# ---------------------------------------------------------------------------


def load_mri_scores_from_outputs(outputs_path: Path) -> dict[str, float]:
    """Load MRI p(csPCa) scores from a saved ``epoch_*_outputs.pt`` file.

    Extracts ``target_val`` probabilities only and returns a dict mapping
    patient_id -> P(positive).

    This mode should only be used with outputs files produced from a target-CV
    fold, where ``target_val`` contains the held-out out-of-fold patients.
    ``target_train`` predictions are intentionally ignored because they are
    in-fold target samples that were seen by UDA adaptation.
    """
    import torch

    log.info("Loading MRI scores from %s", outputs_path)
    outputs = torch.load(str(outputs_path), map_location="cpu", weights_only=False)

    scores: dict[str, float] = {}
    split_name = "target_val"
    if split_name not in outputs:
        raise KeyError(
            f"Split '{split_name}' not found in outputs. Use a target-CV outputs "
            "file with held-out target_val predictions."
        )

    split = outputs[split_name]
    sample_ids = split["sample_ids"]
    probs = split["probabilities"]
    if not isinstance(probs, np.ndarray):
        probs = probs.numpy()

    p_positive = probs[:, -1]
    for sid, p in zip(sample_ids, p_positive):
        pid = str(int(sid)).zfill(10) if sid.isdigit() else str(sid).zfill(10)
        scores[pid] = float(p)

    log.info("Loaded MRI scores for %d target patients", len(scores))
    return scores


# ---------------------------------------------------------------------------
# Mode 3: Load from checkpoint (re-run inference)
# ---------------------------------------------------------------------------


def load_mri_scores_from_checkpoint(
    checkpoint_path: Path,
    backbone: str = "resnet10",
    num_classes: int = 2,
    variant: str = "prostate_clinical",
    device: str = "cpu",
) -> dict[str, float]:
    """Run inference on PET subset using a saved checkpoint.

    Loads the model, loads preprocessed MRI data for each PET patient,
    and returns MRI p(csPCa) scores.
    """
    import torch
    from cnn.model import create_model

    log.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    use_prostate_prior = "prostate" in variant
    use_clinical = "clinical" in variant
    num_channels = 4 if use_prostate_prior else 3
    clinical_feature_dim = 4 if use_clinical else 0

    model = create_model(
        backbone=backbone,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_rate=0.0,
        use_batchnorm=True,
        clinical_feature_dim=clinical_feature_dim,
        clinical_fusion="early",
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load PET patient list from pet.xlsx
    pet_df = pd.read_excel(PET_XLSX, sheet_name=PET_SHEET)
    pet_df[COL_PATIENT_ID] = pet_df[COL_PATIENT_ID].apply(
        lambda x: str(int(x)).zfill(10) if pd.notna(x) else None
    )

    # Load man.xlsx for study_id mapping
    man_df = pd.read_excel(MAN_XLSX)
    man_pid_col = "PatientenID"
    man_study_col = "AuftragsID"
    man_df[man_pid_col] = man_df[man_pid_col].apply(
        lambda x: str(int(x)).zfill(10) if pd.notna(x) else None
    )
    pid_to_study = dict(zip(man_df[man_pid_col], man_df[man_study_col].astype(str)))

    # Also add mappings from pet.xlsx (has its own AuftragsID)
    pet_study_col = "AuftragsID"
    for _, row in pet_df.iterrows():
        pid = row[COL_PATIENT_ID]
        sid = row.get(pet_study_col)
        if pid and pd.notna(sid):
            pid_to_study[pid] = str(int(sid))

    preprocessed_dir = PROJECT_ROOT / "0ii" / "files" / "registered"
    prior_dir = (
        PROJECT_ROOT
        / "workdir"
        / "prostate_prior_cache"
        / "target"
        / "soft_whole_gland"
    )

    scores: dict[str, float] = {}
    for pid in pet_df[COL_PATIENT_ID].dropna().unique():
        study_id = pid_to_study.get(pid)
        if study_id is None:
            log.warning("No study_id for patient %s, skipping", pid)
            continue

        mri_path = preprocessed_dir / pid / f"{pid}_{study_id}_registered.npy"
        if not mri_path.exists():
            log.warning("MRI not found: %s", mri_path)
            continue

        data = np.load(str(mri_path)).astype(np.float32)

        if use_prostate_prior:
            prior_path = prior_dir / f"{pid}_{study_id}.npy"
            if prior_path.exists():
                prior = np.load(str(prior_path)).astype(np.float32)
                if prior.ndim == 3:
                    prior = prior[np.newaxis]
                data = np.concatenate([data, prior], axis=0)
            else:
                log.warning("Prior not found: %s, using zeros", prior_path)
                zeros = np.zeros((1, *data.shape[1:]), dtype=np.float32)
                data = np.concatenate([data, zeros], axis=0)

        input_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)

        clinical_tensor = None
        if use_clinical:
            row = pet_df[pet_df[COL_PATIENT_ID] == pid].iloc[0]
            psa_val = pd.to_numeric(row.get(COL_IPSA), errors="coerce")
            if pd.isna(psa_val):
                psa_val = pd.to_numeric(row.get(COL_UPSA), errors="coerce")
            vol_val = pd.to_numeric(row.get(COL_PROSTATE_VOL), errors="coerce")
            psad_val = (
                psa_val / vol_val
                if pd.notna(psa_val) and pd.notna(vol_val) and vol_val > 0
                else np.nan
            )
            age_val = pd.to_numeric(row.get(COL_AGE), errors="coerce")

            feats = []
            for val in [psa_val, psad_val, vol_val, age_val]:
                if pd.isna(val):
                    feats.append(0.0)
                elif val in [psa_val, psad_val, vol_val] and val is not age_val:
                    feats.append(float(np.log1p(max(0.0, val))))
                else:
                    feats.append(float(val))

            clinical_tensor = (
                torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            )

        with torch.no_grad():
            outputs = model(input_tensor, clinical_features=clinical_tensor)
            logits = outputs["classification"]
            probs = torch.softmax(logits, dim=1)
            p_positive = probs[0, -1].item()

        scores[pid] = p_positive
        log.info("  %s -> p(csPCa) = %.4f", pid, p_positive)

    log.info("Computed MRI scores for %d patients", len(scores))
    return scores


# ---------------------------------------------------------------------------
# Mode 4: Dummy scores for dry-run testing
# ---------------------------------------------------------------------------


def generate_dummy_mri_scores(
    patient_ids: list[str],
    labels: pd.Series | dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Generate random MRI scores for dry-run testing.

    If *labels* is provided, scores are weakly correlated with true labels
    (base 0.6 for positive, 0.3 for negative). Otherwise, random ~N(0.4, 0.2).
    """
    rng = np.random.RandomState(seed)
    scores: dict[str, float] = {}
    for pid in patient_ids:
        if labels is not None:
            lab = labels[pid] if isinstance(labels, dict) else labels.get(pid)
            if pd.notna(lab):
                base = 0.6 if lab == 1 else 0.3
                scores[pid] = float(np.clip(base + rng.normal(0, 0.2), 0.01, 0.99))
        else:
            scores[pid] = float(np.clip(rng.normal(0.4, 0.2), 0.01, 0.99))
    log.info("Generated dummy MRI scores for %d patients", len(scores))
    return scores
