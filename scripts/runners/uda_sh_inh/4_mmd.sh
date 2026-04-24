#!/usr/bin/env bash
# MMD UDA Experiments for merged public source -> UULM target
cd "$(dirname "$0")/../../.."
echo "Working directory: $(pwd)"

PYTHON=".venv-cnn/bin/python"
SCRIPT="scripts/runners/1_cnn_uda_runner.py"

BACKBONE="resnet10"
LR_SCHEDULER="--lr-scheduler inv"
SPLIT="--dataset-profile public_to_uulm --center-pairs RUMC+PCNN+ZGT_to_UULM"
UULM_META="--uulm-use-dual-metadata --uulm-label-file 0ii/man.xlsx --uulm-pet-label-file 0ii/pet.xlsx --uulm-pet-sheet-name Auswertung"
COMMON="--da-method mmd --binary --backbone $BACKBONE --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --target-cv-folds 3 --class-weights $LR_SCHEDULER $SPLIT $UULM_META --aug-all"

echo "========================================================"
echo "MMD UDA Experiments (RUMC+PCNN+ZGT -> UULM)"
echo "Total full-reg runs: 3 DA weights x 1 center pair = 3 runs"
echo "========================================================"

# PHASE 1: Simple baseline
# $PYTHON $SCRIPT $COMMON --dropout 0.0 --no-batchnorm --weight-decay 0.0

# PHASE 2: Full regularization
$PYTHON $SCRIPT $COMMON --dropout 0.3 --weight-decay 1e-4

echo
echo "All MMD experiments completed!"
