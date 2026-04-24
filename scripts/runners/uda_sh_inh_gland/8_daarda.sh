#!/usr/bin/env bash
# DAARDA gland-prior UDA Experiments for merged public source -> UULM target
cd "$(dirname "$0")/../../.."
echo "Working directory: $(pwd)"

PYTHON=".venv-cnn/bin/python"
SCRIPT="scripts/runners/1_cnn_uda_runner.py"

BACKBONE="resnet10"
LR_SCHEDULER="--lr-scheduler inv"
SPLIT="--dataset-profile public_to_uulm --center-pairs RUMC+PCNN+ZGT_to_UULM"
DAARDA_ARGS="--daarda-divergence js_beta --daarda-relax 1.5 --daarda-grad-penalty 0.0"
UULM_META="--uulm-use-dual-metadata --uulm-label-file 0ii/man.xlsx --uulm-pet-label-file 0ii/pet.xlsx --uulm-pet-sheet-name Auswertung"
GLAND_ARGS="--model-variant prostate_prior --prostate-prior-type whole_gland --prostate-prior-source bosma22b --prostate-prior-target pseudo --prostate-prior-target-dir 0ii/files/gland_masks --prostate-prior-cache-dir workdir/prostate_prior_cache"
COMMON="--da-method daarda --binary --backbone $BACKBONE --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --target-cv-folds 3 --class-weights $LR_SCHEDULER $SPLIT $UULM_META $GLAND_ARGS --aug-all $DAARDA_ARGS"

echo "========================================================"
echo "DAARDA GLAND-PRIOR UDA Experiments (RUMC+PCNN+ZGT -> UULM)"
echo "ARDA settings: divergence=js_beta, relax=1.5, grad_penalty=0.0"
echo "========================================================"

$PYTHON $SCRIPT $COMMON --dropout 0.3 --weight-decay 1e-4

echo
echo "All DAARDA gland-prior experiments completed!"
