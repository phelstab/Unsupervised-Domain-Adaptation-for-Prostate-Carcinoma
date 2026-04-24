#!/usr/bin/env bash
# HYBRID (CORAL + Entropy Minimization) UDA Experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
echo "Working directory: $(pwd)"

PYTHON=.venv-cnn/bin/python
SCRIPT=scripts/runners/1_cnn_uda_runner.py

BACKBONE=resnet18
# LR Scheduler (always enabled - not regularization, makes learning better)
LR_SCHEDULER="--lr-scheduler inv"
# 2-split mode: 85/15 train/eval (supervisor recommended for limited data)
COMMON="--da-method hybrid --binary --backbone $BACKBONE --plot-oracle --checkpoint-interval 2 --no-early-stopping --two-splits-source --two-splits-target $LR_SCHEDULER --aug-all"

echo "========================================================"
echo "HYBRID (CORAL + Entropy Min) UDA Experiments"
echo "Phase 1: Simple baseline (no regularization)"
echo "Phase 2: Full regularization (dropout + batchnorm + weight decay + LR scheduler)"
echo "Each phase tests da_weight: 0.1, 0.5, 1.0 on 2 center pairs"
echo "Total: 2 phases x 3 da_weights x 2 pairs = 12 runs"
echo "========================================================"

# PHASE 1: Simple baseline
echo ""
# echo "[PHASE 1/2] Simple baseline (LR scheduler always active per supervisor)"
# $PYTHON $SCRIPT $COMMON --dropout 0.0 --no-batchnorm --weight-decay 0.0

# PHASE 2: Full regularization
echo ""
echo "[PHASE 2/2] Full regularization"
$PYTHON $SCRIPT $COMMON --dropout 0.3 --weight-decay 1e-4

echo ""
echo "All HYBRID experiments completed! (12 runs)"
