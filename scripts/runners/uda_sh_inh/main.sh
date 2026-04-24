#!/usr/bin/env bash
# Main runner: Execute all UDA algorithm experiments for merged public -> UULM
cd "$(dirname "$0")/../../.."
echo "========================================================"
echo "MAIN UDA EXPERIMENT RUNNER (NEW SPLIT)"
echo "Working directory: $(pwd)"
echo "Split: RUMC+PCNN+ZGT -> UULM"
echo "Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM, DAARDA"
echo "Total full-reg runs: 7 algorithms x 3 DA weights x 1 pair x 5 folds = 105 experiments"
echo "========================================================"
echo

echo "[2/7] Running DANN experiments..."
bash "$(dirname "$0")/3_dann.sh"

echo "[3/7] Running MMD experiments..."
bash "$(dirname "$0")/4_mmd.sh"

echo "[1/7] Running MCD experiments..."
bash "$(dirname "$0")/2_mcd.sh"

echo "[4/7] Running HYBRID experiments..."
bash "$(dirname "$0")/5_hybrid.sh"

echo "[5/7] Running MCC experiments..."
bash "$(dirname "$0")/6_mcc.sh"

echo "[6/7] Running BNM experiments..."
bash "$(dirname "$0")/7_bnm.sh"

echo "[7/7] Running DAARDA experiments..."
bash "$(dirname "$0")/8_daarda.sh"

echo
echo "========================================================"
echo "ALL NEW-SPLIT EXPERIMENTS COMPLETED!"
echo "========================================================"
