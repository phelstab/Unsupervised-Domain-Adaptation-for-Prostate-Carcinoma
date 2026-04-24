#!/usr/bin/env bash
# Main runner: Execute all gland-prior UDA experiments for merged public -> UULM
cd "$(dirname "$0")/../../.."
echo "========================================================"
echo "MAIN UDA GLAND-PRIOR RUNNER"
echo "Working directory: $(pwd)"
echo "Split: RUMC+PCNN+ZGT -> UULM"
echo "Variant: prostate_prior (whole_gland)"
echo "Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM, DAARDA"
echo "========================================================"
echo

echo "[2/7] Running DANN gland-prior experiments..."
bash "$(dirname "$0")/3_dann.sh"

echo "[3/7] Running MMD gland-prior experiments..."
bash "$(dirname "$0")/4_mmd.sh"

echo "[1/7] Running MCD gland-prior experiments..."
bash "$(dirname "$0")/2_mcd.sh"

echo "[4/7] Running HYBRID gland-prior experiments..."
bash "$(dirname "$0")/5_hybrid.sh"

echo "[5/7] Running MCC gland-prior experiments..."
bash "$(dirname "$0")/6_mcc.sh"

echo "[6/7] Running BNM gland-prior experiments..."
bash "$(dirname "$0")/7_bnm.sh"

echo "[7/7] Running DAARDA gland-prior experiments..."
bash "$(dirname "$0")/8_daarda.sh"

echo
echo "========================================================"
echo "ALL GLAND-PRIOR EXPERIMENTS COMPLETED!"
echo "========================================================"
