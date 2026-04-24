#!/usr/bin/env bash
# Main runner: Execute all UDA algorithm experiments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
echo "========================================================"
echo "MAIN UDA EXPERIMENT RUNNER"
echo "Working directory: $(pwd)"
echo "========================================================"
echo ""
echo "Algorithms: MCD, DANN, MMD, HYBRID, MCC, BNM"
echo "Each: 2 phases x 2 da_weights x 2 center pairs = 6 runs"
echo "Total: 6 algorithms x 6 runs = 36 experiments"
echo "========================================================"
echo ""


# echo "[5/6] Running MCC experiments..."
# bash "$SCRIPT_DIR/6_mcc.sh"

# echo "[6/6] Running BNM experiments..."
# bash "$SCRIPT_DIR/7_bnm.sh"

echo "[1/6] Running MCD experiments..."
bash "$SCRIPT_DIR/2_mcd.sh"

# echo "[3/6] Running MMD experiments..."
# bash "$SCRIPT_DIR/4_mmd.sh"

# echo "[2/6] Running DANN experiments..."
# bash "$SCRIPT_DIR/3_dann.sh"

# echo "[4/6] Running HYBRID experiments..."
# bash "$SCRIPT_DIR/5_hybrid.sh"

echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================"
read -p "Press Enter to continue..."
