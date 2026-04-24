#!/bin/bash
# PI-CAI Baseline Directory Setup Script for Linux/macOS

echo "Setting up PI-CAI baseline directory structure..."

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Root directory: $ROOT_DIR"

echo "Creating main directories..."

mkdir -p "$ROOT_DIR/workdir/nnUNet_raw_data"
mkdir -p "$ROOT_DIR/workdir/nnDet_raw_data"
mkdir -p "$ROOT_DIR/workdir/results/UNet/weights"
mkdir -p "$ROOT_DIR/workdir/results/UNet/overviews"
mkdir -p "$ROOT_DIR/workdir/results/nnUNet"
mkdir -p "$ROOT_DIR/workdir/results/nnDet"
mkdir -p "$ROOT_DIR/workdir/splits"
mkdir -p "$ROOT_DIR/input/images"

echo "Directory structure created successfully!"
echo ""
echo "Created directories:"
echo "   workdir/                    - Working directory for processing"
echo "   workdir/nnUNet_raw_data/    - nnU-Net preprocessed data"
echo "   workdir/nnDet_raw_data/     - nnDetection preprocessed data"
echo "   workdir/results/            - Training results and models"
echo "   workdir/splits/             - Cross-validation splits"
echo "   input/                      - Input data directory"
echo "   input/images/               - PI-CAI dataset images"
echo "   output/                     - Final outputs"
echo "   logs/                       - Training and processing logs"
echo ""
echo "Next steps:"
echo "   1. Download PI-CAI dataset to input/images/"
echo "   2. Ensure picai_labels is available in Datasets/MRI/PI-CAI_Challenge_2024/"
echo "   3. Run data preparation scripts"
echo ""
echo "Usage: ./setup_directories.sh"
