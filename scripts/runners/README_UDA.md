# UDA CNN Runner

Unsupervised Domain Adaptation for prostate cancer ISUP grading using center-based domain splits.

## Quick Start

```bash
# Quick test (3 configs)
python scripts\runners\1_cnn_uda_runner.py --quick

# Full test (7 configs)
python scripts\runners\1_cnn_uda_runner.py

# Specify GPU
python scripts\runners\1_cnn_uda_runner.py --gpu_id 0
```

## Features

- **Center-based domain adaptation**: RUMC → PCNN and RUMC → ZGT
- **Progressive sample sizes**: 1, 5, 10, 50, 100, 200, full dataset
- **CORAL alignment**: Covariance-based domain adaptation
- **Domain discriminator**: Validation of feature alignment (target: ~50% accuracy)
- **Full metrics**: Macro/micro F1, precision, recall for all classes
- **TensorBoard logging**: All metrics tracked per epoch
- **Dated log files**: CLI output saved to `1_cnn_uda_runner_YYYYMMDD_HHMMSS.txt`

## Output Structure

```
workdir/uda/YYYYMMDD_HHMMSS/
├── 1_cnn_uda_runner_YYYYMMDD_HHMMSS.txt   # CLI output
├── results.json                             # All experiment results
├── summary.txt                              # Summary report
├── RUMC_to_PCNN/
│   └── tensorboard/                         # TensorBoard logs
└── RUMC_to_ZGT/
    └── tensorboard/                         # TensorBoard logs
```

## Test Configurations

### Quick (--quick flag)
- 1 sample: 50 epochs, lr=0.01, bs=1, no CORAL
- 5 samples: 30 epochs, lr=0.01, bs=1, no CORAL
- 50 samples: 20 epochs, lr=0.001, bs=4, CORAL weight=0.5

### Full (default)
- 1 sample: 500 epochs, no CORAL (baseline test)
- 5 samples: 500 epochs, no CORAL
- 10 samples: 300 epochs, no CORAL
- 50 samples: 200 epochs, no CORAL
- 100 samples: 150 epochs, CORAL weight=0.5
- 200 samples: 100 epochs, CORAL weight=0.5
- Full dataset: 100 epochs, CORAL weight=0.5

## Metrics

### Per Epoch (TensorBoard)
- **Train**: Loss, accuracy, classification loss, CORAL loss, discriminator accuracy
- **Validation (source)**: Loss, accuracy, macro/micro F1/precision/recall
- **Target**: Accuracy (unsupervised), macro/micro F1/precision/recall

### Final Results
- Source validation accuracy
- Target domain accuracy (ground truth for validation only)
- Best macro/micro F1 scores
- Domain discriminator accuracy (50% = perfect alignment)
- Training time

## Domain Adaptation Components

1. **Feature Encoder**: 3D CNN extracting 512-d features
2. **CORAL Loss**: Aligns source/target feature covariances
3. **Domain Discriminator**: Validates alignment (trained adversarially)
4. **Classifier**: ISUP grade prediction (6 classes: 0-5)
