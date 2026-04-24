# Dynamic Class Weights

## Problem

The source dataset has an imbalanced label distribution (~70% benign, ~30% csPCa).
Without weighting, the model can achieve decent accuracy by simply predicting the majority class,
which hurts sensitivity for clinically significant prostate cancer.

## Solution

We weight the cross-entropy loss inversely proportional to class frequency in the source training set:

```
weight(class) = total_samples / (num_classes × count(class))
```

## Example (binary, ~70/30 split)

| Class | Count | Weight | Interpretation |
|---|---|---|---|
| 0 (benign) | 700 | 1000 / (2 × 700) = **0.71** | Majority → lower weight |
| 1 (csPCa) | 300 | 1000 / (2 × 300) = **1.67** | Minority → higher weight |

The minority class gets ~2.3× higher weight, so misclassifying a positive case incurs a larger loss.

## Usage

Add `--class-weights` to the runner command:

```bat
%PYTHON% %SCRIPT% --class-weights --binary ...
```

## Notes

- Weights are computed dynamically from `source_train` labels each run — no hardcoded values.
- Only source labels are used (target labels are unavailable in UDA).
- As the advisor noted: this helps but is not a complete solution for imbalance on its own.
