"""Shared constants for validator table generation."""

DEFAULT_REPORT_NAME = "validator_table.md"

KNOWN_BACKBONES = (
    "RESNET10",
    "RESNET18",
    "RESNET34",
    "RESNET50",
)

KNOWN_ALGORITHMS = (
    "MCC",
    "BNM",
    "HYBRID",
    "MMD",
    "DANN",
    "MCD",
    "CORAL",
    "ENTROPY",
    "DAARDA",
)

REGULARIZATION_TITLES = {
    "baseline": "Baseline",
    "regularized": "Regularized",
}

REGULARIZATION_DESCRIPTIONS = {
    "baseline": "No regularization markers in the run name.",
    "regularized": (
        "Detected regularization markers such as dropout or weight decay."
    ),
}
