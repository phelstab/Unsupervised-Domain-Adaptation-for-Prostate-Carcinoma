"""Model-selection validators used to score checkpoint outputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module

from .models import CheckpointData, ValidatorResult


def balanced_accuracy_score(y_true, y_pred) -> float:
    """Compute balanced accuracy without requiring scikit-learn."""
    numpy = import_module("numpy")
    true_values = numpy.asarray(y_true)
    pred_values = numpy.asarray(y_pred)
    if true_values.size == 0:
        return 0.0

    recalls: list[float] = []
    for label in numpy.unique(true_values):
        label_mask = true_values == label
        label_total = int(label_mask.sum())
        if label_total == 0:
            continue
        label_correct = int((pred_values[label_mask] == label).sum())
        recalls.append(label_correct / label_total)

    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def target_balanced_accuracy(checkpoint: CheckpointData) -> float:
    """Compute target balanced accuracy in percent."""
    if checkpoint.target_labels.size == 0:
        return 0.0
    return (
        balanced_accuracy_score(
            checkpoint.target_labels,
            checkpoint.target_predictions,
        )
        * 100.0
    )


class Validator(ABC):
    """Base class for checkpoint selection validators."""

    name: str

    @abstractmethod
    def score(self, checkpoint: CheckpointData) -> float:
        """Return a score where higher values indicate better checkpoints."""

    def select_best(
        self,
        checkpoints: list[CheckpointData],
    ) -> ValidatorResult | None:
        """Select the highest-scoring checkpoint from a pool."""
        if not checkpoints:
            return None

        best_checkpoint = max(checkpoints, key=self.score)
        best_score = self.score(best_checkpoint)
        return ValidatorResult(
            checkpoint=best_checkpoint,
            score=best_score,
            target_bal_acc=target_balanced_accuracy(best_checkpoint),
        )


class OracleValidator(Validator):
    """Upper-bound validator that uses target labels."""

    name = "Oracle"

    def score(self, checkpoint: CheckpointData) -> float:
        return target_balanced_accuracy(checkpoint)


class SrcAccValidator(Validator):
    """Select using source validation balanced accuracy."""

    name = "Src-Acc"

    def score(self, checkpoint: CheckpointData) -> float:
        if checkpoint.source_labels.size == 0:
            return 0.0
        predictions = checkpoint.source_probs.argmax(axis=1)
        return float(
            balanced_accuracy_score(checkpoint.source_labels, predictions)
        )


class EntropyValidator(Validator):
    """Prefer low-entropy target predictions."""

    name = "Entropy"

    def score(self, checkpoint: CheckpointData) -> float:
        numpy = import_module("numpy")
        probabilities = numpy.clip(checkpoint.target_probs, 1e-10, 1.0)
        entropy = -numpy.sum(probabilities * numpy.log(probabilities), axis=1)
        return float(-entropy.mean())


class InfoMaxValidator(Validator):
    """Prefer confident yet diverse target predictions."""

    name = "InfoMax"

    def score(self, checkpoint: CheckpointData) -> float:
        numpy = import_module("numpy")
        probabilities = numpy.clip(checkpoint.target_probs, 1e-10, 1.0)
        conditional = -numpy.sum(
            probabilities * numpy.log(probabilities),
            axis=1,
        )
        mean_prob = numpy.clip(probabilities.mean(axis=0), 1e-10, 1.0)
        marginal = -numpy.sum(mean_prob * numpy.log(mean_prob))
        return float(marginal - conditional.mean())


class CorrCValidator(Validator):
    """Correlation consistency between feature similarity and labels."""

    name = "Corr-C"

    def score(self, checkpoint: CheckpointData) -> float:
        numpy = import_module("numpy")
        features = checkpoint.target_features
        probabilities = checkpoint.target_probs
        if len(features) < 2:
            return 0.0

        norms = numpy.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normalized = features / norms
        feature_similarity = normalized @ normalized.T

        predicted_classes = probabilities.argmax(axis=1)
        prediction_similarity = (
            predicted_classes[:, None] == predicted_classes[None, :]
        ).astype(float)

        upper_triangle = numpy.triu_indices(len(features), k=1)
        feature_values = feature_similarity[upper_triangle]
        prediction_values = prediction_similarity[upper_triangle]

        if len(feature_values) < 2:
            return 0.0
        if numpy.std(feature_values) < 1e-10:
            return 0.0
        if numpy.std(prediction_values) < 1e-10:
            return 0.0

        correlation = numpy.corrcoef(feature_values, prediction_values)[0, 1]
        if numpy.isnan(correlation):
            return 0.0
        return float(correlation)


class SNDValidator(Validator):
    """Soft neighborhood density from target features and confidences."""

    name = "SND"

    def score(self, checkpoint: CheckpointData) -> float:
        numpy = import_module("numpy")
        features = checkpoint.target_features
        probabilities = checkpoint.target_probs
        if len(features) < 2:
            return 0.0

        norms = numpy.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normalized = features / norms
        similarity = normalized @ normalized.T

        scaled = similarity / 0.1
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        weights = numpy.exp(scaled)
        numpy.fill_diagonal(weights, 0.0)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)

        confidences = probabilities.max(axis=1)
        return float((weights @ confidences).mean())


def build_default_validators() -> list[Validator]:
    """Return validators in report display order."""
    return [
        OracleValidator(),
        SrcAccValidator(),
        EntropyValidator(),
        InfoMaxValidator(),
        CorrCValidator(),
        SNDValidator(),
    ]
