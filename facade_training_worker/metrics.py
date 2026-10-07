from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def _arrays(targets: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or scores.shape != truth.shape or truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("Targets and probabilities must be non-empty matrices with identical shapes.")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("Multilabel targets must contain only 0 and 1.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Probabilities must be finite values from 0 to 1.")
    return truth, scores


def calibrate_thresholds(targets: Any, probabilities: Any, class_ids: list[str]) -> dict[str, float]:
    truth, scores = _arrays(targets, probabilities)
    if truth.shape[1] != len(class_ids):
        raise ValueError("Class identifiers do not match the prediction matrix.")
    thresholds: dict[str, float] = {}
    for index, class_id in enumerate(class_ids):
        labels = truth[:, index]
        if int(labels.sum()) == 0:
            raise ValueError(f"Development split has no positive example for class {class_id}.")
        candidates = sorted({0.0, 1.0, *(float(value) for value in scores[:, index])}, reverse=True)
        best: tuple[float, int, float] | None = None
        for threshold in candidates:
            predicted = scores[:, index] >= threshold
            tp = int(np.logical_and(predicted, labels == 1).sum())
            fp = int(np.logical_and(predicted, labels == 0).sum())
            fn = int(np.logical_and(~predicted, labels == 1).sum())
            denominator = 2 * tp + fp + fn
            f1 = (2 * tp / denominator) if denominator else 0.0
            candidate = (f1, -fp, threshold)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        thresholds[class_id] = float(best[2])
    return thresholds


def multilabel_metrics(
    targets: Any,
    probabilities: Any,
    class_ids: list[str],
    thresholds: dict[str, float],
) -> dict[str, float]:
    truth, scores = _arrays(targets, probabilities)
    if truth.shape[1] != len(class_ids) or set(thresholds) != set(class_ids):
        raise ValueError("Metric class identifiers do not match predictions and thresholds.")
    threshold_array = np.asarray([float(thresholds[class_id]) for class_id in class_ids])
    predicted = scores >= threshold_array
    tp = np.logical_and(predicted, truth == 1).sum(axis=0).astype(np.float64)
    fp = np.logical_and(predicted, truth == 0).sum(axis=0).astype(np.float64)
    fn = np.logical_and(~predicted, truth == 1).sum(axis=0).astype(np.float64)
    denominators = 2 * tp + fp + fn
    per_class_f1 = np.divide(2 * tp, denominators, out=np.zeros_like(tp), where=denominators > 0)
    micro_denominator = float(2 * tp.sum() + fp.sum() + fn.sum())
    micro_f1 = float(2 * tp.sum() / micro_denominator) if micro_denominator else 0.0
    aps = [float(average_precision_score(truth[:, index], scores[:, index])) for index in range(len(class_ids))]
    values = {
        "development_macro_map": float(np.mean(aps)),
        "development_macro_f1": float(np.mean(per_class_f1)),
        "development_micro_f1": micro_f1,
        "development_exact_match_accuracy": float(np.all(predicted == truth, axis=1).mean()),
        "development_false_positive_count": float(fp.sum()),
        "development_false_negative_count": float(fn.sum()),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Computed development metrics are not finite.")
    return values
