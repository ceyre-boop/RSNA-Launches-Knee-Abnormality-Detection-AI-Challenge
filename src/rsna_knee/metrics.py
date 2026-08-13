from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def binary_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have same length")
    if not y_true:
        raise ValueError("Inputs cannot be empty")

    positives = sum(1 for item in y_true if item == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Both positive and negative samples are required")

    ranks = _average_ranks(y_score)
    pos_rank_sum = sum(rank for label, rank in zip(y_true, ranks) if label == 1)
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_auc(
    y_true_by_label: Dict[str, Sequence[int]],
    y_score_by_label: Dict[str, Sequence[float]],
    labels: Iterable[str],
) -> Tuple[float, Dict[str, float]]:
    per_label = {}
    selected_labels = list(labels)
    for label in selected_labels:
        if label not in y_true_by_label or label not in y_score_by_label:
            raise KeyError(f"Missing label: {label}")
        per_label[label] = binary_roc_auc(y_true_by_label[label], y_score_by_label[label])
    macro = sum(per_label.values()) / len(per_label)
    return macro, per_label

