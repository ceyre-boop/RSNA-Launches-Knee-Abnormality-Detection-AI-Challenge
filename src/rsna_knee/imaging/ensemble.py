"""Rank-average ensembling.

Macro AUC reads only the order of predictions within a target, so blending on
percentile ranks (rather than raw probabilities) removes any calibration
mismatch between ensemble members.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS


def percentile_rank(values: pd.Series) -> pd.Series:
    """Average-tie percentile rank in (0, 1]."""
    return values.rank(method="average", pct=True)


def rank_average(
    preds: Mapping[str, pd.DataFrame],
    weights: Optional[Mapping[str, float]] = None,
    *,
    targets: Sequence[str] = TARGET_LABELS,
    study_col: str = STUDY_ID_COLUMN,
) -> pd.DataFrame:
    """Weighted percentile-rank blend of several prediction frames.

    Every frame must contain ``study_col`` plus all ``targets`` and cover the
    same set of studies. Output rows follow the first frame's study order and
    values are in (0, 1].
    """
    if not preds:
        raise ValueError("preds must not be empty")

    names = list(preds.keys())
    resolved: Dict[str, float] = {name: 1.0 for name in names} if weights is None else dict(weights)
    missing_weights = [name for name in names if name not in resolved]
    if missing_weights:
        raise KeyError(f"Missing weights for: {missing_weights}")
    total_weight = float(sum(resolved[name] for name in names))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be positive")

    reference = preds[names[0]]
    if study_col not in reference.columns:
        raise ValueError(f"Frame '{names[0]}' missing column {study_col}")
    study_ids = reference[study_col].astype(str).tolist()

    aligned: Dict[str, pd.DataFrame] = {}
    for name in names:
        frame = preds[name]
        if study_col not in frame.columns:
            raise ValueError(f"Frame '{name}' missing column {study_col}")
        missing_targets = [target for target in targets if target not in frame.columns]
        if missing_targets:
            raise ValueError(f"Frame '{name}' missing targets: {missing_targets}")
        indexed = frame.copy()
        indexed[study_col] = indexed[study_col].astype(str)
        indexed = indexed.set_index(study_col)
        if set(indexed.index) != set(study_ids):
            raise ValueError(f"Frame '{name}' does not cover the same studies as '{names[0]}'")
        aligned[name] = indexed.loc[study_ids]

    output = pd.DataFrame({study_col: study_ids})
    for target in targets:
        blended = None
        for name in names:
            ranks = percentile_rank(aligned[name][target].astype(float))
            contribution = ranks.to_numpy() * (resolved[name] / total_weight)
            blended = contribution if blended is None else blended + contribution
        output[target] = blended
    return output
