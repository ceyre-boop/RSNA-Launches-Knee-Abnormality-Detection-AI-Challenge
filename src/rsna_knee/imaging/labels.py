"""Study-level label targets and confidence weights for the imaging model.

Pseudo labels are soft probabilities from the report teachers; the 58 gold rows
in train.csv are human-grade and get hard labels plus a large weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS

WEIGHT_COLUMN = "sample_weight"
MIN_WEIGHT = 0.25
MAX_WEIGHT = 1.0
GOLD_WEIGHT = 3.0


def confidence_weight(
    probs: np.ndarray,
    *,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """Map soft labels to per-study weights in ``[min_weight, max_weight]``.

    Confidence is the mean distance of each label from 0.5, rescaled so that an
    all-0.5 study earns ``min_weight`` and a fully saturated study ``max_weight``.
    """
    values = np.asarray(probs, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    confidence = np.nanmean(np.abs(values - 0.5) * 2.0, axis=1)
    confidence = np.nan_to_num(confidence, nan=0.0)
    return (min_weight + (max_weight - min_weight) * np.clip(confidence, 0.0, 1.0)).astype(np.float32)


def load_label_table(
    labels_csv: str | Path,
    gold_csv: Optional[str | Path] = None,
    *,
    targets: Sequence[str] = TARGET_LABELS,
    gold_weight: float = GOLD_WEIGHT,
) -> pd.DataFrame:
    """Return one row per study: study id, ``targets``, and ``sample_weight``.

    Gold rows (all target columns present in ``gold_csv``) override the pseudo
    labels with hard 0/1 values at ``gold_weight``.
    """
    frame = pd.read_csv(labels_csv)
    missing = [column for column in [STUDY_ID_COLUMN, *targets] if column not in frame.columns]
    if missing:
        raise ValueError(f"labels_csv missing columns: {missing}")

    table = frame[[STUDY_ID_COLUMN, *targets]].copy()
    table[STUDY_ID_COLUMN] = table[STUDY_ID_COLUMN].astype(str)
    for target in targets:
        table[target] = pd.to_numeric(table[target], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    table[WEIGHT_COLUMN] = confidence_weight(table[list(targets)].to_numpy())
    table = table.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").set_index(STUDY_ID_COLUMN)

    if gold_csv is not None and Path(gold_csv).exists():
        gold = pd.read_csv(gold_csv)
        if all(column in gold.columns for column in targets):
            gold = gold[[STUDY_ID_COLUMN, *targets]].copy()
            gold[STUDY_ID_COLUMN] = gold[STUDY_ID_COLUMN].astype(str)
            for target in targets:
                gold[target] = pd.to_numeric(gold[target], errors="coerce")
            gold = gold.dropna(subset=list(targets))
            if not gold.empty:
                gold = gold.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").set_index(STUDY_ID_COLUMN)
                for target in targets:
                    gold[target] = (gold[target] >= 0.5).astype(float)
                gold[WEIGHT_COLUMN] = float(gold_weight)
                table = table.drop(index=[i for i in gold.index if i in table.index])
                table = pd.concat([table, gold[[*targets, WEIGHT_COLUMN]]])

    return table.reset_index()
