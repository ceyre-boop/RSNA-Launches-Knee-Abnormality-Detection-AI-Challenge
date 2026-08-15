from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS
from .types import ExtractionResult


def read_train(train_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    missing = [c for c in [STUDY_ID_COLUMN, "Report", *TARGET_LABELS] if c not in df.columns]
    if missing:
        raise ValueError(f"train csv missing columns: {missing}")
    return df


def gold_mask(df: pd.DataFrame) -> pd.Series:
    return df[list(TARGET_LABELS)].notna().all(axis=1)


def write_pseudo_labels(
    results: Iterable[ExtractionResult], output_csv: str | Path
) -> pd.DataFrame:
    rows: List[dict] = []
    for r in results:
        row = {STUDY_ID_COLUMN: r.study_uid, "engine": r.engine, "language": r.language}
        for label in TARGET_LABELS:
            row[label] = r.labels[label].score
        rows.append(row)
    df = pd.DataFrame(rows, columns=[STUDY_ID_COLUMN, *TARGET_LABELS, "engine", "language"])
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
