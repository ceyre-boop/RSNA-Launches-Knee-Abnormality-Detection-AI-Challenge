from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from .constants import STUDY_ID_COLUMN, TARGET_LABELS


def _read_table(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_submission(ids_csv: str | Path, preds_csv: str | Path, output_csv: str | Path) -> None:
    ids_rows = _read_table(ids_csv)
    pred_rows = _read_table(preds_csv)
    pred_map = {row[STUDY_ID_COLUMN]: row for row in pred_rows}

    fieldnames = [STUDY_ID_COLUMN] + TARGET_LABELS
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ids_rows:
            study_id = row[STUDY_ID_COLUMN]
            if study_id not in pred_map:
                raise ValueError(f"Missing predictions for {study_id}")
            pred_row = pred_map[study_id]
            output = {STUDY_ID_COLUMN: study_id}
            for label in TARGET_LABELS:
                if label not in pred_row:
                    raise ValueError(f"Missing prediction column: {label}")
                output[label] = pred_row[label]
            writer.writerow(output)

