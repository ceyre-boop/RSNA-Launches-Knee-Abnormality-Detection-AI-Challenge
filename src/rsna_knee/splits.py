from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence


def _hash_to_fold(value: str, n_folds: int, seed: int) -> int:
    key = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % n_folds


def build_cv_splits(
    train_csv: str | Path,
    output_csv: str | Path,
    *,
    n_folds: int = 5,
    seed: int = 2026,
    study_col: str = "StudyInstanceUID",
    patient_col: str = "PatientID",
    site_col: str = "SiteID",
    language_col: str = "Language",
) -> None:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    train_path = Path(train_csv)
    rows: List[Dict[str, str]] = []
    with train_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {study_col}
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for row in reader:
            rows.append(row)

    fold_rows = []
    for row in rows:
        patient = row.get(patient_col) or row.get(study_col)
        site = row.get(site_col, "unknown")
        language = row.get(language_col, "unknown")
        group_key = f"{patient}|{site}|{language}"
        fold = _hash_to_fold(group_key, n_folds, seed)
        fold_rows.append({study_col: row[study_col], "fold": str(fold)})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[study_col, "fold"])
        writer.writeheader()
        writer.writerows(fold_rows)


def validate_no_group_leakage(
    split_csv: str | Path,
    train_csv: str | Path,
    *,
    study_col: str = "StudyInstanceUID",
    patient_col: str = "PatientID",
) -> bool:
    split_path = Path(split_csv)
    train_path = Path(train_csv)

    split_map: Dict[str, str] = {}
    with split_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split_map[row[study_col]] = row["fold"]

    patient_to_folds: Dict[str, set[str]] = {}
    with train_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            patient = row.get(patient_col) or row[study_col]
            fold = split_map.get(row[study_col])
            if fold is None:
                continue
            patient_to_folds.setdefault(patient, set()).add(fold)

    return all(len(folds) == 1 for folds in patient_to_folds.values())

