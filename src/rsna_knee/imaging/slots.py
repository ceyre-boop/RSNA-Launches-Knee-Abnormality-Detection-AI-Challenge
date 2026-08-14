"""Fixed plane x sequence "slots" selected from the organizer series metadata.

The organizers publish Fluid_Sensitive / Fat_Suppression / Anatomical_Plane per
series, so slot routing needs no SeriesDescription regex (a known failure mode of
competing solutions -- vendor quirks silently misroute series).

Caveat measured on data/train_series.csv (24,371 series): Fluid_Sensitive and
Fat_Suppression are perfectly collinear there -- every fluid-sensitive series is
fat-suppressed and every non-fluid series is not. SAG_NOFS is therefore always
empty under these columns, and slot coverage is
SAG_FLUID_FS 94.2% / COR_FLUID_FS 96.4% / AX_FLUID_FS 100% / SAG_NOFS 0% /
COR_T1 77.3% / SAG_T1 96.8%. The slot is kept for definitional parity, but
callers that care about encoder throughput can pass a five-slot list to
StudySlotDataset.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from ..constants import STUDY_ID_COLUMN

SERIES_ID_COLUMN = "SeriesInstanceUID"
FLUID_COLUMN = "Fluid_Sensitive"
FAT_SUP_COLUMN = "Fat_Suppression"
PLANE_COLUMN = "Anatomical_Plane"

SLOT_NAMES: List[str] = [
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_NOFS",
    "COR_T1",
    "SAG_T1",
]

# slot -> (plane, fluid_sensitive, fat_suppression or None for "don't care")
SLOT_RULES: Dict[str, tuple] = {
    "SAG_FLUID_FS": ("Sagittal", 1, 1),
    "COR_FLUID_FS": ("Coronal", 1, 1),
    "AX_FLUID_FS": ("Axial", 1, 1),
    "SAG_NOFS": ("Sagittal", 1, 0),
    "COR_T1": ("Coronal", 0, None),
    "SAG_T1": ("Sagittal", 0, None),
}

SliceCounts = Optional[Union[str, Callable[[str], float], Dict[str, float]]]


def _slice_count_lookup(counts: SliceCounts, frame: pd.DataFrame) -> pd.Series:
    """Return a per-row slice count Series aligned to ``frame``'s index."""
    if counts is None:
        return pd.Series(0.0, index=frame.index, dtype=float)
    if isinstance(counts, str):
        if counts not in frame.columns:
            raise KeyError(f"Slice-count column not found: {counts}")
        return pd.to_numeric(frame[counts], errors="coerce").fillna(0.0).astype(float)
    if isinstance(counts, dict):
        return frame[SERIES_ID_COLUMN].map(lambda uid: float(counts.get(uid, 0.0)))
    if callable(counts):
        return frame[SERIES_ID_COLUMN].map(lambda uid: float(counts(uid)))
    raise TypeError("slice_counts must be a column name, dict, callable, or None")


def _matches(frame: pd.DataFrame, plane: str, fluid: int, fat_sup: Optional[int]) -> pd.DataFrame:
    planes = frame[PLANE_COLUMN].astype(str).str.strip().str.lower()
    mask = planes == plane.lower()
    mask &= pd.to_numeric(frame[FLUID_COLUMN], errors="coerce") == fluid
    if fat_sup is not None:
        mask &= pd.to_numeric(frame[FAT_SUP_COLUMN], errors="coerce") == fat_sup
    return frame[mask]


def select_slots(
    series_df: pd.DataFrame,
    study_id: str,
    *,
    slice_counts: SliceCounts = None,
) -> Dict[str, Optional[str]]:
    """Pick at most one SeriesInstanceUID per slot for a single study.

    Ties on slice count are broken by the lexicographically smallest
    SeriesInstanceUID so selection is deterministic across runs and machines.
    Slots with no matching series map to ``None``.
    """
    required = [STUDY_ID_COLUMN, SERIES_ID_COLUMN, FLUID_COLUMN, FAT_SUP_COLUMN, PLANE_COLUMN]
    missing = [column for column in required if column not in series_df.columns]
    if missing:
        raise ValueError(f"series_df missing required columns: {missing}")

    study_frame = series_df[series_df[STUDY_ID_COLUMN] == study_id]
    selected: Dict[str, Optional[str]] = {}
    for slot in SLOT_NAMES:
        plane, fluid, fat_sup = SLOT_RULES[slot]
        candidates = _matches(study_frame, plane, fluid, fat_sup)
        if candidates.empty:
            selected[slot] = None
            continue
        counts = _slice_count_lookup(slice_counts, candidates)
        ranked = candidates.assign(_n_slices=counts).sort_values(
            by=["_n_slices", SERIES_ID_COLUMN],
            ascending=[False, True],
            kind="stable",
        )
        selected[slot] = str(ranked.iloc[0][SERIES_ID_COLUMN])
    return selected


def build_slot_table(
    series_df: pd.DataFrame,
    *,
    slice_counts: SliceCounts = None,
) -> pd.DataFrame:
    """Slot assignment for every study: one row per study, one column per slot."""
    studies = series_df[STUDY_ID_COLUMN].astype(str).drop_duplicates().tolist()
    rows = []
    for study_id in studies:
        assignment = select_slots(series_df, study_id, slice_counts=slice_counts)
        rows.append({STUDY_ID_COLUMN: study_id, **assignment})
    return pd.DataFrame(rows, columns=[STUDY_ID_COLUMN] + SLOT_NAMES)
