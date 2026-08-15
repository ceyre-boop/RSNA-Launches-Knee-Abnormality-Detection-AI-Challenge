"""Imaging pipeline for the RSNA knee competition.

Torch-free modules (``slots``, ``dicom_geom``, ``volume``, ``labels``,
``ensemble``) are re-exported here. ``dataset``, ``model`` and ``train`` require
the optional ``train`` dependency group and must be imported directly.
"""

from __future__ import annotations

from .dicom_geom import (
    crop_bounds_mm,
    infer_laterality,
    order_indices,
    slice_normal,
    slice_positions,
)
from .ensemble import rank_average
from .slots import SLOT_NAMES, build_slot_table, select_slots

__all__ = [
    "SLOT_NAMES",
    "build_slot_table",
    "crop_bounds_mm",
    "infer_laterality",
    "order_indices",
    "rank_average",
    "select_slots",
    "slice_normal",
    "slice_positions",
]
