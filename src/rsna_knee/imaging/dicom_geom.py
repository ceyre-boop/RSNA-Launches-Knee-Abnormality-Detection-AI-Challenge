"""Pure geometry helpers for DICOM MR series.

None of these functions import pydicom: they take plain sequences of floats so
they can be unit tested without any DICOM files on disk.

Key competition insight (see intel/competitor-salemali7.md): slice order must be
derived from ImagePositionPatient projected onto the slice normal. Filename order
and InstanceNumber are effectively random in this dataset.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Vector3 = Sequence[float]
Orientation6 = Sequence[float]


def slice_normal(orientation6: Orientation6) -> np.ndarray:
    """Return the unit slice normal for a DICOM ImageOrientationPatient value.

    ``orientation6`` is the 6-element DICOM tag: the first three numbers are the
    direction cosines of the image row, the last three of the image column.
    """
    values = np.asarray(orientation6, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError("orientation6 must have exactly 6 elements")
    row = values[:3]
    col = values[3:]
    normal = np.cross(row, col)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("Degenerate ImageOrientationPatient (zero cross product)")
    return normal / norm


def slice_positions(
    frames: Sequence[Tuple[Vector3, Orientation6]],
    *,
    normal: Optional[Vector3] = None,
) -> List[float]:
    """Project each frame's ImagePositionPatient onto the slice normal.

    ``frames`` is a sequence of ``(position3, orientation6)`` pairs. The normal is
    taken from the first frame unless one is supplied explicitly; MR series in
    this dataset share a single orientation across slices.
    """
    if not frames:
        return []
    if normal is None:
        unit_normal = slice_normal(frames[0][1])
    else:
        vector = np.asarray(normal, dtype=np.float64).reshape(-1)
        if vector.size != 3:
            raise ValueError("normal must have exactly 3 elements")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("normal must be non-zero")
        unit_normal = vector / norm

    projections: List[float] = []
    for position, _orientation in frames:
        point = np.asarray(position, dtype=np.float64).reshape(-1)
        if point.size != 3:
            raise ValueError("position3 must have exactly 3 elements")
        projections.append(float(np.dot(point, unit_normal)))
    return projections


def order_indices(
    frames: Sequence[Tuple[Vector3, Orientation6]],
    *,
    normal: Optional[Vector3] = None,
) -> List[int]:
    """Indices that sort ``frames`` by ascending position along the slice normal."""
    projections = slice_positions(frames, normal=normal)
    if not projections:
        return []
    return [int(index) for index in np.argsort(np.asarray(projections), kind="stable")]


def infer_laterality(tag_value: Optional[str], centroid_x: Optional[float]) -> str:
    """Return ``"L"`` or ``"R"`` for a knee study.

    Prefers the DICOM Laterality / ImageLaterality tag. When the tag is absent or
    unrecognised (roughly half the studies), falls back to the sign of the
    patient-coordinate X of the volume centroid: DICOM patient coordinates are
    LPS, so +X points to the patient's left.
    """
    if tag_value is not None:
        token = str(tag_value).strip().upper()
        if token.startswith("L"):
            return "L"
        if token.startswith("R"):
            return "R"
    if centroid_x is None:
        return "L"
    return "L" if float(centroid_x) >= 0.0 else "R"


def crop_bounds_mm(
    shape: Tuple[int, int],
    pixel_spacing: Sequence[float],
    crop_mm: float = 130.0,
) -> Tuple[int, int, int, int]:
    """Centred physical crop bounds as ``(row_start, row_end, col_start, col_end)``.

    ``shape`` is ``(rows, cols)`` and ``pixel_spacing`` is the DICOM PixelSpacing
    tag ``(row_spacing_mm, col_spacing_mm)``. The crop never exceeds the image;
    bounds are half-open so ``array[r0:r1, c0:c1]`` is the crop.
    """
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("shape must be positive")
    spacing = [float(value) for value in pixel_spacing]
    if len(spacing) != 2:
        raise ValueError("pixel_spacing must have exactly 2 elements")
    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError("pixel_spacing must be positive")
    if crop_mm <= 0.0:
        raise ValueError("crop_mm must be positive")

    crop_rows = min(rows, max(1, int(round(crop_mm / spacing[0]))))
    crop_cols = min(cols, max(1, int(round(crop_mm / spacing[1]))))
    row_start = (rows - crop_rows) // 2
    col_start = (cols - crop_cols) // 2
    return row_start, row_start + crop_rows, col_start, col_start + crop_cols


def centroid_x(positions: Sequence[Vector3]) -> Optional[float]:
    """Mean patient-coordinate X over a set of ImagePositionPatient values."""
    if not positions:
        return None
    points = np.asarray([np.asarray(p, dtype=np.float64).reshape(-1)[0] for p in positions])
    return float(points.mean())
