"""Load a DICOM series directory into a normalized float32 volume.

Pipeline per series:
  1. read every DICOM file (corrupt/unreadable files are skipped with a warning)
  2. order slices geometrically (ImagePositionPatient projected on the normal)
  3. apply RescaleSlope / RescaleIntercept
  4. 130 mm physical centre crop using PixelSpacing
  5. resize to a fixed pixel grid (cv2)
  6. robust per-series intensity normalization (1st-99th percentile -> [0, 1])
  7. laterality normalization to a left-knee convention
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .dicom_geom import centroid_x, crop_bounds_mm, infer_laterality, order_indices

DEFAULT_CROP_MM = 130.0
DEFAULT_IMG_SIZE = 224

_SAGITTAL = "sagittal"


class Slice:
    """One decoded DICOM frame plus the geometry needed to order and crop it."""

    __slots__ = ("pixels", "position", "orientation", "pixel_spacing", "laterality")

    def __init__(
        self,
        pixels: np.ndarray,
        position: Sequence[float],
        orientation: Sequence[float],
        pixel_spacing: Sequence[float],
        laterality: Optional[str],
    ) -> None:
        self.pixels = pixels
        self.position = tuple(float(value) for value in position)
        self.orientation = tuple(float(value) for value in orientation)
        self.pixel_spacing = tuple(float(value) for value in pixel_spacing)
        self.laterality = laterality


def _as_floats(value, count: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
    try:
        floats = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return default
    if len(floats) != count:
        return default
    return floats


def read_series(directory: str | Path) -> List[Slice]:
    """Read every DICOM file in ``directory``; skip anything that fails to decode."""
    import pydicom  # imported lazily so the module stays importable without files

    paths = sorted(path for path in Path(directory).iterdir() if path.is_file())
    slices: List[Slice] = []
    for path in paths:
        try:
            dataset = pydicom.dcmread(str(path), force=True)
            pixels = dataset.pixel_array.astype(np.float32)
        except Exception as error:  # noqa: BLE001 - any decode failure is skippable
            warnings.warn(f"Skipping unreadable DICOM {path}: {error}", stacklevel=2)
            continue
        if pixels.ndim != 2:
            warnings.warn(f"Skipping non-2D DICOM {path} with shape {pixels.shape}", stacklevel=2)
            continue

        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        pixels = pixels * slope + intercept

        position = _as_floats(getattr(dataset, "ImagePositionPatient", None), 3, (0.0, 0.0, 0.0))
        orientation = _as_floats(
            getattr(dataset, "ImageOrientationPatient", None), 6, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        )
        spacing = _as_floats(getattr(dataset, "PixelSpacing", None), 2, (1.0, 1.0))
        laterality = getattr(dataset, "ImageLaterality", None) or getattr(dataset, "Laterality", None)
        slices.append(Slice(pixels, position, orientation, spacing, laterality))
    return slices


def normalize_intensity(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip to the [low, high] percentile of the whole series and scale to [0, 1]."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(np.nan_to_num(volume, nan=lo), lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def crop_and_resize(
    pixels: np.ndarray,
    pixel_spacing: Sequence[float],
    *,
    crop_mm: float = DEFAULT_CROP_MM,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    row_start, row_end, col_start, col_end = crop_bounds_mm(pixels.shape[:2], pixel_spacing, crop_mm)
    cropped = pixels[row_start:row_end, col_start:col_end]
    resized = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float32)


def apply_laterality(volume: np.ndarray, plane: str, side: str) -> np.ndarray:
    """Normalize a right knee to the left-knee convention.

    Sagittal stacks have their slice ORDER reversed (medial/lateral flips along
    the stacking axis); coronal and axial stacks are mirrored horizontally.
    Left knees pass through untouched. Never flip vertically -- that would break
    the superior/inferior anatomy the model keys on.
    """
    if side.upper() != "R":
        return volume
    if str(plane).strip().lower() == _SAGITTAL:
        return volume[::-1].copy()
    return volume[:, :, ::-1].copy()


def load_series_volume(
    directory: str | Path,
    *,
    plane: str,
    img_size: int = DEFAULT_IMG_SIZE,
    crop_mm: float = DEFAULT_CROP_MM,
    side: Optional[str] = None,
) -> np.ndarray:
    """Return a ``[n_slices, img_size, img_size]`` float32 volume in [0, 1].

    ``side`` overrides laterality inference when the caller already knows it.
    Raises ``ValueError`` when no slice in the directory could be decoded.
    """
    slices = read_series(directory)
    if not slices:
        raise ValueError(f"No readable DICOM slices in {directory}")

    frames = [(item.position, item.orientation) for item in slices]
    ordered = [slices[index] for index in order_indices(frames)]

    planes = [
        crop_and_resize(item.pixels, item.pixel_spacing, crop_mm=crop_mm, img_size=img_size)
        for item in ordered
    ]
    volume = normalize_intensity(np.stack(planes, axis=0))

    if side is None:
        tag = next((item.laterality for item in ordered if item.laterality), None)
        side = infer_laterality(tag, centroid_x([item.position for item in ordered]))
    return apply_laterality(volume, plane, side)
