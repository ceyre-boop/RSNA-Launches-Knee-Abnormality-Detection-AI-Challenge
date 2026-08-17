"""Torch Dataset producing one 2.5D slice group per slot per study.

Requires the optional ``train`` dependency group (``pip install -e '.[train]'``).
"""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS
from .labels import WEIGHT_COLUMN
from .slots import SLOT_NAMES
from .volume import DEFAULT_CROP_MM, DEFAULT_IMG_SIZE, load_series_volume

GROUP = 3
CENTRAL_BAND = (0.2, 0.8)
# Preprocessed caches are already band-trimmed, so sample across the whole stack.
FULL_BAND = (0.0, 1.0)
CACHE_SUFFIX = ".npz"

SLOT_PLANES: Dict[str, str] = {
    "SAG_FLUID_FS": "Sagittal",
    "COR_FLUID_FS": "Coronal",
    "AX_FLUID_FS": "Axial",
    "SAG_NOFS": "Sagittal",
    "COR_T1": "Coronal",
    "SAG_T1": "Sagittal",
}

VolumeLoader = Callable[[str, str, str], np.ndarray]


class AugmentConfig:
    """Rigid-only augmentation. No flips: they would break laterality/anatomy."""

    def __init__(
        self,
        max_rotation_deg: float = 8.0,
        max_zoom_in: float = 0.08,
        max_translate: float = 0.05,
        max_intensity: float = 0.10,
    ) -> None:
        self.max_rotation_deg = max_rotation_deg
        self.max_zoom_in = max_zoom_in
        self.max_translate = max_translate
        self.max_intensity = max_intensity


class SeriesVolumeLoader:
    """Default loader: ``data_root/<study_uid>/<series_uid>/*.dcm``, optional .npy cache."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        img_size: int = DEFAULT_IMG_SIZE,
        crop_mm: float = DEFAULT_CROP_MM,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.crop_mm = crop_mm
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def __call__(self, study_id: str, slot: str, series_id: str) -> np.ndarray:
        if self.cache_dir is not None:
            cached = self.cache_dir / study_id / f"{series_id}_{self.img_size}.npy"
            if cached.exists():
                return np.load(cached).astype(np.float32)
        volume = load_series_volume(
            self.data_root / study_id / series_id,
            plane=SLOT_PLANES.get(slot, "Sagittal"),
            img_size=self.img_size,
            crop_mm=self.crop_mm,
        )
        if self.cache_dir is not None:
            cached = self.cache_dir / study_id / f"{series_id}_{self.img_size}.npy"
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.save(cached, volume)
        return volume


def sample_group_indices(
    n_slices: int,
    *,
    group: int = GROUP,
    band: Sequence[float] = CENTRAL_BAND,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Pick ``group`` consecutive slice indices from the central band of a stack.

    Deterministic (band centre) when ``rng`` is None. Short stacks are padded by
    repeating their edge slices.
    """
    if n_slices <= 0:
        raise ValueError("n_slices must be positive")
    if n_slices < group:
        return [min(index, n_slices - 1) for index in range(group)]

    low = max(0, min(int(math.floor(band[0] * n_slices)), n_slices - group))
    high = min(int(math.ceil(band[1] * n_slices)), n_slices)
    last_start = max(low, min(high - group, n_slices - group))
    start = rng.randint(low, last_start) if rng is not None else (low + last_start) // 2
    return list(range(start, start + group))


def build_cache_index(cache_dirs: Sequence[str | Path]) -> Dict[str, Path]:
    """Map ``StudyInstanceUID -> npz path`` across one or more (sharded) cache dirs.

    A study lives in exactly one shard; if it somehow appears twice the first
    directory listed wins.
    """
    index: Dict[str, Path] = {}
    for cache_dir in cache_dirs:
        directory = Path(cache_dir)
        if not directory.is_dir():
            warnings.warn(f"Cache dir missing: {directory}", stacklevel=2)
            continue
        for path in sorted(directory.glob(f"*{CACHE_SUFFIX}")):
            index.setdefault(path.stem, path)
    return index


def cached_to_float(volume: np.ndarray) -> np.ndarray:
    """uint8 caches store 0-255; float caches already store 0-1."""
    if volume.dtype == np.uint8:
        return volume.astype(np.float32) / 255.0
    return volume.astype(np.float32)


def rigid_augment(
    images: torch.Tensor,
    config: AugmentConfig,
    rng: random.Random,
) -> torch.Tensor:
    """Apply one shared rigid transform + intensity scale to a ``[C, H, W]`` group."""
    angle = math.radians(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg))
    zoom = 1.0 + rng.uniform(0.0, config.max_zoom_in)  # zoom-in only
    tx = rng.uniform(-config.max_translate, config.max_translate)
    ty = rng.uniform(-config.max_translate, config.max_translate)

    # affine_grid maps output coords to input coords, so a zoom-in is a 1/zoom scale
    scale = 1.0 / zoom
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    theta = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=images.dtype
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, [1, *images.shape], align_corners=False)
    warped = torch.nn.functional.grid_sample(
        images.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)

    intensity = 1.0 + rng.uniform(-config.max_intensity, config.max_intensity)
    return (warped * intensity).clamp(0.0, 1.0)


class StudySlotDataset(Dataset):
    """One item per study: ``[n_slots, GROUP, H, W]`` plus mask, labels and weight."""

    def __init__(
        self,
        study_ids: Sequence[str],
        label_table: pd.DataFrame,
        slot_table: pd.DataFrame,
        volume_loader: Optional[VolumeLoader] = None,
        *,
        img_size: int = DEFAULT_IMG_SIZE,
        slots: Sequence[str] = SLOT_NAMES,
        targets: Sequence[str] = TARGET_LABELS,
        group: int = GROUP,
        train: bool = False,
        augment: Optional[AugmentConfig] = None,
        seed: int = 2026,
        cache_dirs: Optional[Sequence[str | Path]] = None,
        cache_band_trimmed: bool = True,
    ) -> None:
        self.study_ids = [str(study_id) for study_id in study_ids]
        self.slots = list(slots)
        self.targets = list(targets)
        self.group = group
        self.img_size = img_size
        self.train = train
        self.augment = augment or AugmentConfig()
        self.seed = seed
        self.volume_loader = volume_loader

        labels = label_table.copy()
        labels[STUDY_ID_COLUMN] = labels[STUDY_ID_COLUMN].astype(str)
        self._labels = labels.set_index(STUDY_ID_COLUMN)

        assignments = slot_table.copy()
        assignments[STUDY_ID_COLUMN] = assignments[STUDY_ID_COLUMN].astype(str)
        self._slots = assignments.set_index(STUDY_ID_COLUMN)

        self.cache_band_trimmed = cache_band_trimmed
        self.cache_dirs = [Path(d) for d in cache_dirs] if cache_dirs else []
        self._cache_index: Dict[str, Path] = (
            build_cache_index(self.cache_dirs) if self.cache_dirs else {}
        )
        if self.cache_dirs:
            missing = [uid for uid in self.study_ids if uid not in self._cache_index]
            if missing and self.volume_loader is None:
                raise KeyError(
                    f"{len(missing)} of {len(self.study_ids)} studies missing from cache "
                    f"{[str(d) for d in self.cache_dirs]} and no DICOM fallback loader given; "
                    f"first missing: {missing[:3]}"
                )
            if missing:
                warnings.warn(
                    f"{len(missing)} of {len(self.study_ids)} studies missing from cache; "
                    "falling back to the DICOM loader for those",
                    stacklevel=2,
                )
        elif self.volume_loader is None:
            raise ValueError("StudySlotDataset needs a volume_loader, cache_dirs, or both")

    def __len__(self) -> int:
        return len(self.study_ids)

    def _rng(self, index: int) -> Optional[random.Random]:
        if not self.train:
            return None
        # Fold in the torch seed so each epoch/worker draws a different transform.
        return random.Random(f"{self.seed}|{index}|{torch.initial_seed()}")

    def _group_from_volume(
        self,
        volume: np.ndarray,
        rng: Optional[random.Random],
        band: Sequence[float],
    ) -> Optional[torch.Tensor]:
        if volume is None or volume.ndim != 3 or volume.shape[0] == 0:
            return None
        indices = sample_group_indices(volume.shape[0], group=self.group, band=band, rng=rng)
        stacked = torch.from_numpy(np.ascontiguousarray(volume[indices])).float()
        if rng is not None:
            stacked = rigid_augment(stacked, self.augment, rng)
        return stacked

    def _series_id(self, study_id: str, slot: str) -> Optional[str]:
        if study_id not in self._slots.index or slot not in self._slots.columns:
            return None
        series_id = self._slots.loc[study_id, slot]
        if series_id is None or pd.isna(series_id):
            return None
        return str(series_id)

    def _cached_groups(
        self, study_id: str, rng: Optional[random.Random]
    ) -> Dict[str, torch.Tensor]:
        """Slot -> group tensor for one study, read from its preprocessed npz."""
        path = self._cache_index[study_id]
        band = FULL_BAND if self.cache_band_trimmed else CENTRAL_BAND
        groups: Dict[str, torch.Tensor] = {}
        try:
            with np.load(path) as archive:
                keys = set(archive.files)
                for slot in self.slots:
                    key = slot
                    if key not in keys:
                        # Older caches key arrays by series UID rather than slot name.
                        series_id = self._series_id(study_id, slot)
                        if series_id is None or series_id not in keys:
                            continue
                        key = series_id
                    group = self._group_from_volume(cached_to_float(archive[key]), rng, band)
                    if group is not None:
                        groups[slot] = group
        except Exception as error:  # noqa: BLE001 - a bad npz must not kill the epoch
            warnings.warn(f"Cache unusable for {study_id}: {error}", stacklevel=2)
            return {}
        return groups

    def _slot_group(self, study_id: str, slot: str, rng: Optional[random.Random]) -> Optional[torch.Tensor]:
        series_id = self._series_id(study_id, slot)
        # build_slot_table stores absent slots as None, which pandas may surface as NaN
        if series_id is None or self.volume_loader is None:
            return None
        try:
            volume = self.volume_loader(study_id, slot, series_id)
        except Exception as error:  # noqa: BLE001 - a bad series must not kill the epoch
            warnings.warn(f"Slot {slot} unusable for {study_id}: {error}", stacklevel=2)
            return None
        return self._group_from_volume(volume, rng, CENTRAL_BAND)

    def __getitem__(self, index: int) -> Dict[str, object]:
        study_id = self.study_ids[index]
        rng = self._rng(index)

        cached = self._cached_groups(study_id, rng) if study_id in self._cache_index else None

        images = torch.zeros(len(self.slots), self.group, self.img_size, self.img_size)
        mask = torch.zeros(len(self.slots))
        for position, slot in enumerate(self.slots):
            group = cached.get(slot) if cached is not None else self._slot_group(study_id, slot, rng)
            if group is None:
                continue
            images[position] = group
            mask[position] = 1.0

        if study_id in self._labels.index:
            row = self._labels.loc[study_id]
            labels = torch.tensor([float(row[target]) for target in self.targets])
            weight = float(row[WEIGHT_COLUMN]) if WEIGHT_COLUMN in row else 1.0
        else:
            labels = torch.full((len(self.targets),), 0.5)
            weight = 0.0

        return {
            "images": images,
            "mask": mask,
            "labels": labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
            "study_id": study_id,
        }


def collate_slots(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Stack a batch, zero-padding studies that carry fewer slots than the widest."""
    n_slots = max(int(item["images"].shape[0]) for item in batch)
    group = int(batch[0]["images"].shape[1])
    height = int(batch[0]["images"].shape[2])
    width = int(batch[0]["images"].shape[3])

    images = torch.zeros(len(batch), n_slots, group, height, width)
    mask = torch.zeros(len(batch), n_slots)
    for index, item in enumerate(batch):
        present = int(item["images"].shape[0])
        images[index, :present] = item["images"]
        mask[index, :present] = item["mask"]

    return {
        "images": images,
        "mask": mask,
        "labels": torch.stack([item["labels"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
        "study_id": [item["study_id"] for item in batch],
    }
