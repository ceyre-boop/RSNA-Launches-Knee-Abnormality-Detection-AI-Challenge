import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from rsna_knee.constants import TARGET_LABELS  # noqa: E402
from rsna_knee.imaging.dataset import (  # noqa: E402
    GROUP,
    AugmentConfig,
    StudySlotDataset,
    collate_slots,
    rigid_augment,
    sample_group_indices,
)
from rsna_knee.imaging.slots import SLOT_NAMES  # noqa: E402

IMG_SIZE = 32


def fake_loader(study_id, slot, series_id):
    if series_id == "broken":
        raise OSError("unreadable series")
    volume = np.zeros((12, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for index in range(12):
        volume[index] = index / 11.0
    return volume


def build_tables():
    labels = pd.DataFrame(
        [
            {"StudyInstanceUID": "s1", **{label: 0.9 for label in TARGET_LABELS}, "sample_weight": 0.85},
            {"StudyInstanceUID": "s2", **{label: 0.5 for label in TARGET_LABELS}, "sample_weight": 0.25},
        ]
    )
    slot_table = pd.DataFrame(
        [
            {"StudyInstanceUID": "s1", **{slot: f"ser_{slot}" for slot in SLOT_NAMES}},
            {"StudyInstanceUID": "s2", **{slot: None for slot in SLOT_NAMES}, "SAG_FLUID_FS": "ser_a"},
        ]
    )
    return labels, slot_table


class TestSampleGroupIndices:
    def test_consecutive_and_central(self) -> None:
        indices = sample_group_indices(100)
        assert indices == list(range(indices[0], indices[0] + GROUP))
        assert 20 <= indices[0] and indices[-1] <= 80

    def test_deterministic_without_rng(self) -> None:
        assert sample_group_indices(50) == sample_group_indices(50)

    def test_random_stays_in_band(self) -> None:
        import random

        rng = random.Random(0)
        for _ in range(50):
            indices = sample_group_indices(30, rng=rng)
            assert len(indices) == GROUP
            assert indices[0] >= 6 and indices[-1] <= 24

    def test_short_stack_pads(self) -> None:
        assert sample_group_indices(2) == [0, 1, 1]
        assert sample_group_indices(1) == [0, 0, 0]

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            sample_group_indices(0)


class TestRigidAugment:
    def test_shape_and_range_preserved(self) -> None:
        import random

        images = torch.rand(GROUP, IMG_SIZE, IMG_SIZE)
        out = rigid_augment(images, AugmentConfig(), random.Random(1))
        assert out.shape == images.shape
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_zero_config_is_identity(self) -> None:
        import random

        images = torch.rand(GROUP, IMG_SIZE, IMG_SIZE)
        config = AugmentConfig(0.0, 0.0, 0.0, 0.0)
        out = rigid_augment(images, config, random.Random(1))
        assert torch.allclose(out, images, atol=1e-5)

    def test_no_horizontal_flip(self) -> None:
        """A rigid transform must never mirror the image."""
        import random

        images = torch.zeros(GROUP, IMG_SIZE, IMG_SIZE)
        images[:, :, : IMG_SIZE // 2] = 1.0  # bright left half
        rng = random.Random(7)
        for _ in range(25):
            out = rigid_augment(images, AugmentConfig(), rng)
            left = float(out[:, :, : IMG_SIZE // 4].mean())
            right = float(out[:, :, -IMG_SIZE // 4 :].mean())
            assert left > right


class TestStudySlotDataset:
    def test_item_shapes(self) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        item = dataset[0]
        assert item["images"].shape == (len(SLOT_NAMES), GROUP, IMG_SIZE, IMG_SIZE)
        assert item["mask"].shape == (len(SLOT_NAMES),)
        assert float(item["mask"].sum()) == len(SLOT_NAMES)
        assert item["labels"].shape == (len(TARGET_LABELS),)
        assert pytest.approx(float(item["weight"]), abs=1e-6) == 0.85
        assert item["study_id"] == "s1"

    def test_missing_slots_masked_and_zeroed(self) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s2"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        item = dataset[0]
        assert float(item["mask"].sum()) == 1.0
        assert float(item["mask"][SLOT_NAMES.index("SAG_FLUID_FS")]) == 1.0
        absent = SLOT_NAMES.index("COR_T1")
        assert float(item["images"][absent].abs().sum()) == 0.0

    def test_broken_series_is_skipped(self) -> None:
        labels, slot_table = build_tables()
        slot_table.loc[0, "AX_FLUID_FS"] = "broken"
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        with pytest.warns(UserWarning):
            item = dataset[0]
        assert float(item["mask"][SLOT_NAMES.index("AX_FLUID_FS")]) == 0.0
        assert float(item["mask"].sum()) == len(SLOT_NAMES) - 1

    def test_eval_mode_is_deterministic(self) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        assert torch.equal(dataset[0]["images"], dataset[0]["images"])

    def test_unknown_study_gets_zero_weight(self) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["ghost"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        assert float(dataset[0]["weight"]) == 0.0


class TestCollate:
    def test_batches_stack(self) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1", "s2"], labels, slot_table, fake_loader, img_size=IMG_SIZE, train=False
        )
        batch = collate_slots([dataset[0], dataset[1]])
        assert batch["images"].shape == (2, len(SLOT_NAMES), GROUP, IMG_SIZE, IMG_SIZE)
        assert batch["mask"].shape == (2, len(SLOT_NAMES))
        assert batch["labels"].shape == (2, len(TARGET_LABELS))
        assert batch["study_id"] == ["s1", "s2"]

    def test_pads_narrower_items(self) -> None:
        wide = {
            "images": torch.ones(4, GROUP, IMG_SIZE, IMG_SIZE),
            "mask": torch.ones(4),
            "labels": torch.zeros(len(TARGET_LABELS)),
            "weight": torch.tensor(1.0),
            "study_id": "wide",
        }
        narrow = {
            "images": torch.ones(2, GROUP, IMG_SIZE, IMG_SIZE),
            "mask": torch.ones(2),
            "labels": torch.zeros(len(TARGET_LABELS)),
            "weight": torch.tensor(1.0),
            "study_id": "narrow",
        }
        batch = collate_slots([wide, narrow])
        assert batch["images"].shape[1] == 4
        assert float(batch["mask"][1].sum()) == 2.0
        assert float(batch["images"][1, 2:].abs().sum()) == 0.0
