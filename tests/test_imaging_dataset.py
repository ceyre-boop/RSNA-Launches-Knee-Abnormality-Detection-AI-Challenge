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
    build_cache_index,
    cached_to_float,
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


def ramp_uint8(n_slices: int) -> np.ndarray:
    """Slice ``i`` is uniformly ``i`` so the sampled indices are recoverable."""
    volume = np.zeros((n_slices, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for index in range(n_slices):
        volume[index] = index
    return volume


@pytest.fixture
def cache_dirs(tmp_path):
    """Two shards: s1 (uint8, every slot) in shard a, s2 (float16, one slot) in shard b."""
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    shard_a.mkdir()
    shard_b.mkdir()
    np.savez_compressed(shard_a / "s1.npz", **{slot: ramp_uint8(20) for slot in SLOT_NAMES})
    float_volume = np.full((20, IMG_SIZE, IMG_SIZE), 0.25, dtype=np.float16)
    np.savez_compressed(shard_b / "s2.npz", SAG_FLUID_FS=float_volume)
    return [shard_a, shard_b]


class TestCacheHelpers:
    def test_index_spans_dirs(self, cache_dirs) -> None:
        index = build_cache_index(cache_dirs)
        assert set(index) == {"s1", "s2"}
        assert index["s1"].parent == cache_dirs[0]
        assert index["s2"].parent == cache_dirs[1]

    def test_missing_dir_warns_but_indexes_rest(self, cache_dirs, tmp_path) -> None:
        with pytest.warns(UserWarning):
            index = build_cache_index([tmp_path / "nope", *cache_dirs])
        assert set(index) == {"s1", "s2"}

    def test_uint8_scaled_float_passthrough(self) -> None:
        assert cached_to_float(np.array([[[255]]], dtype=np.uint8))[0, 0, 0] == 1.0
        halves = cached_to_float(np.full((1, 1, 1), 0.5, dtype=np.float16))
        assert halves.dtype == np.float32 and halves[0, 0, 0] == 0.5


class TestCachedDataset:
    def test_uint8_cache_item(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, None, img_size=IMG_SIZE, train=False,
            cache_dirs=cache_dirs,
        )
        item = dataset[0]
        assert item["images"].shape == (len(SLOT_NAMES), GROUP, IMG_SIZE, IMG_SIZE)
        assert float(item["mask"].sum()) == len(SLOT_NAMES)
        assert 0.0 <= float(item["images"].min()) and float(item["images"].max()) <= 1.0
        # uint8 slice value i -> i/255
        recovered = (item["images"][0, :, IMG_SIZE // 2, IMG_SIZE // 2] * 255.0).round().tolist()
        assert recovered == [recovered[0] + offset for offset in range(GROUP)]

    def test_float16_cache_not_rescaled(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s2"], labels, slot_table, None, img_size=IMG_SIZE, train=False,
            cache_dirs=cache_dirs, cache_band_trimmed=False,
        )
        item = dataset[0]
        present = SLOT_NAMES.index("SAG_FLUID_FS")
        assert pytest.approx(float(item["images"][present].mean()), abs=1e-4) == 0.25

    def test_slots_absent_from_npz_are_masked(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s2"], labels, slot_table, None, img_size=IMG_SIZE, train=False,
            cache_dirs=cache_dirs,
        )
        item = dataset[0]
        assert float(item["mask"].sum()) == 1.0
        assert float(item["mask"][SLOT_NAMES.index("SAG_FLUID_FS")]) == 1.0
        assert float(item["images"][SLOT_NAMES.index("COR_T1")].abs().sum()) == 0.0

    def test_missing_study_without_loader_raises(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        with pytest.raises(KeyError) as excinfo:
            StudySlotDataset(
                ["s1", "ghost"], labels, slot_table, None, img_size=IMG_SIZE,
                cache_dirs=cache_dirs,
            )
        assert "1 of 2" in str(excinfo.value)

    def test_missing_study_falls_back_to_loader(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        with pytest.warns(UserWarning):
            dataset = StudySlotDataset(
                ["s1", "s2"], labels, slot_table, fake_loader, img_size=IMG_SIZE,
                cache_dirs=[cache_dirs[0]],
            )
        cached_item, fallback_item = dataset[0], dataset[1]
        assert float(cached_item["mask"].sum()) == len(SLOT_NAMES)
        # s2 is not in shard a, so it comes off the DICOM loader (one populated slot)
        assert float(fallback_item["mask"].sum()) == 1.0

    def test_no_loader_and_no_cache_rejected(self) -> None:
        labels, slot_table = build_tables()
        with pytest.raises(ValueError):
            StudySlotDataset(["s1"], labels, slot_table, None, img_size=IMG_SIZE)

    def test_trimmed_cache_samples_whole_stack(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, None, img_size=IMG_SIZE, train=True,
            cache_dirs=cache_dirs, augment=AugmentConfig(0.0, 0.0, 0.0, 0.0),
        )
        seen = set()
        for trial in range(60):
            torch.manual_seed(trial)  # the item rng folds in the torch seed
            starts = (dataset[0]["images"][:, 0, IMG_SIZE // 2, IMG_SIZE // 2] * 255.0).round()
            for start in starts.tolist():
                assert 0 <= start <= 20 - GROUP
                seen.add(start)
        assert min(seen) < 4  # the band edges are reachable, unlike CENTRAL_BAND

    def test_untrimmed_cache_stays_in_central_band(self, cache_dirs) -> None:
        labels, slot_table = build_tables()
        dataset = StudySlotDataset(
            ["s1"], labels, slot_table, None, img_size=IMG_SIZE, train=True,
            cache_dirs=cache_dirs, cache_band_trimmed=False,
            augment=AugmentConfig(0.0, 0.0, 0.0, 0.0),
        )
        for trial in range(60):
            torch.manual_seed(trial)  # the item rng folds in the torch seed
            starts = (dataset[0]["images"][:, 0, IMG_SIZE // 2, IMG_SIZE // 2] * 255.0).round()
            for start in starts.tolist():
                assert 4 <= start <= 16 - GROUP


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
