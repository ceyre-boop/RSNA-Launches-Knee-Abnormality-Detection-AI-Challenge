import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.imaging.labels import confidence_weight, load_label_table  # noqa: E402
from rsna_knee.imaging.volume import (  # noqa: E402
    apply_laterality,
    crop_and_resize,
    normalize_intensity,
)
from rsna_knee.constants import TARGET_LABELS  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_scales_to_unit_range(self) -> None:
        volume = np.linspace(0.0, 1000.0, 400).reshape(4, 10, 10).astype(np.float32)
        out = normalize_intensity(volume)
        self.assertAlmostEqual(float(out.min()), 0.0, places=5)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)
        self.assertEqual(out.dtype, np.float32)

    def test_outliers_are_clipped(self) -> None:
        volume = np.ones((2, 10, 10), dtype=np.float32)
        volume[0, 0, 0] = 1e6
        out = normalize_intensity(volume)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_constant_volume(self) -> None:
        out = normalize_intensity(np.full((2, 4, 4), 7.0, dtype=np.float32))
        self.assertTrue(np.all(out == 0.0))

    def test_nan_safe(self) -> None:
        volume = np.full((2, 4, 4), np.nan, dtype=np.float32)
        self.assertTrue(np.all(normalize_intensity(volume) == 0.0))


class CropResizeTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        image = np.random.RandomState(0).rand(320, 320).astype(np.float32)
        out = crop_and_resize(image, (0.5, 0.5), crop_mm=130.0, img_size=224)
        self.assertEqual(out.shape, (224, 224))
        self.assertEqual(out.dtype, np.float32)

    def test_crop_uses_physical_spacing(self) -> None:
        image = np.zeros((400, 400), dtype=np.float32)
        image[130:270, 130:270] = 1.0  # a 140px block at the centre
        coarse = crop_and_resize(image, (2.0, 2.0), img_size=64)  # 65px crop -> all ones
        fine = crop_and_resize(image, (0.5, 0.5), img_size=64)  # 260px crop -> ring of zeros
        self.assertGreater(float(coarse.mean()), float(fine.mean()))


class LateralityNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    def test_left_is_untouched(self) -> None:
        np.testing.assert_array_equal(apply_laterality(self.volume, "Sagittal", "L"), self.volume)
        np.testing.assert_array_equal(apply_laterality(self.volume, "Coronal", "L"), self.volume)

    def test_right_sagittal_reverses_slice_order(self) -> None:
        out = apply_laterality(self.volume, "Sagittal", "R")
        np.testing.assert_array_equal(out[0], self.volume[1])
        np.testing.assert_array_equal(out[1], self.volume[0])

    def test_right_coronal_mirrors_horizontally(self) -> None:
        out = apply_laterality(self.volume, "Coronal", "R")
        np.testing.assert_array_equal(out[0], self.volume[0][:, ::-1])
        np.testing.assert_array_equal(out, self.volume[:, :, ::-1])

    def test_right_axial_mirrors_horizontally(self) -> None:
        out = apply_laterality(self.volume, "Axial", "R")
        np.testing.assert_array_equal(out, self.volume[:, :, ::-1])

    def test_never_flips_vertically(self) -> None:
        out = apply_laterality(self.volume, "Coronal", "R")
        # row order (superior/inferior) must be preserved
        np.testing.assert_array_equal(out[:, 0, :], self.volume[:, 0, ::-1])


class LabelTests(unittest.TestCase):
    def test_confidence_weight_bounds(self) -> None:
        weights = confidence_weight(np.array([[0.5] * 12, [1.0] * 12, [0.0] * 12]))
        self.assertAlmostEqual(float(weights[0]), 0.25, places=5)
        self.assertAlmostEqual(float(weights[1]), 1.0, places=5)
        self.assertAlmostEqual(float(weights[2]), 1.0, places=5)

    def test_confidence_weight_is_monotone(self) -> None:
        weights = confidence_weight(np.array([[0.5] * 12, [0.7] * 12, [0.95] * 12]))
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_gold_rows_override(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            pseudo = root / "pseudo.csv"
            gold = root / "gold.csv"
            pd.DataFrame(
                [
                    {"StudyInstanceUID": "a", **{label: 0.6 for label in TARGET_LABELS}},
                    {"StudyInstanceUID": "b", **{label: 0.5 for label in TARGET_LABELS}},
                ]
            ).to_csv(pseudo, index=False)
            pd.DataFrame(
                [{"StudyInstanceUID": "a", **{label: 0.8 for label in TARGET_LABELS}}]
            ).to_csv(gold, index=False)

            table = load_label_table(pseudo, gold).set_index("StudyInstanceUID")
            self.assertEqual(float(table.loc["a", "ACL"]), 1.0)
            self.assertEqual(float(table.loc["a", "sample_weight"]), 3.0)
            self.assertAlmostEqual(float(table.loc["b", "sample_weight"]), 0.25, places=5)
            self.assertEqual(len(table), 2)

    def test_missing_gold_file_is_ignored(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            pseudo = pathlib.Path(temp_dir) / "pseudo.csv"
            pd.DataFrame(
                [{"StudyInstanceUID": "a", **{label: 0.9 for label in TARGET_LABELS}}]
            ).to_csv(pseudo, index=False)
            table = load_label_table(pseudo, pathlib.Path(temp_dir) / "nope.csv")
            self.assertEqual(len(table), 1)


if __name__ == "__main__":
    unittest.main()
