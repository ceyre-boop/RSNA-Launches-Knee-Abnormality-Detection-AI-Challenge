import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.imaging.dicom_geom import (  # noqa: E402
    centroid_x,
    crop_bounds_mm,
    infer_laterality,
    order_indices,
    slice_normal,
    slice_positions,
)

SAGITTAL = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
AXIAL = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
CORONAL = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class SliceNormalTests(unittest.TestCase):
    def test_axial_normal_is_z(self) -> None:
        np.testing.assert_allclose(slice_normal(AXIAL), [0.0, 0.0, 1.0], atol=1e-9)

    def test_sagittal_normal_is_x(self) -> None:
        np.testing.assert_allclose(np.abs(slice_normal(SAGITTAL)), [1.0, 0.0, 0.0], atol=1e-9)

    def test_coronal_normal_is_y(self) -> None:
        np.testing.assert_allclose(np.abs(slice_normal(CORONAL)), [0.0, 1.0, 0.0], atol=1e-9)

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            slice_normal([1.0, 0.0, 0.0])

    def test_rejects_degenerate(self) -> None:
        with self.assertRaises(ValueError):
            slice_normal([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


class OrderingTests(unittest.TestCase):
    def _axial_stack(self, z_values):
        return [([0.0, 0.0, z], AXIAL) for z in z_values]

    def test_shuffled_stack_reorders_correctly(self) -> None:
        true_z = [float(index) * 3.0 for index in range(10)]
        shuffled_z = [true_z[i] for i in [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]]
        frames = self._axial_stack(shuffled_z)
        indices = order_indices(frames)
        ordered_z = [shuffled_z[index] for index in indices]
        self.assertEqual(ordered_z, sorted(true_z))

    def test_reversing_positions_reverses_indices(self) -> None:
        z_values = [0.0, 4.0, 8.0, 12.0, 16.0]
        forward = order_indices(self._axial_stack(z_values))
        backward = order_indices(self._axial_stack(list(reversed(z_values))))
        self.assertEqual(forward, [0, 1, 2, 3, 4])
        self.assertEqual(backward, list(reversed(forward)))

    def test_sagittal_projection_uses_x(self) -> None:
        frames = [([5.0, 0.0, 0.0], SAGITTAL), ([-5.0, 0.0, 0.0], SAGITTAL)]
        projections = slice_positions(frames)
        self.assertAlmostEqual(abs(projections[0]), 5.0)
        self.assertAlmostEqual(abs(projections[1]), 5.0)
        self.assertNotEqual(np.sign(projections[0]), np.sign(projections[1]))

    def test_empty_stack(self) -> None:
        self.assertEqual(slice_positions([]), [])
        self.assertEqual(order_indices([]), [])

    def test_explicit_normal_overrides(self) -> None:
        frames = [([0.0, 0.0, 2.0], AXIAL), ([0.0, 0.0, 1.0], AXIAL)]
        self.assertEqual(order_indices(frames, normal=[0.0, 0.0, 1.0]), [1, 0])
        self.assertEqual(order_indices(frames, normal=[0.0, 0.0, -1.0]), [0, 1])


class LateralityTests(unittest.TestCase):
    def test_tag_wins(self) -> None:
        self.assertEqual(infer_laterality("L", -100.0), "L")
        self.assertEqual(infer_laterality("R", 100.0), "R")
        self.assertEqual(infer_laterality("left", None), "L")
        self.assertEqual(infer_laterality(" right ", None), "R")

    def test_geometric_fallback(self) -> None:
        self.assertEqual(infer_laterality(None, 62.5), "L")
        self.assertEqual(infer_laterality(None, -62.5), "R")
        self.assertEqual(infer_laterality("", -62.5), "R")

    def test_unknown_tag_falls_back(self) -> None:
        self.assertEqual(infer_laterality("UNKNOWN", -10.0), "R")

    def test_defaults_to_left_without_evidence(self) -> None:
        self.assertEqual(infer_laterality(None, None), "L")

    def test_centroid_x(self) -> None:
        self.assertAlmostEqual(centroid_x([[10.0, 0, 0], [20.0, 0, 0]]), 15.0)
        self.assertIsNone(centroid_x([]))


class CropBoundsTests(unittest.TestCase):
    def test_exact_half_millimetre_spacing(self) -> None:
        bounds = crop_bounds_mm((512, 512), (0.5, 0.5), 130.0)
        self.assertEqual(bounds, (126, 386, 126, 386))
        self.assertEqual(bounds[1] - bounds[0], 260)

    def test_crop_clamped_to_image(self) -> None:
        # 130mm / 4mm = 33px requested, but a 20px image can only give 20
        self.assertEqual(crop_bounds_mm((20, 20), (4.0, 4.0), 130.0), (0, 20, 0, 20))
        # 130mm / 2mm = 65px fits inside a 100px image
        self.assertEqual(crop_bounds_mm((100, 100), (2.0, 2.0), 130.0), (17, 82, 17, 82))

    def test_anisotropic_spacing(self) -> None:
        row_start, row_end, col_start, col_end = crop_bounds_mm((400, 400), (0.5, 1.0), 130.0)
        self.assertEqual(row_end - row_start, 260)
        self.assertEqual(col_end - col_start, 130)

    def test_crop_is_centred(self) -> None:
        rows, cols = 320, 320
        row_start, row_end, col_start, col_end = crop_bounds_mm((rows, cols), (0.65, 0.65), 130.0)
        self.assertEqual(row_start, rows - row_end)
        self.assertEqual(col_start, cols - col_end)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            crop_bounds_mm((0, 10), (1.0, 1.0))
        with self.assertRaises(ValueError):
            crop_bounds_mm((10, 10), (0.0, 1.0))
        with self.assertRaises(ValueError):
            crop_bounds_mm((10, 10), (1.0, 1.0), crop_mm=0.0)
        with self.assertRaises(ValueError):
            crop_bounds_mm((10, 10), (1.0,))


if __name__ == "__main__":
    unittest.main()
