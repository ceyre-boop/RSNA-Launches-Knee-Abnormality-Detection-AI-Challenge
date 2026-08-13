import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.metrics import binary_roc_auc, macro_auc  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_binary_auc_perfect(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(binary_roc_auc(y_true, y_score), 1.0)

    def test_macro_auc(self) -> None:
        y_true = {"A": [0, 1, 0, 1], "B": [1, 0, 1, 0]}
        y_pred = {"A": [0.1, 0.9, 0.2, 0.8], "B": [0.9, 0.1, 0.8, 0.2]}
        macro, per_label = macro_auc(y_true, y_pred, ["A", "B"])
        self.assertAlmostEqual(per_label["A"], 1.0)
        self.assertAlmostEqual(per_label["B"], 1.0)
        self.assertAlmostEqual(macro, 1.0)


if __name__ == "__main__":
    unittest.main()

