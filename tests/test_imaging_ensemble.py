import pathlib
import sys
import unittest

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.imaging.ensemble import percentile_rank, rank_average  # noqa: E402

TARGETS = ["ACL", "MCL"]


def preds(study_ids, acl, mcl):
    return pd.DataFrame({"StudyInstanceUID": study_ids, "ACL": acl, "MCL": mcl})


class PercentileRankTests(unittest.TestCase):
    def test_ranks_are_evenly_spaced(self) -> None:
        ranks = percentile_rank(pd.Series([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual([round(value, 4) for value in ranks], [0.25, 0.5, 0.75, 1.0])

    def test_ties_average(self) -> None:
        ranks = percentile_rank(pd.Series([1.0, 1.0, 2.0, 3.0]))
        self.assertAlmostEqual(ranks[0], 0.375)
        self.assertAlmostEqual(ranks[1], 0.375)


class RankAverageTests(unittest.TestCase):
    def test_scale_invariance(self) -> None:
        """Members disagree wildly in scale but agree in order -> blend keeps order."""
        a = preds(["x", "y", "z"], [0.01, 0.02, 0.03], [0.9, 0.5, 0.1])
        b = preds(["x", "y", "z"], [100.0, 200.0, 300.0], [9.0, 5.0, 1.0])
        result = rank_average({"a": a, "b": b}, targets=TARGETS)
        self.assertEqual(list(result["StudyInstanceUID"]), ["x", "y", "z"])
        self.assertEqual(
            [round(v, 6) for v in result["ACL"]],
            [round(1 / 3, 6), round(2 / 3, 6), 1.0],
        )
        self.assertEqual(
            [round(v, 6) for v in result["MCL"]],
            [1.0, round(2 / 3, 6), round(1 / 3, 6)],
        )

    def test_weighted_blend_of_opposite_orders(self) -> None:
        a = preds(["x", "y"], [0.1, 0.9], [0.0, 0.0])
        b = preds(["x", "y"], [0.9, 0.1], [0.0, 0.0])
        result = rank_average({"a": a, "b": b}, {"a": 3.0, "b": 1.0}, targets=TARGETS)
        # x: (0.5*3 + 1.0*1)/4 = 0.625 ; y: (1.0*3 + 0.5*1)/4 = 0.875
        self.assertAlmostEqual(result["ACL"][0], 0.625)
        self.assertAlmostEqual(result["ACL"][1], 0.875)

    def test_single_member_returns_its_own_ranks(self) -> None:
        a = preds(["x", "y", "z", "w"], [0.4, 0.1, 0.3, 0.2], [1, 2, 3, 4])
        result = rank_average({"a": a}, targets=TARGETS)
        self.assertEqual([round(v, 4) for v in result["ACL"]], [1.0, 0.25, 0.75, 0.5])

    def test_row_order_follows_first_frame(self) -> None:
        a = preds(["x", "y"], [0.1, 0.2], [0.1, 0.2])
        b = preds(["y", "x"], [0.9, 0.8], [0.9, 0.8])
        result = rank_average({"a": a, "b": b}, targets=TARGETS)
        self.assertEqual(list(result["StudyInstanceUID"]), ["x", "y"])
        # b ranks x below y, a ranks x below y -> x stays lowest
        self.assertLess(result["ACL"][0], result["ACL"][1])

    def test_uniform_weights_default(self) -> None:
        a = preds(["x", "y"], [0.1, 0.2], [0.0, 0.0])
        result = rank_average({"a": a, "b": a.copy()}, targets=TARGETS)
        self.assertAlmostEqual(result["ACL"][0], 0.5)
        self.assertAlmostEqual(result["ACL"][1], 1.0)

    def test_errors(self) -> None:
        a = preds(["x", "y"], [0.1, 0.2], [0.1, 0.2])
        with self.assertRaises(ValueError):
            rank_average({}, targets=TARGETS)
        with self.assertRaises(KeyError):
            rank_average({"a": a, "b": a}, {"a": 1.0}, targets=TARGETS)
        with self.assertRaises(ValueError):
            rank_average({"a": a}, {"a": 0.0}, targets=TARGETS)
        with self.assertRaises(ValueError):
            rank_average({"a": a.drop(columns=["MCL"])}, targets=TARGETS)
        with self.assertRaises(ValueError):
            rank_average({"a": a, "b": preds(["x", "q"], [1, 2], [1, 2])}, targets=TARGETS)
        with self.assertRaises(ValueError):
            rank_average({"a": a.rename(columns={"StudyInstanceUID": "id"})}, targets=TARGETS)


if __name__ == "__main__":
    unittest.main()
