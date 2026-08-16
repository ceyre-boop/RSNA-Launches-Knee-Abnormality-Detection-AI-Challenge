import pathlib
import sys
import unittest

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import STUDY_ID_COLUMN, TARGET_LABELS  # noqa: E402
from rsna_knee.selftrain import corrections  # noqa: E402

N = 200


def synthetic_labels(prevalence: float = 0.3) -> pd.DataFrame:
    data = {STUDY_ID_COLUMN: [f"s-{index:04d}" for index in range(N)]}
    for label in TARGET_LABELS:
        data[label] = [1.0 if index < int(N * prevalence) else 0.0 for index in range(N)]
    return pd.DataFrame(data)


def synthetic_probs(labels_frame: pd.DataFrame, *, flip_strength: float = 0.95) -> pd.DataFrame:
    data = {STUDY_ID_COLUMN: labels_frame[STUDY_ID_COLUMN]}
    for label in TARGET_LABELS:
        current = labels_frame[label].to_numpy()
        data[label + "_pred"] = np.where(current >= 0.5, 1.0 - flip_strength, flip_strength)
    return pd.DataFrame(data)


class ThresholdTests(unittest.TestCase):
    def test_low_count_labels_use_the_pooled_backoff(self) -> None:
        labels_frame = synthetic_labels()
        probs = synthetic_probs(labels_frame, flip_strength=0.2)
        weights = {label: 0.9 for label in TARGET_LABELS}
        counts = {label: 20 for label in TARGET_LABELS}
        counts["Fracture"] = 4
        counts["MCL"] = 9
        thresholds = corrections.derive_thresholds(
            labels_frame, probs, label_weights=weights, gold_positive_counts=counts
        )
        self.assertEqual(thresholds["ACL"][1], "empirical")
        self.assertEqual(thresholds["Fracture"][1], "pooled_backoff")
        self.assertEqual(thresholds["MCL"][1], "pooled_backoff")

    def test_low_count_threshold_boundary_is_ten(self) -> None:
        self.assertEqual(corrections.LOW_COUNT_POSITIVE_THRESHOLD, 10)
        labels_frame = synthetic_labels()
        probs = synthetic_probs(labels_frame)
        thresholds = corrections.derive_thresholds(
            labels_frame,
            probs,
            label_weights={label: 0.8 for label in TARGET_LABELS},
            gold_positive_counts={label: 10 for label in TARGET_LABELS},
        )
        self.assertTrue(all(source == "empirical" for _, source in thresholds.values()))


class BandTests(unittest.TestCase):
    def test_weak_labels_get_the_tightest_band(self) -> None:
        weak_low, weak_high = corrections.prevalence_band(0.30, 0.65)
        strong_low, strong_high = corrections.prevalence_band(0.30, 1.00)
        self.assertLess(weak_high - weak_low, strong_high - strong_low)
        self.assertAlmostEqual(weak_high - weak_low, 0.30 * 2 * corrections.MIN_BAND_RELATIVE)
        self.assertAlmostEqual(strong_high - strong_low, 0.30 * 2 * corrections.MAX_BAND_RELATIVE)

    def test_band_is_clipped_below_the_auc_floor(self) -> None:
        below = corrections.prevalence_band(0.30, 0.40)
        at_floor = corrections.prevalence_band(0.30, corrections.AUC_FLOOR)
        self.assertEqual(below, at_floor)


class CapTests(unittest.TestCase):
    def test_caps_hold_prevalence_inside_the_band(self) -> None:
        current = np.array([1.0] * 60 + [0.0] * 140)
        candidates = np.ones(200, dtype=bool)
        probs = np.linspace(0.0, 1.0, 200)
        prevalence = 60 / 200
        band = corrections.prevalence_band(prevalence, 1.0)
        accepted = corrections.cap_corrections(current, candidates, probs, band=band)
        updated = current.copy()
        updated[accepted] = np.where(current[accepted] >= 0.5, 0.0, 1.0)
        after = float((updated >= 0.5).mean())
        self.assertGreaterEqual(after, band[0] - 1e-9)
        self.assertLessEqual(after, band[1] + 1e-9)
        self.assertLess(int(accepted.sum()), int(candidates.sum()))

    def test_max_corrections_zero_is_the_identity_round(self) -> None:
        current = np.array([1.0] * 60 + [0.0] * 140)
        candidates = np.ones(200, dtype=bool)
        probs = np.linspace(0.0, 1.0, 200)
        accepted = corrections.cap_corrections(
            current, candidates, probs, band=(0.0, 1.0), max_corrections=0
        )
        self.assertEqual(int(accepted.sum()), 0)

    def test_max_corrections_is_respected(self) -> None:
        current = np.zeros(200)
        candidates = np.ones(200, dtype=bool)
        probs = np.linspace(0.0, 1.0, 200)
        accepted = corrections.cap_corrections(
            current, candidates, probs, band=(0.0, 1.0), max_corrections=7
        )
        self.assertEqual(int(accepted.sum()), 7)


class CoDriftTests(unittest.TestCase):
    def test_pairs_are_the_three_correlated_ones(self) -> None:
        self.assertEqual(
            corrections.CORRELATED_PAIRS,
            (
                ("ACL", "Contusion"),
                ("Medial Meniscus", "Medial OA"),
                ("Lateral Meniscus", "Lateral OA"),
            ),
        )

    def _flags(self) -> pd.DataFrame:
        labels_frame = synthetic_labels()
        flags = pd.DataFrame({STUDY_ID_COLUMN: labels_frame[STUDY_ID_COLUMN]})
        for label in TARGET_LABELS:
            flags[label] = False
        return labels_frame, flags

    def test_independent_corrections_do_not_flag(self) -> None:
        labels_frame, flags = self._flags()
        flags.loc[0:19, "ACL"] = True
        flags.loc[100:119, "Contusion"] = True
        rows = corrections.codrift_report(labels_frame, flags)
        acl_row = next(row for row in rows if row["pair"] == ["ACL", "Contusion"])
        self.assertEqual(acl_row["joint_corrections"], 0)
        self.assertFalse(acl_row["flagged"])

    def test_injected_correlated_flip_fires_the_flag(self) -> None:
        labels_frame, flags = self._flags()
        flags.loc[0:19, "ACL"] = True
        flags.loc[0:19, "Contusion"] = True
        rows = corrections.codrift_report(labels_frame, flags)
        acl_row = next(row for row in rows if row["pair"] == ["ACL", "Contusion"])
        self.assertEqual(acl_row["joint_corrections"], 20)
        self.assertGreater(acl_row["ratio"], corrections.CODRIFT_RATIO_THRESHOLD)
        self.assertTrue(acl_row["flagged"])
        self.assertFalse(
            next(row for row in rows if row["pair"] == ["Medial Meniscus", "Medial OA"])["flagged"]
        )


class JointBandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels_frame = synthetic_labels()
        self.probs = synthetic_probs(self.labels_frame)
        self.weights = {label: 0.9 for label in TARGET_LABELS}
        self.thresholds = {label: (0.5, "empirical") for label in TARGET_LABELS}

    def _flags(self, **per_label) -> pd.DataFrame:
        flags = pd.DataFrame({STUDY_ID_COLUMN: self.labels_frame[STUDY_ID_COLUMN]})
        for label in TARGET_LABELS:
            flags[label] = False
        for label, indices in per_label.items():
            flags.loc[indices, label] = True
        return flags

    def test_band_is_twenty_percent_relative(self) -> None:
        self.assertEqual(corrections.JOINT_BAND_RELATIVE, 0.20)
        self.assertEqual(corrections.joint_band(0.30), (0.30 * 0.8, 0.30 * 1.2))

    def test_baseline_cooccurrence_covers_the_three_pairs(self) -> None:
        baseline = corrections.baseline_cooccurrence(self.labels_frame)
        self.assertEqual(set(baseline), set(corrections.CORRELATED_PAIRS))
        # synthetic_labels marks the first 30% of studies positive on every label.
        for value in baseline.values():
            self.assertAlmostEqual(value, 0.30)

    def test_in_band_round_rejects_nothing(self) -> None:
        flags = self._flags(ACL=[0, 1, 2])
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        self.assertEqual(
            result.joint_band_rejections["ACL & Contusion"], 0
        )
        self.assertEqual(result.per_label["ACL"].accepted, 3)
        self.assertTrue(all(row["within_band"] for row in result.joint_bands))

    def test_joint_band_rejection_fires_on_a_synthetic_violation(self) -> None:
        # Disjoint flips on both halves of a pair drive the joint-positive rate
        # down twice as fast as either label's own prevalence: 60 joint positives
        # become 44 (0.22), below the 0.24 band floor.
        flags = self._flags(ACL=list(range(0, 8)), Contusion=list(range(8, 16)))
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        key = "ACL & Contusion"
        row = next(item for item in result.joint_bands if item["pair"] == ["ACL", "Contusion"])

        self.assertGreater(result.joint_band_rejections[key], 0)
        self.assertEqual(result.joint_band_rejections[key], row["rejections"])
        self.assertTrue(row["within_band"])
        self.assertGreaterEqual(row["joint_prevalence_after"], row["band_low"] - 1e-9)
        self.assertLessEqual(row["joint_prevalence_after"], row["band_high"] + 1e-9)
        self.assertAlmostEqual(row["baseline_joint_prevalence"], 0.30)

        # Most-confident flips survive up to the band edge: some corrections stick.
        accepted = result.per_label["ACL"].accepted + result.per_label["Contusion"].accepted
        self.assertGreater(accepted, 0)
        self.assertLess(accepted, 16)
        # The co-drift layer did not fire here - the band is what constrained it.
        self.assertEqual(result.paused_pairs, [])

    def test_untouched_pairs_are_unaffected(self) -> None:
        flags = self._flags(ACL=list(range(0, 8)), Contusion=list(range(8, 16)))
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        self.assertEqual(result.joint_band_rejections["Medial Meniscus & Medial OA"], 0)
        self.assertEqual(result.joint_band_rejections["Lateral Meniscus & Lateral OA"], 0)


class ApplyCorrectionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels_frame = synthetic_labels()
        self.probs = synthetic_probs(self.labels_frame)
        self.weights = {label: 0.9 for label in TARGET_LABELS}
        self.thresholds = {label: (0.5, "empirical") for label in TARGET_LABELS}

    def _flags(self, **per_label) -> pd.DataFrame:
        flags = pd.DataFrame({STUDY_ID_COLUMN: self.labels_frame[STUDY_ID_COLUMN]})
        for label in TARGET_LABELS:
            flags[label] = False
        for label, indices in per_label.items():
            flags.loc[indices, label] = True
        return flags

    def test_accepted_corrections_flip_the_label(self) -> None:
        flags = self._flags(ACL=list(range(0, 3)))
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        report = result.per_label["ACL"]
        self.assertEqual(report.candidates, 3)
        self.assertEqual(report.accepted, 3)
        self.assertEqual(report.positives_after, report.positives_before - 3)
        self.assertTrue((result.labels.loc[0:2, "ACL"] == 0.0).all())

    def test_protected_gold_studies_are_never_corrected(self) -> None:
        flags = self._flags(ACL=list(range(0, 3)))
        protected = set(self.labels_frame[STUDY_ID_COLUMN][:3])
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
            protected_ids=protected,
        )
        self.assertEqual(result.per_label["ACL"].candidates, 0)
        self.assertEqual(result.n_skipped, 3)

    def test_flagged_pair_is_paused_and_reverted(self) -> None:
        flags = self._flags(ACL=list(range(0, 12)), Contusion=list(range(0, 12)))
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        self.assertIn(["ACL", "Contusion"], result.paused_pairs)
        pd.testing.assert_series_equal(
            result.labels["ACL"], self.labels_frame["ACL"], check_names=False
        )
        for label in ("ACL", "Contusion"):
            item = result.per_label[label]
            self.assertTrue(item.paused)
            self.assertEqual(item.accepted, 0)
            self.assertEqual(item.capped, item.candidates)
            self.assertEqual(item.positives_after, item.positives_before)
        self.assertFalse(result.flags["ACL"].any())
        self.assertFalse(result.per_label["MCL"].paused)

    def test_identity_round_changes_nothing(self) -> None:
        flags = self._flags(**{label: list(range(0, 5)) for label in TARGET_LABELS})
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
            max_corrections=0,
        )
        self.assertTrue(all(item.accepted == 0 for item in result.per_label.values()))
        pd.testing.assert_frame_equal(
            result.labels[TARGET_LABELS], self.labels_frame[TARGET_LABELS]
        )

    def test_report_is_json_shaped(self) -> None:
        import json

        flags = self._flags(ACL=[0, 1])
        result = corrections.apply_corrections(
            self.labels_frame,
            self.probs,
            flags,
            label_weights=self.weights,
            thresholds=self.thresholds,
        )
        payload = json.loads(json.dumps(result.report()))
        self.assertEqual(payload["per_label"]["ACL"]["accepted"], 2)
        self.assertEqual(len(payload["codrift"]), 3)
        self.assertEqual(len(payload["joint_bands"]), 3)
        self.assertEqual(
            set(payload["joint_band_rejections"]),
            {"ACL & Contusion", "Medial Meniscus & Medial OA", "Lateral Meniscus & Lateral OA"},
        )


class LabelWeightTests(unittest.TestCase):
    def test_loads_the_committed_weights(self) -> None:
        path = REPO_ROOT / corrections.DEFAULT_LABEL_WEIGHTS
        if not path.exists():
            self.skipTest("label_weights.json not present")
        weights = corrections.load_label_weights(path)
        self.assertEqual(set(weights), set(TARGET_LABELS))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in weights.values()))


class CleanlabTests(unittest.TestCase):
    def test_find_candidates_flags_injected_noise(self) -> None:
        pytest.importorskip("cleanlab")
        labels_frame = synthetic_labels()
        probs = synthetic_probs(labels_frame, flip_strength=0.98)
        # Injected noise: ten studies whose given label contradicts a confident teacher.
        flipped = labels_frame.copy()
        flipped.loc[0:9, "ACL"] = 0.0
        probs.loc[0:9, "ACL_pred"] = 0.99
        flags = corrections.find_candidates(flipped, probs, labels=["ACL"])
        self.assertGreaterEqual(int(flags["ACL"].sum()), 1)

    def test_run_corrections_end_to_end(self) -> None:
        pytest.importorskip("cleanlab")
        labels_frame = synthetic_labels()
        probs = synthetic_probs(labels_frame, flip_strength=0.98)
        result = corrections.run_corrections(
            labels_frame,
            probs,
            label_weights={label: 0.9 for label in TARGET_LABELS},
            gold_positive_counts={label: 20 for label in TARGET_LABELS},
            max_corrections=0,
        )
        self.assertEqual(result.n_studies, N)
        self.assertTrue(all(item.accepted == 0 for item in result.per_label.values()))


if __name__ == "__main__":
    unittest.main()
