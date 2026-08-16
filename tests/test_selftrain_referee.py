import pathlib
import random
import sys
import unittest

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import STUDY_ID_COLUMN, TARGET_LABELS  # noqa: E402
from rsna_knee.selftrain import referee  # noqa: E402
from rsna_knee.selftrain.gold_split import GoldSplit  # noqa: E402

LABELS = TARGET_LABELS[:3]
ALL_LABELS = list(referee.TRUSTED_LABELS) + list(referee.GAP_LABELS)


def synthetic_oof(
    n: int = 200,
    *,
    noise: float,
    seed: int = 0,
    prefix: str = "s",
    labels=LABELS,
) -> pd.DataFrame:
    """Predictions that get better as ``noise`` shrinks; truth is fixed by index."""
    rng = random.Random(seed)
    data = {STUDY_ID_COLUMN: [f"{prefix}-{index:04d}" for index in range(n)]}
    for offset, label in enumerate(labels):
        truth = [1 if (index + offset) % 3 == 0 else 0 for index in range(n)]
        preds = [value + rng.uniform(-noise, noise) for value in truth]
        data[label + "_true"] = truth
        data[label + "_pred"] = preds
    return pd.DataFrame(data)


class CvGateTests(unittest.TestCase):
    def test_accepts_a_clear_improvement(self) -> None:
        prev = synthetic_oof(noise=0.9, seed=1)
        new = synthetic_oof(noise=0.35, seed=1)
        result = referee.cv_gate(prev, new, labels=LABELS)
        self.assertTrue(result.accepted, msg=result.reasons)
        self.assertGreaterEqual(result.macro_delta, referee.CV_GATE_MACRO_MIN_DELTA)
        self.assertEqual(result.n_studies, 200)

    def test_rejects_a_flat_round(self) -> None:
        prev = synthetic_oof(noise=0.6, seed=2)
        result = referee.cv_gate(prev, prev.copy(), labels=LABELS)
        self.assertFalse(result.accepted)
        self.assertAlmostEqual(result.macro_delta, 0.0)
        self.assertTrue(any("macro delta" in reason for reason in result.reasons))

    def test_rejects_when_one_label_degrades_past_the_floor(self) -> None:
        prev = synthetic_oof(noise=0.9, seed=3)
        new = synthetic_oof(noise=0.2, seed=3)
        broken = LABELS[0]
        new[broken + "_pred"] = [1.0 - value for value in new[broken + "_pred"]]
        result = referee.cv_gate(prev, new, labels=LABELS)
        self.assertFalse(result.accepted)
        self.assertIn(broken, result.degraded)

    def test_constants_are_the_precommitted_values(self) -> None:
        self.assertEqual(referee.CV_GATE_MACRO_MIN_DELTA, 0.003)
        self.assertEqual(referee.CV_GATE_PER_LABEL_DEGRADATION_FLOOR, 0.01)
        self.assertEqual(referee.GOLD_BOOTSTRAP_DRAWS, 3000)
        self.assertEqual(referee.GOLD_DIRECTION_AGREEMENT, 0.90)
        self.assertEqual(referee.GOLD_EFFECT_FLOOR, 0.02)
        self.assertEqual(referee.GOLD_CHECK_EVERY_N_ROUNDS, 2)
        self.assertEqual(referee.MAX_ROUNDS, 3)


class StratifiedCvGateTests(unittest.TestCase):
    """The gate decides on TRUSTED_LABELS only; GAP_LABELS are advisory."""

    def _pair(self, seed: int = 7):
        prev = synthetic_oof(noise=0.9, seed=seed, labels=ALL_LABELS)
        new = synthetic_oof(noise=0.3, seed=seed, labels=ALL_LABELS)
        return prev, new

    @staticmethod
    def _invert(frame: pd.DataFrame, label: str) -> None:
        frame[label + "_pred"] = [1.0 - value for value in frame[label + "_pred"]]

    def test_label_partition_is_the_precommitted_one(self) -> None:
        self.assertEqual(
            referee.TRUSTED_LABELS,
            ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Baker's", "Fracture"],
        )
        self.assertEqual(
            referee.GAP_LABELS,
            ["Effusion", "Synovitis", "Medial OA", "Lateral OA", "PF OA", "Contusion"],
        )
        # The two sets partition the target labels exactly, with no overlap.
        self.assertEqual(
            set(referee.TRUSTED_LABELS) | set(referee.GAP_LABELS), set(TARGET_LABELS)
        )
        self.assertEqual(set(referee.TRUSTED_LABELS) & set(referee.GAP_LABELS), set())

    def test_cv_lb_anchor_records_step_zero(self) -> None:
        self.assertEqual(
            referee.CV_LB_ANCHOR,
            {
                "cv": 0.804,
                "lb": 0.840,
                "date": "2026-08-16",
                "note": "fold-0 v1.1; LB>CV consistent with directional report-gap bias",
            },
        )
        self.assertGreater(referee.CV_LB_ANCHOR["lb"], referee.CV_LB_ANCHOR["cv"])

    def test_accepts_when_trusted_improve_even_if_gap_labels_degrade(self) -> None:
        prev, new = self._pair()
        for label in referee.GAP_LABELS:
            self._invert(new, label)
        result = referee.cv_gate(prev, new, labels=ALL_LABELS)

        self.assertTrue(result.accepted, msg=result.reasons)
        # The gate macro is over trusted labels only.
        self.assertEqual(set(result.labels), set(referee.TRUSTED_LABELS))
        self.assertGreaterEqual(result.macro_delta, referee.CV_GATE_MACRO_MIN_DELTA)
        # Gap labels are visibly wrecked, reported, and ignored.
        self.assertEqual(set(result.gap_labels), set(referee.GAP_LABELS))
        self.assertEqual(set(result.advisory_gap_deltas), set(referee.GAP_LABELS))
        self.assertTrue(
            all(delta < -referee.CV_GATE_PER_LABEL_DEGRADATION_FLOOR
                for delta in result.advisory_gap_deltas.values())
        )
        self.assertEqual(result.degraded, [])

    def test_rejects_when_a_trusted_label_degrades(self) -> None:
        prev, new = self._pair()
        broken = referee.TRUSTED_LABELS[2]
        self._invert(new, broken)
        result = referee.cv_gate(prev, new, labels=ALL_LABELS)

        self.assertFalse(result.accepted)
        self.assertIn(broken, result.degraded)
        self.assertTrue(any("trusted labels degraded" in reason for reason in result.reasons))

    def test_gap_deltas_cannot_rescue_a_flat_trusted_macro(self) -> None:
        prev, _ = self._pair()
        new = prev.copy()
        # Gap labels improve a lot; trusted labels do not move at all.
        better = synthetic_oof(noise=0.05, seed=7, labels=ALL_LABELS)
        for label in referee.GAP_LABELS:
            new[label + "_pred"] = better[label + "_pred"]
        result = referee.cv_gate(prev, new, labels=ALL_LABELS)

        self.assertFalse(result.accepted)
        self.assertAlmostEqual(result.macro_delta, 0.0)
        self.assertTrue(all(delta > 0 for delta in result.advisory_gap_deltas.values()))

    def test_gate_refuses_to_decide_on_gap_labels_alone(self) -> None:
        prev = synthetic_oof(noise=0.9, seed=8, labels=referee.GAP_LABELS)
        new = synthetic_oof(noise=0.2, seed=8, labels=referee.GAP_LABELS)
        with self.assertRaises(ValueError):
            referee.cv_gate(prev, new, labels=referee.GAP_LABELS)


class GoldCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        ids = [f"g-{index:04d}" for index in range(58)]
        self.split = GoldSplit(version="v1", seed=2026, working=ids[:40], locked=ids[40:])
        self.prev = synthetic_oof(58, noise=0.9, seed=5, prefix="g")
        self.new_better = synthetic_oof(58, noise=0.05, seed=5, prefix="g")
        self.new_worse = self.prev.copy()
        for label in LABELS:
            self.new_worse[label + "_pred"] = [
                1.0 - value for value in synthetic_oof(58, noise=0.05, seed=5, prefix="g")[label + "_pred"]
            ]

    def test_runs_only_on_every_second_round(self) -> None:
        self.assertFalse(referee.should_run_gold_check(1))
        self.assertTrue(referee.should_run_gold_check(2))
        self.assertFalse(referee.should_run_gold_check(3))
        skipped = referee.gold_check(
            self.prev, self.new_better, self.split, round_index=1, labels=LABELS
        )
        self.assertFalse(skipped.ran)
        self.assertEqual(skipped.signal, "skipped")

    def test_positive_direction_never_approves(self) -> None:
        result = referee.gold_check(
            self.prev, self.new_better, self.split, round_index=2, draws=300, labels=LABELS
        )
        self.assertTrue(result.ran)
        self.assertEqual(result.n_studies, 40)
        self.assertEqual(result.signal, "positive")
        self.assertFalse(result.veto)

    def test_strong_negative_vetoes(self) -> None:
        result = referee.gold_check(
            self.prev, self.new_worse, self.split, round_index=2, draws=300, labels=LABELS
        )
        self.assertEqual(result.signal, "negative")
        self.assertTrue(result.veto)
        self.assertLess(result.macro_delta, -referee.GOLD_EFFECT_FLOOR)

    def test_gold_check_never_touches_the_locked_slice(self) -> None:
        result = referee.gold_check(
            self.prev, self.new_better, self.split, round_index=2, draws=100, labels=LABELS
        )
        self.assertEqual(result.slice_name, "working")
        self.assertEqual(result.n_studies, len(self.split.working))

    def test_final_check_requires_the_unlock(self) -> None:
        with self.assertRaises(referee.LockedSliceError):
            referee.final_check(self.prev, self.new_better, self.split)
        result = referee.final_check(
            self.prev, self.new_better, self.split, final_unlock=True, draws=100, labels=LABELS
        )
        self.assertEqual(result.slice_name, "locked")
        self.assertEqual(result.n_studies, len(self.split.locked))

    def test_final_check_is_a_smoke_alarm_not_a_thermometer(self) -> None:
        result = referee.final_check(
            self.prev, self.new_better, self.split, final_unlock=True, draws=100, labels=LABELS
        )
        self.assertTrue(result.smoke_alarm_only)
        self.assertGreater(result.macro_delta, 0.0)
        # A large positive delta still cannot be called an improvement at n=18.
        self.assertEqual(result.signal, "none")
        self.assertFalse(result.veto)
        self.assertTrue(any("smoke alarm only" in reason for reason in result.reasons))
        self.assertTrue(
            any("NOT confirmation of improvement" in reason for reason in result.reasons)
        )

    def test_final_check_fires_on_catastrophic_drift(self) -> None:
        result = referee.final_check(
            self.prev, self.new_worse, self.split, final_unlock=True, draws=100, labels=LABELS
        )
        self.assertTrue(result.smoke_alarm_only)
        self.assertLess(result.macro_delta, referee.SMOKE_ALARM_FLOOR)
        self.assertEqual(result.signal, "catastrophic")
        self.assertTrue(result.veto)

    def test_smoke_alarm_floor_is_precommitted(self) -> None:
        self.assertEqual(referee.SMOKE_ALARM_FLOOR, -0.10)

    def test_working_gold_check_is_not_a_smoke_alarm(self) -> None:
        result = referee.gold_check(
            self.prev, self.new_better, self.split, round_index=2, draws=100, labels=LABELS
        )
        self.assertFalse(result.smoke_alarm_only)

    def test_cv_gate_rejects_locked_studies_when_a_split_is_given(self) -> None:
        with self.assertRaises(referee.LockedSliceError):
            referee.cv_gate(self.prev, self.new_better, split=self.split, labels=LABELS)
        result = referee.cv_gate(
            referee.restrict_to_working(referee.load_oof(self.prev), self.split),
            referee.restrict_to_working(referee.load_oof(self.new_better), self.split),
            split=self.split,
            labels=LABELS,
        )
        self.assertTrue(result.accepted)

    def test_bootstrap_direction_agrees_on_a_real_improvement(self) -> None:
        truth = [1 if index % 3 == 0 else 0 for index in range(60)]
        rng = random.Random(11)
        weak = [value + rng.uniform(-0.9, 0.9) for value in truth]
        strong = [value + rng.uniform(-0.05, 0.05) for value in truth]
        delta, agreement = referee.bootstrap_direction(truth, weak, strong, draws=200)
        self.assertGreater(delta, 0.0)
        self.assertGreater(agreement, referee.GOLD_DIRECTION_AGREEMENT)


class StopRuleTests(unittest.TestCase):
    def _result(self, accepted: bool) -> referee.CvGateResult:
        return referee.CvGateResult(
            accepted=accepted,
            macro_prev=0.80,
            macro_new=0.81,
            macro_delta=0.01,
            per_label_delta={},
            degraded=[],
            n_studies=877,
            labels=list(LABELS),
            reasons=[] if accepted else ["macro delta below floor"],
        )

    def test_rejection_stops_and_rolls_back(self) -> None:
        decision = referee.stop_decision(1, self._result(False))
        self.assertTrue(decision.stop)
        self.assertTrue(decision.rollback)

    def test_round_cap_stops_without_rollback(self) -> None:
        decision = referee.stop_decision(referee.MAX_ROUNDS, self._result(True))
        self.assertTrue(decision.stop)
        self.assertFalse(decision.rollback)

    def test_accepted_mid_campaign_round_continues(self) -> None:
        decision = referee.stop_decision(1, self._result(True))
        self.assertFalse(decision.stop)


if __name__ == "__main__":
    unittest.main()
