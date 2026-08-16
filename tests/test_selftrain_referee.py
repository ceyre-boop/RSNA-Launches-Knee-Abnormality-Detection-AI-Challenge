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


def synthetic_oof(n: int = 200, *, noise: float, seed: int = 0, prefix: str = "s") -> pd.DataFrame:
    """Predictions that get better as ``noise`` shrinks; truth is fixed by index."""
    rng = random.Random(seed)
    data = {STUDY_ID_COLUMN: [f"{prefix}-{index:04d}" for index in range(n)]}
    for offset, label in enumerate(LABELS):
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
