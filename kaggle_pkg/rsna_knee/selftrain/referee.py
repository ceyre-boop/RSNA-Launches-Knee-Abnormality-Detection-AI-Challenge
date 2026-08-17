"""The referee: what is allowed to accept or reject a self-training round.

Primary gate is cross-validated (``cv_gate``) on the ~877-study fold OOF, because
gold-58 has no power to resolve the 0.01-0.03 gains a round produces. That gate
is stratified: only ``TRUSTED_LABELS`` decide accept/reject, because on
``GAP_LABELS`` the pseudo-label noise is directionally biased and CV movement is
anti-correlated with the real objective (see ``cv_gate`` and ``CV_LB_ANCHOR``).
The gold working slice gives an advisory directional read every second round
(``gold_check``); the locked slice is spent exactly once, at the end of the
campaign (``final_check``), and only as a smoke alarm for catastrophic drift.

Every threshold below is pre-committed. They are constants, not parameters, so
that nothing is tuned after peeking at a result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS
from ..metrics import binary_roc_auc
from .gold_split import GoldSplit, load_split

# --- pre-committed constants -------------------------------------------------
CV_GATE_MACRO_MIN_DELTA = 0.003
CV_GATE_PER_LABEL_DEGRADATION_FLOOR = 0.01

# Step-zero record: the one measured CV/LB pair this campaign is calibrated on.
# CV underestimates the image-annotated truth because the pseudo-label noise is
# directional (findings visible on the image, absent from the report), so the
# public LB reads higher than the fold OOF on the same checkpoint.
CV_LB_ANCHOR = {
    "cv": 0.804,
    "lb": 0.840,
    "date": "2026-08-16",
    "note": "fold-0 v1.1; LB>CV consistent with directional report-gap bias",
}

# Labels whose report extraction quality is >= 0.83 and whose residual error the
# audit found to be threshold-artifact-dominated: CV movement on these tracks the
# real objective, so they alone decide accept/reject.
TRUSTED_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Baker's",
    "Fracture",
]

# Labels the audit found to be annotator-vs-report-dominated: the image is
# annotated positive and the report is silent. CV on these measures agreement
# with the report, not with the truth being scored, so it is uninformative and
# strictly advisory. The leaderboard is the arbiter for these six.
GAP_LABELS = [
    "Effusion",
    "Synovitis",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Contusion",
]

# final_check is a smoke alarm, not a thermometer: n=18 at ~22% label noise can
# only see a fire, never a degree.
SMOKE_ALARM_FLOOR = -0.10

GOLD_BOOTSTRAP_DRAWS = 3000
GOLD_DIRECTION_AGREEMENT = 0.90
GOLD_EFFECT_FLOOR = 0.02
GOLD_CHECK_EVERY_N_ROUNDS = 2
BOOTSTRAP_SEED = 2026

MAX_ROUNDS = 3

TRUE_SUFFIX = "_true"
PRED_SUFFIX = "_pred"


class LockedSliceError(RuntimeError):
    """Raised when an evaluation would touch the locked gold slice."""


@dataclass(frozen=True)
class CvGateResult:
    accepted: bool
    macro_prev: float
    macro_new: float
    macro_delta: float
    per_label_delta: Dict[str, float]
    degraded: List[str]
    n_studies: int
    labels: List[str]
    reasons: List[str] = field(default_factory=list)
    # Gap-label deltas are reported, never gated on. See ``cv_gate``.
    advisory_gap_deltas: Dict[str, float] = field(default_factory=dict)
    gap_labels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GoldCheckResult:
    ran: bool
    slice_name: str
    n_studies: int
    macro_delta: float
    per_label_delta: Dict[str, float]
    direction_agreement: float
    draws: int
    signal: str  # "positive" | "negative" | "none" | "skipped"
    veto: bool
    labels: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    # True on the locked slice: the read can only flag catastrophic drift.
    smoke_alarm_only: bool = False


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    rollback: bool
    reason: str


# --- OOF plumbing ------------------------------------------------------------


def load_oof(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Accept a path or a frame; normalise the study id column to str."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if STUDY_ID_COLUMN not in frame.columns:
        raise ValueError(f"OOF frame missing {STUDY_ID_COLUMN}")
    frame[STUDY_ID_COLUMN] = frame[STUDY_ID_COLUMN].astype(str)
    return frame.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").reset_index(drop=True)


def scored_labels(frame: pd.DataFrame, labels: Iterable[str] = TARGET_LABELS) -> List[str]:
    """Labels with a truth column, a prediction column, and both classes present."""
    usable = []
    for label in labels:
        truth_column, pred_column = label + TRUE_SUFFIX, label + PRED_SUFFIX
        if truth_column not in frame.columns or pred_column not in frame.columns:
            continue
        truth = (pd.to_numeric(frame[truth_column], errors="coerce") >= 0.5).astype(int)
        if truth.min() == truth.max():
            continue
        usable.append(label)
    return usable


def per_label_auc(frame: pd.DataFrame, labels: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for label in labels:
        truth = (pd.to_numeric(frame[label + TRUE_SUFFIX], errors="coerce") >= 0.5).astype(int)
        preds = pd.to_numeric(frame[label + PRED_SUFFIX], errors="coerce").fillna(0.5)
        scores[label] = binary_roc_auc(truth.tolist(), preds.tolist())
    return scores


def _align(prev: pd.DataFrame, new: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shared = sorted(set(prev[STUDY_ID_COLUMN]) & set(new[STUDY_ID_COLUMN]))
    if not shared:
        raise ValueError("previous and new OOF frames share no studies")
    prev_aligned = prev.set_index(STUDY_ID_COLUMN).loc[shared].reset_index()
    new_aligned = new.set_index(STUDY_ID_COLUMN).loc[shared].reset_index()
    return prev_aligned, new_aligned


# --- locked-slice enforcement ------------------------------------------------


def _resolve_split(split: str | Path | GoldSplit) -> GoldSplit:
    return split if isinstance(split, GoldSplit) else load_split(split)


def assert_locked_excluded(
    study_ids: Iterable[str],
    split: str | Path | GoldSplit,
    *,
    final_unlock: bool = False,
    context: str = "evaluation",
) -> None:
    """Raise unless the evaluated studies are free of the locked gold slice."""
    if final_unlock:
        return
    locked = _resolve_split(split).locked_set
    offenders = sorted(locked & {str(value) for value in study_ids})
    if offenders:
        raise LockedSliceError(
            f"{context} includes {len(offenders)} locked gold studies "
            f"(e.g. {offenders[0]}); pass final_unlock=True only for the single "
            "end-of-campaign final check"
        )


def restrict_to_working(
    frame: pd.DataFrame,
    split: str | Path | GoldSplit,
) -> pd.DataFrame:
    """Drop everything that is not a working-slice gold study."""
    working = _resolve_split(split).working_set
    return frame[frame[STUDY_ID_COLUMN].isin(working)].reset_index(drop=True)


def restrict_to_locked(
    frame: pd.DataFrame,
    split: str | Path | GoldSplit,
) -> pd.DataFrame:
    locked = _resolve_split(split).locked_set
    return frame[frame[STUDY_ID_COLUMN].isin(locked)].reset_index(drop=True)


# --- gates -------------------------------------------------------------------


def cv_gate(
    oof_prev: str | Path | pd.DataFrame,
    oof_new: str | Path | pd.DataFrame,
    *,
    split: Optional[str | Path | GoldSplit] = None,
    final_unlock: bool = False,
    labels: Sequence[str] = TARGET_LABELS,
    macro_min_delta: float = CV_GATE_MACRO_MIN_DELTA,
    degradation_floor: float = CV_GATE_PER_LABEL_DEGRADATION_FLOOR,
    trusted_labels: Sequence[str] = TRUSTED_LABELS,
    gap_labels: Sequence[str] = GAP_LABELS,
) -> CvGateResult:
    """Primary accept/reject on fold OOF AUC deltas, stratified by label trust.

    Accept iff the macro delta **over ``TRUSTED_LABELS`` only** is at least
    ``macro_min_delta`` and no trusted label drops by more than
    ``degradation_floor``. Deltas on ``GAP_LABELS`` are still computed and
    returned in ``advisory_gap_deltas``, but they never enter the decision.

    Why stratify. The pseudo-label noise here is not symmetric: on the gap
    labels the finding is visible on the image and simply absent from the
    report, so the OOF truth column disagrees with the truth being scored in a
    single direction. CV movement on those labels measures agreement with the
    report, which is anti-correlated with the real objective - a round that
    learns the finding looks like a regression against report-derived truth. The
    trusted labels (extraction quality >= 0.83, residual error audited as
    threshold-artifact-dominated) do not have that problem, so they carry the
    decision; the leaderboard is the arbiter for the gap labels.

    Empirical anchor (``CV_LB_ANCHOR``): fold-0 CV 0.804 vs LB 0.840, a +0.036
    gap in the direction the bias predicts - CV underestimates image-annotated
    truth. Gating on a pooled macro that includes gap labels would therefore
    reject exactly the rounds that close that gap.

    If ``split`` is given, the evaluated studies are checked against the locked
    slice first.
    """
    prev, new = _align(load_oof(oof_prev), load_oof(oof_new))
    if split is not None:
        assert_locked_excluded(
            prev[STUDY_ID_COLUMN], split, final_unlock=final_unlock, context="cv_gate"
        )

    usable = [label for label in scored_labels(prev, labels) if label in set(scored_labels(new, labels))]
    if not usable:
        raise ValueError("no label is scorable on both OOF frames")

    prev_auc = per_label_auc(prev, usable)
    new_auc = per_label_auc(new, usable)
    deltas = {label: new_auc[label] - prev_auc[label] for label in usable}

    trusted_set, gap_set = set(trusted_labels), set(gap_labels)
    trusted_usable = [label for label in usable if label in trusted_set]
    gap_usable = [label for label in usable if label in gap_set]
    if not trusted_usable:
        raise ValueError(
            "no trusted label is scorable on both OOF frames; the gate cannot be "
            f"decided on gap labels alone (trusted: {sorted(trusted_set)})"
        )

    macro_prev = sum(prev_auc[label] for label in trusted_usable) / len(trusted_usable)
    macro_new = sum(new_auc[label] for label in trusted_usable) / len(trusted_usable)
    macro_delta = macro_new - macro_prev
    degraded = sorted(
        label for label in trusted_usable if deltas[label] < -degradation_floor
    )

    reasons: List[str] = []
    if macro_delta < macro_min_delta:
        reasons.append(
            f"trusted macro delta {macro_delta:+.4f} below floor {macro_min_delta:+.4f}"
        )
    if degraded:
        reasons.append(
            "trusted labels degraded past floor: "
            + ", ".join(f"{label} {deltas[label]:+.4f}" for label in degraded)
        )

    advisory = {label: deltas[label] for label in gap_usable}
    if advisory:
        reasons.append(
            "advisory only (not gated): "
            + ", ".join(f"{label} {advisory[label]:+.4f}" for label in gap_usable)
        )

    accepted = not (macro_delta < macro_min_delta or degraded)
    return CvGateResult(
        accepted=accepted,
        macro_prev=macro_prev,
        macro_new=macro_new,
        macro_delta=macro_delta,
        per_label_delta=deltas,
        degraded=degraded,
        n_studies=len(prev),
        labels=trusted_usable,
        reasons=reasons,
        advisory_gap_deltas=advisory,
        gap_labels=gap_usable,
    )


def bootstrap_direction(
    truth: Sequence[int],
    scores_prev: Sequence[float],
    scores_new: Sequence[float],
    *,
    draws: int = GOLD_BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Stratified bootstrap of the AUC delta.

    Positives and negatives are resampled separately so every draw keeps both
    classes. Returns ``(point_delta, agreement)`` where agreement is the share of
    draws whose delta has the same sign as the point estimate.
    """
    positives = [index for index, value in enumerate(truth) if value == 1]
    negatives = [index for index, value in enumerate(truth) if value != 1]
    if not positives or not negatives:
        raise ValueError("Both positive and negative samples are required")

    point = binary_roc_auc(truth, scores_new) - binary_roc_auc(truth, scores_prev)
    rng = random.Random(seed)
    agree = 0
    for _ in range(draws):
        picks = [rng.choice(positives) for _ in positives] + [
            rng.choice(negatives) for _ in negatives
        ]
        sampled_truth = [truth[index] for index in picks]
        delta = binary_roc_auc(sampled_truth, [scores_new[i] for i in picks]) - binary_roc_auc(
            sampled_truth, [scores_prev[i] for i in picks]
        )
        if (delta > 0) == (point > 0) and delta != 0:
            agree += 1
    return point, agree / draws if draws else 0.0


def should_run_gold_check(round_index: int, every_n: int = GOLD_CHECK_EVERY_N_ROUNDS) -> bool:
    """Gold is looked at every ``every_n``-th round, never every round."""
    return every_n > 0 and round_index > 0 and round_index % every_n == 0


def gold_check(
    oof_prev: str | Path | pd.DataFrame,
    oof_new: str | Path | pd.DataFrame,
    split: str | Path | GoldSplit,
    *,
    round_index: int,
    every_n: int = GOLD_CHECK_EVERY_N_ROUNDS,
    draws: int = GOLD_BOOTSTRAP_DRAWS,
    agreement_floor: float = GOLD_DIRECTION_AGREEMENT,
    effect_floor: float = GOLD_EFFECT_FLOOR,
    labels: Sequence[str] = TARGET_LABELS,
    force: bool = False,
    final_unlock: bool = False,
) -> GoldCheckResult:
    """Advisory directional read on the working gold slice.

    Never approves a round on its own. It can only veto, and only on a strong
    negative: agreement above the floor and an effect at or beyond the floor.
    """
    if not force and not should_run_gold_check(round_index, every_n):
        return GoldCheckResult(
            ran=False,
            slice_name="working",
            n_studies=0,
            macro_delta=float("nan"),
            per_label_delta={},
            direction_agreement=float("nan"),
            draws=0,
            signal="skipped",
            veto=False,
            reasons=[f"round {round_index} is not a gold-check round (every {every_n})"],
        )
    return _bootstrap_slice(
        oof_prev,
        oof_new,
        split,
        slice_name="working",
        draws=draws,
        agreement_floor=agreement_floor,
        effect_floor=effect_floor,
        labels=labels,
        final_unlock=final_unlock,
    )


def final_check(
    oof_prev: str | Path | pd.DataFrame,
    oof_new: str | Path | pd.DataFrame,
    split: str | Path | GoldSplit,
    *,
    final_unlock: bool = False,
    draws: int = GOLD_BOOTSTRAP_DRAWS,
    agreement_floor: float = GOLD_DIRECTION_AGREEMENT,
    effect_floor: float = GOLD_EFFECT_FLOOR,
    labels: Sequence[str] = TARGET_LABELS,
    smoke_alarm_floor: float = SMOKE_ALARM_FLOOR,
) -> GoldCheckResult:
    """Single end-of-campaign read on the locked slice: a smoke alarm, not a thermometer.

    The locked slice is 18 studies carrying roughly 22% label noise. That is
    enough resolution to notice the building on fire and nothing else. So this
    check has exactly one output with any authority: it fires when the macro
    delta is worse than ``smoke_alarm_floor`` (-0.10), i.e. catastrophic drift.

    It cannot confirm an improvement. A positive delta here is not evidence the
    campaign worked - at n=18 the confidence interval swallows every gain a
    round can plausibly produce, so ``signal`` is never "positive" and ``veto``
    is never lifted by a good read. Do not report this number as a result.
    ``smoke_alarm_only`` is set to ``True`` on the returned object to keep that
    reading attached to the data.

    Requires ``final_unlock``.
    """
    if not final_unlock:
        raise LockedSliceError(
            "final_check reads the locked gold slice; call it once, with final_unlock=True"
        )
    return _bootstrap_slice(
        oof_prev,
        oof_new,
        split,
        slice_name="locked",
        draws=draws,
        agreement_floor=agreement_floor,
        effect_floor=effect_floor,
        labels=labels,
        final_unlock=True,
        smoke_alarm=True,
        smoke_alarm_floor=smoke_alarm_floor,
    )


def _bootstrap_slice(
    oof_prev: str | Path | pd.DataFrame,
    oof_new: str | Path | pd.DataFrame,
    split: str | Path | GoldSplit,
    *,
    slice_name: str,
    draws: int,
    agreement_floor: float,
    effect_floor: float,
    labels: Sequence[str],
    final_unlock: bool,
    smoke_alarm: bool = False,
    smoke_alarm_floor: float = SMOKE_ALARM_FLOOR,
) -> GoldCheckResult:
    resolved = _resolve_split(split)
    prev, new = _align(load_oof(oof_prev), load_oof(oof_new))
    if slice_name == "working":
        prev, new = restrict_to_working(prev, resolved), restrict_to_working(new, resolved)
    else:
        prev, new = restrict_to_locked(prev, resolved), restrict_to_locked(new, resolved)
    assert_locked_excluded(
        prev[STUDY_ID_COLUMN], resolved, final_unlock=final_unlock, context=f"{slice_name} gold check"
    )
    if prev.empty:
        raise ValueError(f"no {slice_name} gold studies present in the supplied OOF frames")

    usable = [label for label in scored_labels(prev, labels) if label in set(scored_labels(new, labels))]
    if not usable:
        raise ValueError(f"no label is scorable on the {slice_name} gold slice")

    deltas: Dict[str, float] = {}
    agreements: List[float] = []
    for label in usable:
        truth = (
            (pd.to_numeric(prev[label + TRUE_SUFFIX], errors="coerce") >= 0.5).astype(int).tolist()
        )
        scores_prev = pd.to_numeric(prev[label + PRED_SUFFIX], errors="coerce").fillna(0.5).tolist()
        scores_new = pd.to_numeric(new[label + PRED_SUFFIX], errors="coerce").fillna(0.5).tolist()
        delta, agreement = bootstrap_direction(truth, scores_prev, scores_new, draws=draws)
        deltas[label] = delta
        agreements.append(agreement)

    macro_delta = sum(deltas.values()) / len(deltas)
    macro_agreement = sum(agreements) / len(agreements)

    reasons = [
        f"macro delta {macro_delta:+.4f} on {len(prev)} {slice_name} gold studies",
        f"direction agreement {macro_agreement:.3f} over {draws} draws",
    ]

    if smoke_alarm:
        # Smoke alarm: the only thing this slice can resolve is a fire.
        signal = "catastrophic" if macro_delta < smoke_alarm_floor else "none"
        reasons.append(
            f"smoke alarm only: n={len(prev)} at ~22% label noise can flag drift "
            f"worse than {smoke_alarm_floor:+.2f} and nothing else"
        )
        if signal == "none":
            reasons.append(
                "no catastrophic drift detected; this is NOT confirmation of improvement"
            )
        return GoldCheckResult(
            ran=True,
            slice_name=slice_name,
            n_studies=len(prev),
            macro_delta=macro_delta,
            per_label_delta=deltas,
            direction_agreement=macro_agreement,
            draws=draws,
            signal=signal,
            veto=signal == "catastrophic",
            labels=usable,
            reasons=reasons,
            smoke_alarm_only=True,
        )

    strong = macro_agreement >= agreement_floor and abs(macro_delta) >= effect_floor
    if not strong:
        signal = "none"
    else:
        signal = "positive" if macro_delta > 0 else "negative"

    if signal == "positive":
        reasons.append("advisory only: a positive gold read never approves a round")

    return GoldCheckResult(
        ran=True,
        slice_name=slice_name,
        n_studies=len(prev),
        macro_delta=macro_delta,
        per_label_delta=deltas,
        direction_agreement=macro_agreement,
        draws=draws,
        signal=signal,
        veto=signal == "negative",
        labels=usable,
        reasons=reasons,
    )


def stop_decision(
    round_index: int,
    cv_result: CvGateResult,
    gold_result: Optional[GoldCheckResult] = None,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> StopDecision:
    """Pre-committed stopping rule: reject once, or hit the round cap, and stop."""
    if not cv_result.accepted:
        return StopDecision(
            stop=True,
            rollback=True,
            reason="cv_gate rejected: " + "; ".join(cv_result.reasons),
        )
    if gold_result is not None and gold_result.veto:
        return StopDecision(
            stop=True,
            rollback=True,
            reason="gold_check strong negative: " + "; ".join(gold_result.reasons),
        )
    if round_index >= max_rounds:
        return StopDecision(
            stop=True, rollback=False, reason=f"round cap reached ({max_rounds} rounds)"
        )
    return StopDecision(stop=False, rollback=False, reason="accepted; continue")
