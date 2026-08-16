"""Per-label pseudo-label correction with caps and co-drift guards.

Candidate flips come from one-vs-rest Confident Learning (cleanlab) over the
teacher's out-of-fold probabilities against the current label set. Everything
after that is deliberately conservative:

- low-count labels (fewer than ten positives in the gold working slice) do not
  get to set their own threshold from a handful of examples; theirs is pooled
  across labels, weighted by CV-estimated per-label AUC
- accepted flips per label are capped so post-correction prevalence stays inside
  a band around current prevalence, and the band tightens as label AUC falls
- the three correlated pairs carry a hard joint-prevalence band: post-correction
  joint-positive rate must stay within +/-20% relative of the pre-round baseline,
  and flips that would breach it are rejected (most confident kept to the edge)
- on top of that band, the same pairs are still watched for joint drift; a pair
  that moves together far more than chance pauses that pair for manual review

``cleanlab`` is imported lazily so the rest of the module - caps, bands, co-drift
- runs and is testable without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS

# --- pre-committed constants -------------------------------------------------
LOW_COUNT_POSITIVE_THRESHOLD = 10

# Prevalence band as a fraction of current prevalence. Interpolated on label AUC:
# a 0.65-AUC label gets the tightest band, a 1.0-AUC label the widest.
AUC_FLOOR = 0.65
AUC_CEIL = 1.00
MIN_BAND_RELATIVE = 0.02
MAX_BAND_RELATIVE = 0.20

CORRELATED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("ACL", "Contusion"),
    ("Medial Meniscus", "Medial OA"),
    ("Lateral Meniscus", "Lateral OA"),
)
CODRIFT_RATIO_THRESHOLD = 2.0
CODRIFT_MIN_JOINT = 3

# Hard constraint, not a monitor: the joint-positive rate of each correlated pair
# must stay within +/-20% RELATIVE of its pre-round baseline. Corrections that
# would push a pair outside the band are rejected (most-confident flips are kept
# up to the band edge). The co-drift pause remains as a second layer on top.
JOINT_BAND_RELATIVE = 0.20

DEFAULT_LABEL_WEIGHTS = "artifacts/fold0/label_weights.json"
PRED_SUFFIX = "_pred"


class CleanlabUnavailableError(ImportError):
    """Raised when Confident Learning is requested without cleanlab installed."""


@dataclass(frozen=True)
class LabelCorrection:
    label: str
    auc: float
    threshold: float
    threshold_source: str
    candidates: int
    accepted: int
    capped: int
    positives_before: int
    positives_after: int
    prevalence_before: float
    prevalence_after: float
    band_low: float
    band_high: float
    paused: bool = False


@dataclass(frozen=True)
class CorrectionResult:
    labels: pd.DataFrame
    per_label: Dict[str, LabelCorrection]
    codrift: List[Dict[str, object]]
    paused_pairs: List[List[str]]
    n_studies: int
    n_skipped: int
    flags: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    joint_bands: List[Dict[str, object]] = field(default_factory=list)
    joint_band_rejections: Dict[str, int] = field(default_factory=dict)

    def report(self) -> Dict[str, object]:
        return {
            "n_studies": self.n_studies,
            "n_skipped_gold": self.n_skipped,
            "per_label": {
                label: {
                    "auc": item.auc,
                    "threshold": item.threshold,
                    "threshold_source": item.threshold_source,
                    "candidates": item.candidates,
                    "accepted": item.accepted,
                    "capped": item.capped,
                    "positives_before": item.positives_before,
                    "positives_after": item.positives_after,
                    "prevalence_before": item.prevalence_before,
                    "prevalence_after": item.prevalence_after,
                    "band_low": item.band_low,
                    "band_high": item.band_high,
                    "paused": item.paused,
                }
                for label, item in self.per_label.items()
            },
            "codrift": self.codrift,
            "paused_pairs": self.paused_pairs,
            "joint_bands": self.joint_bands,
            "joint_band_rejections": self.joint_band_rejections,
        }


# --- inputs ------------------------------------------------------------------


def load_label_weights(
    path: str | Path = DEFAULT_LABEL_WEIGHTS,
    *,
    labels: Sequence[str] = TARGET_LABELS,
) -> Dict[str, float]:
    """Per-label CV AUC estimates, defaulting to 0.5 for anything missing."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {label: float(payload.get(label, 0.5)) for label in labels}


def working_positive_counts(
    gold_frame: pd.DataFrame,
    working_ids: Iterable[str],
    *,
    labels: Sequence[str] = TARGET_LABELS,
) -> Dict[str, int]:
    """Positive counts per label over the gold working slice."""
    subset = gold_frame[gold_frame[STUDY_ID_COLUMN].astype(str).isin({str(i) for i in working_ids})]
    return {label: int((pd.to_numeric(subset[label], errors="coerce") >= 0.5).sum()) for label in labels}


def align_frames(
    labels_frame: pd.DataFrame,
    probs_frame: pd.DataFrame,
    *,
    labels: Sequence[str] = TARGET_LABELS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to the studies they share, in a stable order."""
    left = labels_frame.copy()
    right = probs_frame.copy()
    left[STUDY_ID_COLUMN] = left[STUDY_ID_COLUMN].astype(str)
    right[STUDY_ID_COLUMN] = right[STUDY_ID_COLUMN].astype(str)
    missing = [label for label in labels if label not in left.columns]
    if missing:
        raise ValueError(f"labels frame missing columns: {missing}")
    missing_preds = [label for label in labels if label + PRED_SUFFIX not in right.columns]
    if missing_preds:
        raise ValueError(f"teacher OOF missing columns: {missing_preds}")
    shared = [
        study for study in left[STUDY_ID_COLUMN] if study in set(right[STUDY_ID_COLUMN])
    ]
    if not shared:
        raise ValueError("label frame and teacher OOF share no studies")
    left = left.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").set_index(STUDY_ID_COLUMN)
    right = right.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").set_index(STUDY_ID_COLUMN)
    return left.loc[shared].reset_index(), right.loc[shared].reset_index()


# --- thresholds --------------------------------------------------------------


def derive_thresholds(
    labels_frame: pd.DataFrame,
    probs_frame: pd.DataFrame,
    *,
    label_weights: Dict[str, float],
    gold_positive_counts: Dict[str, int],
    labels: Sequence[str] = TARGET_LABELS,
    low_count: int = LOW_COUNT_POSITIVE_THRESHOLD,
) -> Dict[str, Tuple[float, str]]:
    """cleanlab-style per-class thresholds, with a pooled backoff for rare labels.

    The empirical threshold for a label is the mean predicted probability among
    the studies currently labelled positive. For labels with fewer than
    ``low_count`` positives in the gold working slice (Fracture, MCL) that
    estimate is noise, so it is blended toward an AUC-weighted pool of the other
    labels' thresholds; the weight on the label's own estimate is its own AUC
    rescaled from chance.
    """
    empirical: Dict[str, float] = {}
    for label in labels:
        current = pd.to_numeric(labels_frame[label], errors="coerce").fillna(0.0).to_numpy()
        probs = pd.to_numeric(probs_frame[label + PRED_SUFFIX], errors="coerce").fillna(0.5).to_numpy()
        positives = probs[current >= 0.5]
        empirical[label] = float(positives.mean()) if positives.size else 0.5

    weights = {label: max(float(label_weights.get(label, 0.5)), 0.0) for label in labels}
    total_weight = sum(weights.values())
    pooled = (
        sum(empirical[label] * weights[label] for label in labels) / total_weight
        if total_weight > 0
        else float(np.mean(list(empirical.values())))
    )

    thresholds: Dict[str, Tuple[float, str]] = {}
    for label in labels:
        if gold_positive_counts.get(label, 0) >= low_count:
            thresholds[label] = (empirical[label], "empirical")
            continue
        own_weight = float(np.clip((weights[label] - 0.5) / 0.5, 0.0, 1.0))
        blended = own_weight * empirical[label] + (1.0 - own_weight) * pooled
        thresholds[label] = (float(blended), "pooled_backoff")
    return thresholds


def prevalence_band(prevalence: float, auc: float) -> Tuple[float, float]:
    """Allowed prevalence window after correction; tighter for weaker labels."""
    span = max(AUC_CEIL - AUC_FLOOR, 1e-9)
    scaled = float(np.clip((auc - AUC_FLOOR) / span, 0.0, 1.0))
    relative = MIN_BAND_RELATIVE + (MAX_BAND_RELATIVE - MIN_BAND_RELATIVE) * scaled
    return prevalence * (1.0 - relative), prevalence * (1.0 + relative)


# --- candidate discovery -----------------------------------------------------


def _require_cleanlab():
    try:
        from cleanlab import count as cleanlab_count  # type: ignore
        from cleanlab import filter as cleanlab_filter  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised only without cleanlab
        raise CleanlabUnavailableError(
            "cleanlab>=2.6 is required for Confident Learning candidates; "
            "install it or pass precomputed flags to apply_corrections"
        ) from error
    return cleanlab_filter, cleanlab_count


def find_candidates(
    labels_frame: pd.DataFrame,
    probs_frame: pd.DataFrame,
    *,
    labels: Sequence[str] = TARGET_LABELS,
    thresholds: Optional[Dict[str, Tuple[float, str]]] = None,
) -> pd.DataFrame:
    """One-vs-rest Confident Learning flags, one boolean column per label.

    ``thresholds`` is accepted so the caller's derived per-class thresholds are
    handed to cleanlab rather than recomputed from the same probabilities.
    """
    cleanlab_filter, cleanlab_count = _require_cleanlab()
    flags = pd.DataFrame({STUDY_ID_COLUMN: labels_frame[STUDY_ID_COLUMN].astype(str)})
    for label in labels:
        given = (
            pd.to_numeric(labels_frame[label], errors="coerce").fillna(0.0).to_numpy() >= 0.5
        ).astype(int)
        positive = np.clip(
            pd.to_numeric(probs_frame[label + PRED_SUFFIX], errors="coerce").fillna(0.5).to_numpy(),
            1e-6,
            1 - 1e-6,
        )
        pred_probs = np.stack([1.0 - positive, positive], axis=1)
        if given.min() == given.max():
            flags[label] = False
            continue
        confident_joint = None
        if thresholds is not None and label in thresholds:
            # find_label_issues has no thresholds argument; custom per-class
            # thresholds enter through the confident joint instead.
            positive_threshold = float(np.clip(thresholds[label][0], 1e-6, 1 - 1e-6))
            confident_joint = cleanlab_count.compute_confident_joint(
                labels=given,
                pred_probs=pred_probs,
                thresholds=[1.0 - positive_threshold, positive_threshold],
            )
        issues = cleanlab_filter.find_label_issues(
            labels=given,
            pred_probs=pred_probs,
            filter_by="prune_by_noise_rate",
            confident_joint=confident_joint,
        )
        flags[label] = np.asarray(issues, dtype=bool)
    return flags


# --- caps --------------------------------------------------------------------


def cap_corrections(
    current: np.ndarray,
    candidates: np.ndarray,
    probs: np.ndarray,
    *,
    band: Tuple[float, float],
    max_corrections: Optional[int] = None,
) -> np.ndarray:
    """Accept candidate flips, most confident first, while prevalence stays in band.

    ``current`` are the existing (possibly soft) labels, ``candidates`` the CL
    flags, ``probs`` the teacher probabilities. Returns a boolean acceptance mask
    aligned with ``candidates``.

    This is the single-label band only. The joint-prevalence band across the
    correlated pairs is a second hard constraint applied by ``apply_corrections``
    through ``enforce_joint_bands``, which can revoke flips accepted here.
    """
    n = len(current)
    accepted = np.zeros(n, dtype=bool)
    if n == 0 or max_corrections == 0:
        return accepted

    binary = (current >= 0.5).astype(int)
    positives = int(binary.sum())
    low, high = band
    lower_count = low * n
    upper_count = high * n

    order = sorted(
        (index for index in range(n) if candidates[index]),
        key=lambda index: (-abs(float(probs[index]) - 0.5), index),
    )
    for index in order:
        if max_corrections is not None and int(accepted.sum()) >= max_corrections:
            break
        delta = 1 if binary[index] == 0 else -1
        candidate_positives = positives + delta
        if candidate_positives < lower_count - 1e-9 or candidate_positives > upper_count + 1e-9:
            continue
        accepted[index] = True
        positives = candidate_positives
    return accepted


# --- joint prevalence bands --------------------------------------------------


def _pair_key(first: str, second: str) -> str:
    return f"{first} & {second}"


def _binary(frame: pd.DataFrame, label: str) -> np.ndarray:
    return pd.to_numeric(frame[label], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def baseline_cooccurrence(
    labels_df: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[str, str]] = CORRELATED_PAIRS,
    threshold: float = 0.5,
) -> Dict[Tuple[str, str], float]:
    """Pre-round joint-positive rate for each correlated pair, at ``threshold``.

    This is the anchor for the hard band enforced by ``enforce_joint_bands``: a
    round may move a pair's joint prevalence, but not by more than
    ``JOINT_BAND_RELATIVE`` of where it started.
    """
    n = len(labels_df)
    rates: Dict[Tuple[str, str], float] = {}
    for first, second in pairs:
        if first not in labels_df.columns or second not in labels_df.columns:
            continue
        if not n:
            rates[(first, second)] = 0.0
            continue
        joint = (_binary(labels_df, first) >= threshold) & (
            _binary(labels_df, second) >= threshold
        )
        rates[(first, second)] = float(joint.mean())
    return rates


def joint_band(baseline: float, *, relative: float = JOINT_BAND_RELATIVE) -> Tuple[float, float]:
    """Allowed post-correction joint prevalence window: +/-``relative`` of baseline."""
    return baseline * (1.0 - relative), baseline * (1.0 + relative)


def enforce_joint_bands(
    original: Dict[str, np.ndarray],
    updated: Dict[str, np.ndarray],
    accepted: Dict[str, np.ndarray],
    probs: Dict[str, np.ndarray],
    baseline: Dict[Tuple[str, str], float],
    *,
    pairs: Sequence[Tuple[str, str]] = CORRELATED_PAIRS,
    relative: float = JOINT_BAND_RELATIVE,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Revoke accepted flips until every pair's joint prevalence is back in band.

    Mutates ``updated`` and ``accepted`` in place. Flips are revoked least
    confident first, so the most confident corrections survive right up to the
    band edge. Returns ``(rows, rejections_by_pair)``.
    """
    rows: List[Dict[str, object]] = []
    rejections: Dict[str, int] = {}
    for first, second in pairs:
        if first not in updated or second not in updated:
            continue
        key = _pair_key(first, second)
        base = float(baseline.get((first, second), 0.0))
        low, high = joint_band(base, relative=relative)
        n = len(updated[first])
        revoked = 0
        while n:
            joint = int(
                np.sum((updated[first] >= 0.5) & (updated[second] >= 0.5))
            )
            rate = joint / n
            if low - 1e-9 <= rate <= high + 1e-9:
                break
            need_decrease = rate > high
            best: Optional[Tuple[float, str, int]] = None
            for label, partner in ((first, second), (second, first)):
                partner_positive = updated[partner] >= 0.5
                for index in np.flatnonzero(accepted[label]):
                    before = bool(updated[label][index] >= 0.5) and bool(
                        partner_positive[index]
                    )
                    after = bool(original[label][index] >= 0.5) and bool(
                        partner_positive[index]
                    )
                    change = int(after) - int(before)
                    if need_decrease and change >= 0:
                        continue
                    if not need_decrease and change <= 0:
                        continue
                    confidence = abs(float(probs[label][index]) - 0.5)
                    if best is None or (confidence, label, int(index)) < best:
                        best = (confidence, label, int(index))
            if best is None:
                break
            _, label, index = best
            updated[label][index] = original[label][index]
            accepted[label][index] = False
            revoked += 1

        joint = int(np.sum((updated[first] >= 0.5) & (updated[second] >= 0.5))) if n else 0
        rejections[key] = revoked
        rows.append(
            {
                "pair": [first, second],
                "baseline_joint_prevalence": base,
                "band_low": low,
                "band_high": high,
                "joint_prevalence_after": (joint / n) if n else 0.0,
                "rejections": revoked,
                "within_band": bool(
                    n == 0 or (low - 1e-9 <= joint / n <= high + 1e-9)
                ),
            }
        )
    return rows, rejections


# --- co-drift ----------------------------------------------------------------


def codrift_report(
    labels_frame: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[str, str]] = CORRELATED_PAIRS,
    ratio_threshold: float = CODRIFT_RATIO_THRESHOLD,
    min_joint: int = CODRIFT_MIN_JOINT,
) -> List[Dict[str, object]]:
    """Joint correction counts for the correlated pairs vs their chance rate."""
    n = len(labels_frame)
    rows: List[Dict[str, object]] = []
    for first, second in pairs:
        if first not in accepted.columns or second not in accepted.columns:
            continue
        flags_first = accepted[first].to_numpy(dtype=bool)
        flags_second = accepted[second].to_numpy(dtype=bool)
        joint = int(np.sum(flags_first & flags_second))
        expected = float(flags_first.sum()) * float(flags_second.sum()) / n if n else 0.0
        ratio = joint / expected if expected > 0 else (float("inf") if joint else 0.0)
        baseline = (
            float(
                np.mean(
                    (pd.to_numeric(labels_frame[first], errors="coerce").fillna(0.0) >= 0.5)
                    & (pd.to_numeric(labels_frame[second], errors="coerce").fillna(0.0) >= 0.5)
                )
            )
            if n
            else 0.0
        )
        flagged = joint >= min_joint and ratio >= ratio_threshold
        rows.append(
            {
                "pair": [first, second],
                "corrections_a": int(flags_first.sum()),
                "corrections_b": int(flags_second.sum()),
                "joint_corrections": joint,
                "expected_joint": expected,
                "ratio": ratio,
                "baseline_cooccurrence": baseline,
                "flagged": flagged,
            }
        )
    return rows


# --- orchestration -----------------------------------------------------------


def apply_corrections(
    labels_frame: pd.DataFrame,
    probs_frame: pd.DataFrame,
    flags: pd.DataFrame,
    *,
    label_weights: Dict[str, float],
    thresholds: Dict[str, Tuple[float, str]],
    labels: Sequence[str] = TARGET_LABELS,
    max_corrections: Optional[int] = None,
    protected_ids: Optional[Set[str]] = None,
) -> CorrectionResult:
    """Apply capped, band-constrained, co-drift-checked corrections.

    ``flags`` is the candidate mask (from ``find_candidates`` or supplied
    directly). ``protected_ids`` - the gold studies - are never corrected.

    Three layers, in order: the per-label prevalence cap (``cap_corrections``),
    the hard joint-prevalence band on the correlated pairs
    (``enforce_joint_bands``, +/-``JOINT_BAND_RELATIVE`` of the pre-round
    baseline), and finally the co-drift pause, which reverts a pair outright when
    its corrections move together far more than chance.
    """
    frame = labels_frame.copy()
    frame[STUDY_ID_COLUMN] = frame[STUDY_ID_COLUMN].astype(str)
    protected = {str(value) for value in (protected_ids or set())}
    protected_mask = frame[STUDY_ID_COLUMN].isin(protected).to_numpy()

    accepted_frame = pd.DataFrame({STUDY_ID_COLUMN: frame[STUDY_ID_COLUMN]})
    per_label: Dict[str, LabelCorrection] = {}
    n = len(frame)

    original_values: Dict[str, np.ndarray] = {}
    updated_values: Dict[str, np.ndarray] = {}
    accepted_masks: Dict[str, np.ndarray] = {}
    prob_values: Dict[str, np.ndarray] = {}

    for label in labels:
        current = pd.to_numeric(frame[label], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        probs = (
            pd.to_numeric(probs_frame[label + PRED_SUFFIX], errors="coerce")
            .fillna(0.5)
            .to_numpy(dtype=float)
        )
        candidates = flags[label].to_numpy(dtype=bool) if label in flags.columns else np.zeros(n, bool)
        candidates = candidates & ~protected_mask

        auc = float(label_weights.get(label, 0.5))
        positives_before = int((current >= 0.5).sum())
        prevalence_before = positives_before / n if n else 0.0
        band = prevalence_band(prevalence_before, auc)
        accepted = cap_corrections(
            current, candidates, probs, band=band, max_corrections=max_corrections
        )

        updated = current.copy()
        updated[accepted] = np.where(current[accepted] >= 0.5, 0.0, 1.0)

        original_values[label] = current
        updated_values[label] = updated
        accepted_masks[label] = accepted
        prob_values[label] = probs

        positives_after = int((updated >= 0.5).sum())
        per_label[label] = LabelCorrection(
            label=label,
            auc=auc,
            threshold=thresholds.get(label, (0.5, "unset"))[0],
            threshold_source=thresholds.get(label, (0.5, "unset"))[1],
            candidates=int(candidates.sum()),
            accepted=int(accepted.sum()),
            capped=int(candidates.sum()) - int(accepted.sum()),
            positives_before=positives_before,
            positives_after=positives_after,
            prevalence_before=prevalence_before,
            prevalence_after=positives_after / n if n else 0.0,
            band_low=band[0],
            band_high=band[1],
        )

    # Layer two: the hard joint-prevalence band on the correlated pairs. This may
    # revoke flips the per-label cap accepted, least confident first.
    baseline = baseline_cooccurrence(labels_frame)
    joint_bands, joint_band_rejections = enforce_joint_bands(
        original_values, updated_values, accepted_masks, prob_values, baseline
    )

    for label in labels:
        updated = updated_values[label]
        accepted = accepted_masks[label]
        frame[label] = updated
        accepted_frame[label] = accepted
        item = per_label[label]
        positives_after = int((updated >= 0.5).sum())
        per_label[label] = replace(
            item,
            accepted=int(accepted.sum()),
            capped=item.candidates - int(accepted.sum()),
            positives_after=positives_after,
            prevalence_after=positives_after / n if n else 0.0,
        )

    codrift = codrift_report(labels_frame, accepted_frame)
    paused = [list(row["pair"]) for row in codrift if row["flagged"]]

    # A paused pair keeps its previous labels: the round does not move it.
    for pair in paused:
        for label in pair:
            frame[label] = pd.to_numeric(labels_frame[label], errors="coerce").fillna(0.0).to_numpy()
            accepted_frame[label] = False
            item = per_label[label]
            per_label[label] = replace(
                item,
                accepted=0,
                capped=item.candidates,
                positives_after=item.positives_before,
                prevalence_after=item.prevalence_before,
                paused=True,
            )

    return CorrectionResult(
        labels=frame,
        per_label=per_label,
        codrift=codrift,
        paused_pairs=paused,
        n_studies=n,
        n_skipped=int(protected_mask.sum()),
        flags=accepted_frame,
        joint_bands=joint_bands,
        joint_band_rejections=joint_band_rejections,
    )


def run_corrections(
    labels_frame: pd.DataFrame,
    probs_frame: pd.DataFrame,
    *,
    label_weights: Dict[str, float],
    gold_positive_counts: Dict[str, int],
    labels: Sequence[str] = TARGET_LABELS,
    max_corrections: Optional[int] = None,
    protected_ids: Optional[Set[str]] = None,
    flags: Optional[pd.DataFrame] = None,
) -> CorrectionResult:
    """Full per-label correction pass: thresholds, candidates, caps, co-drift."""
    aligned_labels, aligned_probs = align_frames(labels_frame, probs_frame, labels=labels)
    thresholds = derive_thresholds(
        aligned_labels,
        aligned_probs,
        label_weights=label_weights,
        gold_positive_counts=gold_positive_counts,
        labels=labels,
    )
    if flags is None:
        if max_corrections == 0:
            # Identity round: no flip can be accepted, so skip Confident Learning
            # entirely and keep the dry run free of the cleanlab dependency.
            flags = pd.DataFrame({STUDY_ID_COLUMN: aligned_labels[STUDY_ID_COLUMN]})
            for label in labels:
                flags[label] = False
        else:
            flags = find_candidates(
                aligned_labels, aligned_probs, labels=labels, thresholds=thresholds
            )
    return apply_corrections(
        aligned_labels,
        aligned_probs,
        flags,
        label_weights=label_weights,
        thresholds=thresholds,
        labels=labels,
        max_corrections=max_corrections,
        protected_ids=protected_ids,
    )
