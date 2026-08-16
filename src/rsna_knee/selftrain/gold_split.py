"""Deterministic split of the gold studies into working-40 and locked-18.

The gold studies are the rows of ``data/train.csv`` where every one of the twelve
target labels is present. The split is stratified on per-label positive counts,
seeded with the pre-committed seed 2026, written once to
``data/gold_split_v1.json`` and never regenerated: the locked slice is spent a
single time, at the end of the campaign, by ``referee.final_check``.

CLI::

    python -m rsna_knee.selftrain.gold_split
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS

# Pre-committed constants. Changing any of these invalidates the campaign.
GOLD_SPLIT_SEED = 2026
LOCKED_SIZE = 18
WORKING_SIZE = 40
SPLIT_VERSION = "v1"

DEFAULT_TRAIN_CSV = "data/train.csv"
DEFAULT_SPLIT_PATH = "data/gold_split_v1.json"


class SplitExistsError(RuntimeError):
    """Raised when a split file already exists; splits are written once."""


@dataclass(frozen=True)
class GoldSplit:
    """A materialised gold split. ``locked`` must never be scored mid-campaign."""

    version: str
    seed: int
    working: List[str]
    locked: List[str]

    @property
    def locked_set(self) -> Set[str]:
        return set(self.locked)

    @property
    def working_set(self) -> Set[str]:
        return set(self.working)


def gold_studies(
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    *,
    targets: Sequence[str] = TARGET_LABELS,
) -> pd.DataFrame:
    """Rows of ``train_csv`` with every target label present, sorted by study id."""
    frame = pd.read_csv(train_csv)
    missing = [column for column in [STUDY_ID_COLUMN, *targets] if column not in frame.columns]
    if missing:
        raise ValueError(f"train_csv missing columns: {missing}")
    table = frame[[STUDY_ID_COLUMN, *targets]].copy()
    table[STUDY_ID_COLUMN] = table[STUDY_ID_COLUMN].astype(str)
    for target in targets:
        table[target] = pd.to_numeric(table[target], errors="coerce")
    table = table.dropna(subset=list(targets))
    table = table.drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last")
    return table.sort_values(STUDY_ID_COLUMN).reset_index(drop=True)


def _penalty(counts: Dict[str, float], targets: Dict[str, float], size: int, capacity: int) -> float:
    """Squared deviation from pro-rata targets, weighted toward the rare labels.

    A one-study miss on Fracture (18 positives) costs far more than a one-study
    miss on Effusion (35), which is what keeps the thin labels representable in
    both slices.
    """
    share = size / capacity if capacity else 0.0
    return sum(
        ((counts[label] - target * share) ** 2) / max(target, 1e-6)
        for label, target in targets.items()
    )


def stratified_split(
    table: pd.DataFrame,
    *,
    targets: Sequence[str] = TARGET_LABELS,
    seed: int = GOLD_SPLIT_SEED,
    locked_size: int = LOCKED_SIZE,
) -> GoldSplit:
    """Greedy stratified allocation: each study goes where it hurts balance least.

    Studies are visited in a seeded shuffle of the id-sorted gold table, so the
    result is a pure function of (table contents, seed, locked_size).
    """
    # Sorting here (not just in gold_studies) makes the result independent of the
    # row order the caller happened to hand us.
    table = table.copy()
    table[STUDY_ID_COLUMN] = table[STUDY_ID_COLUMN].astype(str)
    table = table.sort_values(STUDY_ID_COLUMN).reset_index(drop=True)

    study_ids = [str(value) for value in table[STUDY_ID_COLUMN]]
    total = len(study_ids)
    if locked_size >= total:
        raise ValueError(f"locked_size {locked_size} must be smaller than {total} gold studies")
    working_size = total - locked_size

    positives = {
        str(study): {label: float(row[label] >= 0.5) for label in targets}
        for study, row in zip(study_ids, table.to_dict("records"))
    }
    label_totals = {
        label: sum(positives[study][label] for study in study_ids) for label in targets
    }
    locked_targets = {
        label: value * locked_size / total for label, value in label_totals.items()
    }
    working_targets = {
        label: value * working_size / total for label, value in label_totals.items()
    }

    order = list(study_ids)
    random.Random(seed).shuffle(order)

    locked: List[str] = []
    working: List[str] = []
    locked_counts = {label: 0.0 for label in targets}
    working_counts = {label: 0.0 for label in targets}

    for study in order:
        row = positives[study]
        if len(locked) >= locked_size:
            choose_locked = False
        elif len(working) >= working_size:
            choose_locked = True
        else:
            trial_locked = {label: locked_counts[label] + row[label] for label in targets}
            trial_working = {label: working_counts[label] + row[label] for label in targets}
            cost_locked = _penalty(trial_locked, locked_targets, len(locked) + 1, locked_size)
            cost_locked += _penalty(working_counts, working_targets, len(working), working_size)
            cost_working = _penalty(locked_counts, locked_targets, len(locked), locked_size)
            cost_working += _penalty(trial_working, working_targets, len(working) + 1, working_size)
            choose_locked = cost_locked < cost_working
        if choose_locked:
            locked.append(study)
            for label in targets:
                locked_counts[label] += row[label]
        else:
            working.append(study)
            for label in targets:
                working_counts[label] += row[label]

    return GoldSplit(
        version=SPLIT_VERSION,
        seed=seed,
        working=sorted(working),
        locked=sorted(locked),
    )


def split_payload(
    split: GoldSplit,
    table: pd.DataFrame,
    *,
    targets: Sequence[str] = TARGET_LABELS,
) -> Dict[str, object]:
    """JSON body for the split file. Deliberately free of timestamps."""
    indexed = table.set_index(STUDY_ID_COLUMN)

    def counts(ids: Sequence[str]) -> Dict[str, int]:
        subset = indexed.loc[[i for i in ids if i in indexed.index]]
        return {label: int((subset[label] >= 0.5).sum()) for label in targets}

    return {
        "version": split.version,
        "seed": split.seed,
        "n_gold": len(split.working) + len(split.locked),
        "n_working": len(split.working),
        "n_locked": len(split.locked),
        "working": list(split.working),
        "locked": list(split.locked),
        "working_positive_counts": counts(split.working),
        "locked_positive_counts": counts(split.locked),
    }


def write_split(
    path: str | Path,
    payload: Dict[str, object],
    *,
    overwrite: bool = False,
) -> Path:
    """Write the split file. Refuses to clobber an existing split."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise SplitExistsError(
            f"{output} already exists; the gold split is written once and committed"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def load_split(path: str | Path = DEFAULT_SPLIT_PATH) -> GoldSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldSplit(
        version=str(payload["version"]),
        seed=int(payload["seed"]),
        working=[str(value) for value in payload["working"]],
        locked=[str(value) for value in payload["locked"]],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the gold working/locked split once")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--seed", type=int, default=GOLD_SPLIT_SEED)
    parser.add_argument("--locked-size", type=int, default=LOCKED_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="danger: regenerating a committed split invalidates the locked slice",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    table = gold_studies(args.train_csv)
    split = stratified_split(table, seed=args.seed, locked_size=args.locked_size)
    payload = split_payload(split, table)
    output = write_split(args.out, payload, overwrite=args.overwrite)
    print(f"gold studies: {payload['n_gold']}")
    print(f"working: {payload['n_working']} -> {payload['working_positive_counts']}")
    print(f"locked:  {payload['n_locked']} -> {payload['locked_positive_counts']}")
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
