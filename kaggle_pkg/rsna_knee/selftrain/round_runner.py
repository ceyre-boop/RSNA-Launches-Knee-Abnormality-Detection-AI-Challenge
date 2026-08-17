"""One local self-training round: teacher OOF in, corrected label set out.

Covers memo steps 2-4 and 6: read the teacher's full-corpus predictions, correct
the pseudo labels with caps and co-drift guards, write ``labels_r{N}.csv`` plus a
correction report, and optionally run the M4 collapse triage before any GPU is
spent on the student. Student training itself stays on Kaggle.

CLI::

    python -m rsna_knee.selftrain.round_runner \\
        --round 1 \\
        --teacher-oof artifacts/fold0/teacher_full.csv \\
        --labels-csv data/pseudo_labels_sonnet_v1_3.csv \\
        --out-dir artifacts/selftrain/r1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS
from . import corrections as corrections_module
from .corrections import (
    DEFAULT_LABEL_WEIGHTS,
    CorrectionResult,
    load_label_weights,
    run_corrections,
    working_positive_counts,
)
from .gold_split import DEFAULT_SPLIT_PATH, DEFAULT_TRAIN_CSV, gold_studies, load_split

DEFAULT_CACHE_DIR = "data/cache224"
TRIAGE_STUDIES = 20
TRIAGE_MIN_VARIANCE = 0.01


class CollapseError(RuntimeError):
    """Raised when a student checkpoint produces near-constant predictions."""


def load_teacher_oof(path: str | Path, *, labels: Sequence[str] = TARGET_LABELS) -> pd.DataFrame:
    """Teacher predictions over the corpus: study id plus ``{label}_pred`` columns."""
    frame = pd.read_csv(path)
    if STUDY_ID_COLUMN not in frame.columns:
        raise ValueError(f"teacher OOF missing {STUDY_ID_COLUMN}")
    frame[STUDY_ID_COLUMN] = frame[STUDY_ID_COLUMN].astype(str)
    keep = [STUDY_ID_COLUMN] + [
        label + corrections_module.PRED_SUFFIX
        for label in labels
        if label + corrections_module.PRED_SUFFIX in frame.columns
    ]
    missing = [
        label
        for label in labels
        if label + corrections_module.PRED_SUFFIX not in frame.columns
    ]
    if missing:
        raise ValueError(f"teacher OOF missing prediction columns for: {missing}")
    return frame[keep].drop_duplicates(subset=[STUDY_ID_COLUMN], keep="last").reset_index(drop=True)


def run_round(
    round_index: int,
    teacher_oof: str | Path,
    labels_csv: str | Path,
    out_dir: str | Path,
    *,
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    label_weights_path: str | Path = DEFAULT_LABEL_WEIGHTS,
    max_corrections: Optional[int] = None,
    labels: Sequence[str] = TARGET_LABELS,
) -> CorrectionResult:
    """Produce ``labels_r{N}.csv`` and ``corrections_report.json`` under ``out_dir``."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    label_frame = pd.read_csv(labels_csv)
    label_frame[STUDY_ID_COLUMN] = label_frame[STUDY_ID_COLUMN].astype(str)
    probs = load_teacher_oof(teacher_oof, labels=labels)

    gold = gold_studies(train_csv, targets=labels)
    split = load_split(split_path)
    weights = load_label_weights(label_weights_path, labels=labels)
    counts = working_positive_counts(gold, split.working, labels=labels)

    protected = set(gold[STUDY_ID_COLUMN].astype(str))
    result = run_corrections(
        label_frame,
        probs,
        label_weights=weights,
        gold_positive_counts=counts,
        labels=labels,
        max_corrections=max_corrections,
        protected_ids=protected,
    )

    labels_path = output / f"labels_r{round_index}.csv"
    # run_corrections carries the input frame's other columns (engine, grades, ...)
    # through untouched; restore the original column order for a clean diff.
    written = result.labels
    ordered = [column for column in label_frame.columns if column in written.columns]
    ordered += [column for column in written.columns if column not in ordered]
    written[ordered].to_csv(labels_path, index=False)

    report = result.report()
    report["round"] = round_index
    report["teacher_oof"] = str(teacher_oof)
    report["labels_csv_in"] = str(labels_csv)
    report["labels_csv_out"] = str(labels_path)
    report["split"] = str(split_path)
    report["max_corrections"] = max_corrections
    report["gold_working_positive_counts"] = counts
    (output / "corrections_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return result


def format_summary(round_index: int, result: CorrectionResult) -> str:
    lines = [
        f"round {round_index}: {result.n_studies} studies, {result.n_skipped} gold studies protected",
        f"{'label':<18} {'auc':>6} {'thr':>6} {'src':>15} {'cand':>5} {'acc':>5} {'cap':>5} {'prev':>7} -> {'prev':>7}",
    ]
    for label, item in result.per_label.items():
        lines.append(
            f"{label:<18} {item.auc:>6.3f} {item.threshold:>6.3f} {item.threshold_source:>15} "
            f"{item.candidates:>5} {item.accepted:>5} {item.capped:>5} "
            f"{item.prevalence_before:>7.4f} -> {item.prevalence_after:>7.4f}"
        )
    lines.append("co-drift:")
    for row in result.codrift:
        first, second = row["pair"]
        lines.append(
            f"  {first} + {second}: joint {row['joint_corrections']} vs expected "
            f"{row['expected_joint']:.2f} (ratio {row['ratio']:.2f}, "
            f"baseline co-occurrence {row['baseline_cooccurrence']:.4f})"
            + ("  FLAGGED - pair paused for manual review" if row["flagged"] else "")
        )
    if result.paused_pairs:
        lines.append(f"paused pairs: {result.paused_pairs}")
    return "\n".join(lines)


# --- M4 triage ---------------------------------------------------------------


def _load_cached_study(path: Path, *, max_slots: int = 4) -> "np.ndarray":
    """Build a [S, 3, H, W] slot stack from one cache224 ``.npz`` study file."""
    with np.load(path) as archive:
        series = sorted(archive.files)[:max_slots]
        slots = []
        for key in series:
            volume = np.asarray(archive[key], dtype=np.float32)
            if volume.ndim != 3 or volume.shape[0] == 0:
                continue
            middle = volume.shape[0] // 2
            indices = [max(middle - 1, 0), middle, min(middle + 1, volume.shape[0] - 1)]
            slots.append(volume[indices])
    if not slots:
        raise ValueError(f"no usable series in {path}")
    return np.stack(slots)


def triage_checkpoint(
    checkpoint_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    n_studies: int = TRIAGE_STUDIES,
    min_variance: float = TRIAGE_MIN_VARIANCE,
    device: str = "cpu",
) -> Dict[str, object]:
    """Cheap collapse check on M4: forward ``n_studies`` cached studies, measure spread.

    Raises ``CollapseError`` when the per-label prediction variance is at or below
    ``min_variance`` - the student has collapsed to a constant and is not worth a
    Kaggle GPU slot. Requires the optional ``train`` dependency group.
    """
    try:
        import torch  # type: ignore
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "triage_checkpoint needs the optional 'train' dependency group (torch, timm)"
        ) from error

    from ..imaging.model import KneeSlotModel  # imported late: pulls in timm

    payload = torch.load(str(checkpoint_path), map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model_kwargs: Dict[str, object] = {
        "pretrained": False,
        "img_size": int(saved_args.get("img_size", 224)),
    }
    if saved_args.get("backbone"):
        model_kwargs["backbone"] = str(saved_args["backbone"])
    model = KneeSlotModel(**model_kwargs)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)

    files = sorted(Path(cache_dir).glob("*.npz"))[:n_studies]
    if not files:
        raise ValueError(f"no cached studies under {cache_dir}")

    rows: List[np.ndarray] = []
    used: List[str] = []
    with torch.no_grad():
        for path in files:
            try:
                slots = _load_cached_study(path)
            except ValueError:
                continue
            images = torch.from_numpy(slots).unsqueeze(0).to(device)
            mask = torch.ones(1, images.shape[1], device=images.device)
            logits = model(images, mask)
            rows.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
            used.append(path.stem)

    if not rows:
        raise ValueError(f"no usable cached studies under {cache_dir}")

    predictions = np.stack(rows)
    per_label_variance = predictions.var(axis=0)
    overall = float(per_label_variance.max())
    result = {
        "n_studies": len(used),
        "studies": used,
        "per_label_variance": {
            label: float(value) for label, value in zip(TARGET_LABELS, per_label_variance)
        },
        "max_variance": overall,
        "min_variance_required": min_variance,
        "passed": overall > min_variance,
    }
    if not result["passed"]:
        raise CollapseError(
            f"prediction variance {overall:.5f} <= {min_variance}; student looks collapsed"
        )
    return result


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local self-training round")
    parser.add_argument("--round", type=int, required=True, dest="round_index")
    parser.add_argument("--teacher-oof", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--split", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--label-weights", default=DEFAULT_LABEL_WEIGHTS)
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=None,
        help="per-label cap on accepted flips; 0 runs the identity round",
    )
    parser.add_argument("--triage-checkpoint", default=None)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_round(
        args.round_index,
        args.teacher_oof,
        args.labels_csv,
        args.out_dir,
        train_csv=args.train_csv,
        split_path=args.split,
        label_weights_path=args.label_weights,
        max_corrections=args.max_corrections,
    )
    print(format_summary(args.round_index, result))
    if args.triage_checkpoint:
        triage = triage_checkpoint(args.triage_checkpoint, args.cache_dir)
        print(f"triage: max variance {triage['max_variance']:.5f} over {triage['n_studies']} studies")
    print(f"wrote {Path(args.out_dir) / f'labels_r{args.round_index}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
