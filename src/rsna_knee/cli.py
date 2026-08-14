from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from .constants import TARGET_LABELS
from .metrics import macro_auc
from .splits import build_cv_splits, validate_no_group_leakage
from .submission import build_submission
from .tracker import append_run


def _read_oof(oof_csv: str) -> tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    y_true = {label: [] for label in TARGET_LABELS}
    y_pred = {label: [] for label in TARGET_LABELS}

    with Path(oof_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for label in TARGET_LABELS:
            expected_true = f"{label}_true"
            expected_pred = f"{label}_pred"
            missing = [col for col in [expected_true, expected_pred] if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"OOF file missing columns: {missing}")
        for row in reader:
            for label in TARGET_LABELS:
                y_true[label].append(int(float(row[f"{label}_true"])))
                y_pred[label].append(float(row[f"{label}_pred"]))
    return y_true, y_pred


def command_build_splits(args: argparse.Namespace) -> None:
    build_cv_splits(
        train_csv=args.train_csv,
        output_csv=args.output_csv,
        n_folds=args.n_folds,
        seed=args.seed,
        study_col=args.study_col,
        patient_col=args.patient_col,
        site_col=args.site_col,
        language_col=args.language_col,
    )
    no_leak = validate_no_group_leakage(
        split_csv=args.output_csv,
        train_csv=args.train_csv,
        study_col=args.study_col,
        patient_col=args.patient_col,
    )
    print(f"Wrote splits: {args.output_csv}")
    print(f"Patient leakage check: {'PASS' if no_leak else 'FAIL'}")


def command_score_oof(args: argparse.Namespace) -> None:
    y_true, y_pred = _read_oof(args.oof_csv)
    macro, per_label = macro_auc(y_true, y_pred, TARGET_LABELS)
    print(f"macro_auc={macro:.6f}")
    for label, value in per_label.items():
        print(f"{label}={value:.6f}")

    if args.tracker_csv:
        append_run(
            args.tracker_csv,
            run_id=args.run_id,
            split=args.split_name,
            model=args.model_name,
            macro_auc=macro,
            public_lb_auc=args.public_lb_auc,
            train_minutes=args.train_minutes,
            cost_usd=args.cost_usd,
            per_label_auc=per_label,
            notes=args.notes,
        )
        print(f"Updated tracker: {args.tracker_csv}")


def command_make_submission(args: argparse.Namespace) -> None:
    build_submission(args.ids_csv, args.preds_csv, args.output_csv)
    print(f"Wrote submission: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RSNA knee challenge execution scaffold")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("build-splits", help="Create deterministic grouped CV splits")
    split_parser.add_argument("--train-csv", required=True)
    split_parser.add_argument("--output-csv", required=True)
    split_parser.add_argument("--n-folds", type=int, default=5)
    split_parser.add_argument("--seed", type=int, default=2026)
    split_parser.add_argument("--study-col", default="StudyInstanceUID")
    split_parser.add_argument("--patient-col", default="PatientID")
    split_parser.add_argument("--site-col", default="SiteID")
    split_parser.add_argument("--language-col", default="Language")
    split_parser.set_defaults(func=command_build_splits)

    score_parser = subparsers.add_parser("score-oof", help="Compute macro-AUC from OOF predictions")
    score_parser.add_argument("--oof-csv", required=True)
    score_parser.add_argument("--tracker-csv")
    score_parser.add_argument("--run-id", default="manual-run")
    score_parser.add_argument("--split-name", default="cv")
    score_parser.add_argument("--model-name", default="model")
    score_parser.add_argument("--public-lb-auc", type=float)
    score_parser.add_argument("--train-minutes", type=float)
    score_parser.add_argument("--cost-usd", type=float)
    score_parser.add_argument("--notes", default="")
    score_parser.set_defaults(func=command_score_oof)

    submit_parser = subparsers.add_parser("make-submission", help="Assemble submission file")
    submit_parser.add_argument("--ids-csv", required=True)
    submit_parser.add_argument("--preds-csv", required=True)
    submit_parser.add_argument("--output-csv", required=True)
    submit_parser.set_defaults(func=command_make_submission)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

