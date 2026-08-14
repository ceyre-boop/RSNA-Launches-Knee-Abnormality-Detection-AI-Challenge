from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from .constants import STUDY_ID_COLUMN, TARGET_LABELS
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


def command_extract_labels(args: argparse.Namespace) -> None:
    from .extraction.io import gold_mask, read_train, write_pseudo_labels
    from .extraction.merge import merge_results
    from .extraction.rules_engine import RulesExtractor

    df = read_train(args.train_csv)
    if args.only_gold:
        df = df[gold_mask(df)]
    if args.limit:
        df = df.head(args.limit)
    items = dict(zip(df[args.study_col], df["Report"].astype(str)))

    rules = RulesExtractor()
    rules_results = {uid: rules.extract(uid, report) for uid, report in items.items()}

    if args.engine == "rules":
        results = list(rules_results.values())
    else:
        from .extraction.llm_engine import LLMExtractor, estimate_cost

        llm = LLMExtractor(cache_dir=args.cache_dir, force=args.force)
        pending = llm.uncached_uids(items)
        if pending:
            mean_chars = sum(len(items[u]) for u in pending) / len(pending)
            cost = estimate_cost(len(pending), mean_chars, batch=args.mode == "batch")
            print(f"{len(pending)} uncached reports; est. cost ${cost:.2f} ({args.mode})")
            if not args.yes:
                answer = input("Proceed with API calls? [y/N] ").strip().lower()
                if answer != "y":
                    print("Aborted before any API call.")
                    return
            if args.mode == "batch":
                batch_id = llm.submit_batch({u: items[u] for u in pending})
                print(f"Submitted batch {batch_id}; poll with collect-batch")
                return
            for i, uid in enumerate(pending, 1):
                llm.extract(uid, items[uid])
                if i % 10 == 0 or i == len(pending):
                    print(f"  live: {i}/{len(pending)}")
        llm_results = {
            uid: llm.result_from_cache(uid, llm.read_cache(uid) or {"error": "missing"})
            for uid in items
        }
        if args.engine == "llm":
            results = [
                llm_results[uid] if not llm_results[uid].error else rules_results[uid]
                for uid in items
            ]
        else:  # merged
            results, audit = merge_results(rules_results, llm_results)
            if audit:
                import csv as _csv

                with Path(args.audit_csv).open("w", newline="", encoding="utf-8") as fh:
                    writer = _csv.DictWriter(fh, fieldnames=list(audit[0].keys()))
                    writer.writeheader()
                    writer.writerows(audit)
                print(f"Wrote {len(audit)} disagreements to {args.audit_csv}")

    out = write_pseudo_labels(results, args.output_csv)
    print(f"Wrote {len(out)} rows to {args.output_csv}")
    print("Language counts:", out["language"].value_counts().head(12).to_dict())


def command_collect_batch(args: argparse.Namespace) -> None:
    from .extraction.llm_engine import LLMExtractor

    llm = LLMExtractor(cache_dir=args.cache_dir)
    batch = llm.client.messages.batches.retrieve(args.batch_id)
    print(f"Batch status: {batch.processing_status}")
    if batch.processing_status == "ended":
        written = llm.collect_batch(args.batch_id)
        print(f"Cached {written} results")


def command_eval_extraction(args: argparse.Namespace) -> None:
    from .extraction.io import gold_mask, read_train

    pseudo = {}
    with Path(args.pseudo_csv).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pseudo[row[STUDY_ID_COLUMN]] = row

    df = read_train(args.train_csv)
    gold = df[gold_mask(df)]
    per_label_auc: Dict[str, float] = {}
    lines = []
    aucs = []
    for label in TARGET_LABELS:
        y_true, y_score = [], []
        for _, row in gold.iterrows():
            uid = row[STUDY_ID_COLUMN]
            if uid in pseudo:
                y_true.append(int(row[label]))
                y_score.append(float(pseudo[uid][label]))
        n = len(y_true)
        correct = sum(
            1 for t, s in zip(y_true, y_score) if (s >= args.threshold) == bool(t)
        )
        acc = correct / n if n else float("nan")
        try:
            auc = macro_auc({label: y_true}, {label: y_score}, [label])[0]
            aucs.append(auc)
            per_label_auc[label] = auc
            auc_str = f"{auc:.4f}"
        except ValueError:
            auc_str = "n/a(single-class)"
        lines.append(f"  {label:<18} n={n:<4} acc={acc:.3f} auc={auc_str}")

    macro = sum(aucs) / len(aucs) if aucs else float("nan")
    print(f"gold-58 eval ({args.pseudo_csv}):")
    print("\n".join(lines))
    print(f"macro_auc (over {len(aucs)} labels with both classes) = {macro:.4f}")

    if args.tracker_csv:
        append_run(
            args.tracker_csv,
            run_id=args.run_id,
            split="gold58",
            model=args.model_name,
            macro_auc=macro,
            public_lb_auc=None,
            train_minutes=None,
            cost_usd=args.cost_usd,
            per_label_auc=per_label_auc,
            notes=f"acc@{args.threshold}",
        )
        print(f"Updated tracker: {args.tracker_csv}")


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

    extract_parser = subparsers.add_parser("extract-labels", help="Extract pseudo-labels from reports")
    extract_parser.add_argument("--train-csv", default="data/train.csv")
    extract_parser.add_argument("--engine", choices=["rules", "llm", "merged"], default="rules")
    extract_parser.add_argument("--output-csv", default="data/pseudo_labels.csv")
    extract_parser.add_argument("--cache-dir", default="data/llm_cache")
    extract_parser.add_argument("--audit-csv", default="data/extraction_audit.csv")
    extract_parser.add_argument("--mode", choices=["live", "batch"], default="live")
    extract_parser.add_argument("--limit", type=int)
    extract_parser.add_argument("--only-gold", action="store_true")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--yes", action="store_true")
    extract_parser.add_argument("--study-col", default=STUDY_ID_COLUMN)
    extract_parser.set_defaults(func=command_extract_labels)

    collect_parser = subparsers.add_parser("collect-batch", help="Collect an LLM extraction batch")
    collect_parser.add_argument("--batch-id", required=True)
    collect_parser.add_argument("--cache-dir", default="data/llm_cache")
    collect_parser.set_defaults(func=command_collect_batch)

    eval_parser = subparsers.add_parser("eval-extraction", help="Score pseudo-labels against gold-58")
    eval_parser.add_argument("--pseudo-csv", default="data/pseudo_labels.csv")
    eval_parser.add_argument("--train-csv", default="data/train.csv")
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.add_argument("--tracker-csv")
    eval_parser.add_argument("--run-id", default="extraction-eval")
    eval_parser.add_argument("--model-name", default="rules-v0")
    eval_parser.add_argument("--cost-usd", type=float)
    eval_parser.set_defaults(func=command_eval_extraction)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

