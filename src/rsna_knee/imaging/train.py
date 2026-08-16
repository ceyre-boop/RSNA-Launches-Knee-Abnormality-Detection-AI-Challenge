"""Single-fold training CLI for the imaging model.

Runs on CUDA (Kaggle) with AMP, and on CPU/MPS for smoke tests via ``--limit``.
Requires the optional ``train`` dependency group.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..constants import STUDY_ID_COLUMN, TARGET_LABELS
from ..metrics import macro_auc
from .dataset import SeriesVolumeLoader, StudySlotDataset, collate_slots
from .labels import load_label_table
from .model import DEFAULT_BACKBONE, DEFAULT_UNFROZEN_BLOCKS, KneeSlotModel, parameter_groups
from .slots import build_slot_table


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted BCE: per-study weight scales that study's whole row."""
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    scaled = per_element.mean(dim=1) * weights
    denominator = weights.sum().clamp_min(1e-8)
    return scaled.sum() / denominator


def score_predictions(
    truth: np.ndarray,
    preds: np.ndarray,
    targets: Sequence[str] = TARGET_LABELS,
) -> tuple[float, Dict[str, float]]:
    """Macro AUC over the targets that have both classes present."""
    y_true: Dict[str, List[int]] = {}
    y_score: Dict[str, List[float]] = {}
    usable = []
    for index, target in enumerate(targets):
        column = (truth[:, index] >= 0.5).astype(int)
        if column.min() == column.max():
            continue
        usable.append(target)
        y_true[target] = column.tolist()
        y_score[target] = preds[:, index].tolist()
    if not usable:
        return float("nan"), {}
    return macro_auc(y_true, y_score, usable)


def resolve_cache_dirs(values: Optional[Sequence[str]]) -> List[Path]:
    """Flatten repeated and comma-separated ``--cache-dirs`` values, order preserved."""
    dirs: List[Path] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and Path(part) not in dirs:
                dirs.append(Path(part))
    return dirs


def build_datasets(args: argparse.Namespace):
    series = pd.read_csv(args.series_csv)
    splits = pd.read_csv(args.splits_csv)
    splits[STUDY_ID_COLUMN] = splits[STUDY_ID_COLUMN].astype(str)
    labels = load_label_table(args.labels_csv, args.gold_csv)

    slot_table = build_slot_table(series)
    labelled = set(labels[STUDY_ID_COLUMN])
    with_slots = set(slot_table[STUDY_ID_COLUMN].astype(str))

    def studies_for(fold_filter) -> List[str]:
        subset = splits[fold_filter]
        ids = [sid for sid in subset[STUDY_ID_COLUMN] if sid in labelled and sid in with_slots]
        return ids[: args.limit] if args.limit else ids

    train_ids = studies_for(splits["fold"].astype(int) != args.fold)
    valid_ids = studies_for(splits["fold"].astype(int) == args.fold)

    loader = SeriesVolumeLoader(args.data_root, img_size=args.img_size, cache_dir=args.cache_dir)
    cache_dirs = resolve_cache_dirs(args.cache_dirs)
    if cache_dirs:
        print(f"preprocessed cache dirs: {[str(path) for path in cache_dirs]}")
    common = dict(
        img_size=args.img_size,
        seed=args.seed,
        cache_dirs=cache_dirs,
        cache_band_trimmed=not args.cache_not_band_trimmed,
    )
    train_set = StudySlotDataset(train_ids, labels, slot_table, loader, train=True, **common)
    valid_set = StudySlotDataset(valid_ids, labels, slot_table, loader, train=False, **common)
    return train_set, valid_set


def run_epoch(
    model: KneeSlotModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> tuple[float, np.ndarray, np.ndarray, List[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss, total_weight = 0.0, 0.0
    all_preds, all_truth, all_ids = [], [], []

    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weight"].to(device)

        with torch.set_grad_enabled(training):
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, mask)
                    loss = weighted_bce(logits.float(), labels, weights)
            else:
                logits = model(images, mask)
                loss = weighted_bce(logits, labels, weights)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_weight = float(weights.sum().item()) or 1.0
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
        all_preds.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_truth.append(labels.detach().cpu().numpy())
        all_ids.extend(batch["study_id"])

    mean_loss = total_loss / max(total_weight, 1e-8)
    if not all_preds:
        empty = np.zeros((0, len(TARGET_LABELS)), dtype=np.float32)
        return mean_loss, empty, empty, []
    return mean_loss, np.concatenate(all_preds), np.concatenate(all_truth), all_ids


def write_oof(path: Path, study_ids: Sequence[str], truth: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [STUDY_ID_COLUMN]
    for label in TARGET_LABELS:
        fieldnames.extend([f"{label}_true", f"{label}_pred"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, study_id in enumerate(study_ids):
            row = {STUDY_ID_COLUMN: study_id}
            for index, label in enumerate(TARGET_LABELS):
                row[f"{label}_true"] = f"{float(truth[row_index, index]):.6f}"
                row[f"{label}_pred"] = f"{float(preds[row_index, index]):.6f}"
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one fold of the RSNA knee imaging model")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--labels-csv", default="data/pseudo_labels_sonnet_v1.csv")
    parser.add_argument("--gold-csv", default="data/train.csv")
    parser.add_argument("--series-csv", default="data/train_series.csv")
    parser.add_argument("--splits-csv", default="data/splits_reporthash_5fold.csv")
    parser.add_argument("--data-root", default="data/train_images")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--cache-dirs",
        action="append",
        default=None,
        help="preprocessed per-study npz cache dir(s); comma-separated or repeatable",
    )
    parser.add_argument(
        "--cache-not-band-trimmed",
        action="store_true",
        help="cache volumes still carry the full stack (older float caches)",
    )
    parser.add_argument("--out", default="artifacts/imaging")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--unfrozen-blocks", type=int, default=DEFAULT_UNFROZEN_BLOCKS)
    parser.add_argument("--backbone-lr", type=float, default=8e-6)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0, help="cap studies per split (smoke test)")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_set, valid_set = build_datasets(args)
    print(f"device={device} train={len(train_set)} valid={len(valid_set)}")
    if len(train_set) == 0:
        raise SystemExit("No training studies after filtering")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_slots,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_slots,
    )

    model = KneeSlotModel(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
        unfrozen_blocks=args.unfrozen_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.backbone_lr, args.head_lr), weight_decay=1e-4
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"fold{args.fold}_best.pt"
    oof_path = out_dir / f"oof_fold{args.fold}.csv"
    best_auc = -1.0

    for epoch in range(args.epochs):
        train_loss, _, _, _ = run_epoch(
            model, train_loader, device, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        valid_loss, preds, truth, ids = run_epoch(model, valid_loader, device)
        macro, per_label = score_predictions(truth, preds)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} macro_auc={macro:.5f}"
        )
        if np.isfinite(macro) and macro > best_auc:
            best_auc = macro
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "macro_auc": macro, "args": vars(args)},
                checkpoint_path,
            )
            write_oof(oof_path, ids, truth, preds)
            for label, value in per_label.items():
                print(f"  {label}={value:.5f}")

    print(f"best_macro_auc={best_auc:.5f} checkpoint={checkpoint_path} oof={oof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
