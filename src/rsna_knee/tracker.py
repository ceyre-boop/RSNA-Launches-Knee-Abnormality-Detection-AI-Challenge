from __future__ import annotations

import csv
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Optional


TRACKER_HEADERS = [
    "run_id",
    "created_at_utc",
    "split",
    "model",
    "macro_auc",
    "public_lb_auc",
    "train_minutes",
    "cost_usd",
    "per_label_auc",
    "notes",
]


def append_run(
    output_csv: str | Path,
    *,
    run_id: str,
    split: str,
    model: str,
    macro_auc: Optional[float],
    public_lb_auc: Optional[float],
    train_minutes: Optional[float],
    cost_usd: Optional[float],
    per_label_auc: Optional[Dict[str, float]],
    notes: str = "",
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()

    row = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "model": model,
        "macro_auc": "" if macro_auc is None else f"{macro_auc:.8f}",
        "public_lb_auc": "" if public_lb_auc is None else f"{public_lb_auc:.8f}",
        "train_minutes": "" if train_minutes is None else f"{train_minutes:.2f}",
        "cost_usd": "" if cost_usd is None else f"{cost_usd:.2f}",
        "per_label_auc": "" if not per_label_auc else repr(per_label_auc),
        "notes": notes,
    }

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKER_HEADERS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

