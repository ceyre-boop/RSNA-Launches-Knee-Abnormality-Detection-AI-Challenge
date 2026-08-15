"""Convert subagent extraction result files into pseudo-label rows.

Subagent workers (Claude Code subscription path) write JSON files mapping
uid -> {language, labels: {label: {status, evidence}}}. This module validates
them against the canonical status vocabulary and produces ExtractionResults
compatible with the rest of the pipeline (merge, eval, write_pseudo_labels).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from ..constants import TARGET_LABELS
from .types import STATUSES, ExtractionResult, LabelEvidence, score_for


def load_agent_results(paths: Iterable[str | Path], engine: str = "llm") -> Dict[str, ExtractionResult]:
    results: Dict[str, ExtractionResult] = {}
    errors: List[str] = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        for uid, entry in data.items():
            labels: Dict[str, LabelEvidence] = {}
            ok = True
            for label in TARGET_LABELS:
                item = (entry.get("labels") or {}).get(label)
                if not item or item.get("status") not in STATUSES:
                    errors.append(f"{uid}:{label}")
                    ok = False
                    break
                labels[label] = LabelEvidence(
                    score=score_for(item["status"]),
                    status=item["status"],
                    evidence=item.get("evidence", ""),
                )
            if ok:
                results[uid] = ExtractionResult(
                    study_uid=uid, engine=engine,
                    language=entry.get("language", "unk"), labels=labels,
                )
            else:
                results[uid] = ExtractionResult(
                    study_uid=uid, engine=engine, language="unk",
                    labels={}, error=f"invalid: {errors[-1]}",
                )
    return results
