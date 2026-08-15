from __future__ import annotations

from typing import Dict, List, Tuple

from ..constants import TARGET_LABELS
from .types import ExtractionResult


def merge_results(
    rules: Dict[str, ExtractionResult],
    llm: Dict[str, ExtractionResult],
) -> Tuple[List[ExtractionResult], List[dict]]:
    """LLM wins when present and non-error; rules is the fallback.

    Returns (merged results, audit rows for per-label status disagreements).
    """
    merged: List[ExtractionResult] = []
    audit: List[dict] = []
    for uid, rules_result in rules.items():
        llm_result = llm.get(uid)
        use_llm = llm_result is not None and not llm_result.error
        chosen = llm_result if use_llm else rules_result
        merged.append(chosen)
        if not use_llm:
            continue
        for label in TARGET_LABELS:
            r_ev = rules_result.labels[label]
            l_ev = llm_result.labels[label]
            if r_ev.status != l_ev.status:
                audit.append(
                    {
                        "StudyInstanceUID": uid,
                        "label": label,
                        "language": llm_result.language or rules_result.language,
                        "rules_status": r_ev.status,
                        "llm_status": l_ev.status,
                        "rules_evidence": r_ev.evidence,
                        "llm_evidence": l_ev.evidence,
                    }
                )
    audit.sort(key=lambda row: (row["label"], row["StudyInstanceUID"]))
    return merged, audit
