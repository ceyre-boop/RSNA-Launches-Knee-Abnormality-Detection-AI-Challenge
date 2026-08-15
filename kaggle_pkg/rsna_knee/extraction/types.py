from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

STATUSES = ("affirmed", "negated", "uncertain", "not_mentioned")

# AUC is rank-based: bin ordering matters more than calibration.
# Exposed as module constant so gold-58 sweeps can patch it.
SCORE_MAP: Dict[str, float] = {
    "affirmed": 0.90,
    "uncertain": 0.60,
    "not_mentioned": 0.35,
    "negated": 0.05,
}
IMPRESSION_AFFIRMED_BONUS = 0.95  # affirmed in the impression section


@dataclass(frozen=True)
class LabelEvidence:
    score: float
    status: str
    evidence: str = ""


@dataclass
class ExtractionResult:
    study_uid: str
    engine: str  # "rules" | "llm"
    language: str  # ISO-ish code; "unk" allowed
    labels: Dict[str, LabelEvidence] = field(default_factory=dict)
    error: str = ""  # non-empty => engine failed for this study


class Extractor(Protocol):
    def extract(self, study_uid: str, report: str) -> ExtractionResult: ...


def score_for(status: str, in_impression: bool = False) -> float:
    if status == "affirmed" and in_impression:
        return IMPRESSION_AFFIRMED_BONUS
    return SCORE_MAP[status]
