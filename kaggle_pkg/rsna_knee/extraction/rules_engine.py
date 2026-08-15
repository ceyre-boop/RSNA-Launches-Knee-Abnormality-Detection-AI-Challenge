from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..constants import TARGET_LABELS
from .language import detect_language, strip_accents
from .lexicon import (
    NEGATION_CUES,
    PATHOLOGY_CUES,
    SELF_EVIDENT_LABELS,
    UNCERTAINTY_CUES,
    compiled_terms,
    cues_for,
)
from .sections import matchable_sections, split_sentences
from .types import ExtractionResult, LabelEvidence, score_for


def _sentence_status(
    label: str,
    sentence_norm: str,
    negation: List[re.Pattern],
    uncertainty: List[re.Pattern],
    pathology: List[re.Pattern],
) -> Optional[str]:
    """Status contributed by one sentence that mentions the label, or None."""
    # Precedence: uncertainty > negation > affirmed. Uncertainty first because
    # cues like "cannot exclude" / "no se puede descartar" contain negation words.
    if any(p.search(sentence_norm) for p in uncertainty):
        return "uncertain"
    if any(p.search(sentence_norm) for p in negation):
        return "negated"
    if label in SELF_EVIDENT_LABELS:
        return "affirmed"
    # Structural labels need a pathology word; a bare anatomical mention is neutral.
    if any(p.search(sentence_norm) for p in pathology):
        return "affirmed"
    return None


def _aggregate(statuses: List[str]) -> str:
    for status in ("affirmed", "uncertain", "negated"):
        if status in statuses:
            return status
    return "not_mentioned"


class RulesExtractor:
    engine = "rules"

    def extract(self, study_uid: str, report: str) -> ExtractionResult:
        report = report if isinstance(report, str) else ""
        lang = detect_language(report)
        terms = compiled_terms(lang)
        negation = cues_for(lang, NEGATION_CUES)
        uncertainty = cues_for(lang, UNCERTAINTY_CUES)
        pathology = cues_for(lang, PATHOLOGY_CUES)
        term_res: Dict[str, List[re.Pattern]] = {}
        for t in terms:
            term_res.setdefault(t.label, []).append(re.compile(t.pattern, re.IGNORECASE))

        # (label, section) -> list of sentence statuses, plus evidence snippets
        per_section: Dict[str, Dict[str, List[str]]] = {}
        evidence: Dict[str, str] = {}
        for section_name, content in matchable_sections(report):
            for sentence in split_sentences(content):
                sentence_norm = strip_accents(sentence.lower())
                for label, patterns in term_res.items():
                    if not any(p.search(sentence_norm) for p in patterns):
                        continue
                    status = _sentence_status(
                        label, sentence_norm, negation, uncertainty, pathology
                    )
                    if status is None:
                        continue
                    per_section.setdefault(label, {}).setdefault(
                        section_name, []
                    ).append(status)
                    if status == "affirmed" or label not in evidence:
                        evidence[label] = sentence.strip()[:160]

        labels: Dict[str, LabelEvidence] = {}
        for label in TARGET_LABELS:
            sections = per_section.get(label, {})
            impression = _aggregate(sections.get("impression", []))
            findings_statuses: List[str] = []
            for name, statuses in sections.items():
                if name != "impression":
                    findings_statuses.extend(statuses)
            findings = _aggregate(findings_statuses)
            if impression != "not_mentioned":
                status, in_impression = impression, True
            else:
                status, in_impression = findings, False
            labels[label] = LabelEvidence(
                score=score_for(status, in_impression),
                status=status,
                evidence=evidence.get(label, ""),
            )
        return ExtractionResult(
            study_uid=study_uid, engine=self.engine, language=lang, labels=labels
        )
