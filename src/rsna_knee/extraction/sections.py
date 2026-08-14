from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .language import strip_accents

# Header vocabularies across all corpus languages (accent-stripped, lowercase).
_IMPRESSION = (
    r"impression|impresion|impressao|conclusion(?:i|e|s|es)?|conclusao|beurteilung|"
    r"sonuc|kanaat|заключение|мнение|misljenje|zakljucak|besluit|συμπερασμα|εντυπωση|"
    r"γνωμη|kanı"
)
_FINDINGS = (
    r"findings|hallazgos|resultados|achados|reperti|befund(?:e|ung)?|bulgular|"
    r"resultats|nalaz|bevindingen|ευρηματα|данни|описание|opis"
)
_TECHNIQUE = (
    r"technique|tecnica|technik|teknik|protocol[oe]?|scanprotocol|tetkik protokolu|"
    r"metodika|метод"
)
# Clinical question / indication lines: label terms here are questions, not findings.
_INDICATION = (
    r"indication|indicacion|indicazione|klinik|clinical (?:history|information)|"
    r"diagnostische vraagstelling|vraagstelling|anamnese|anamnez|historia clinica|"
    r"quesito|renseignements? cliniques?|klinische angaben|endikasyon|история"
)

_HEADER_RE = re.compile(
    r"(?P<header>" + "|".join(
        f"(?P<{name}>{pat})"
        for name, pat in [
            ("impression", _IMPRESSION),
            ("findings", _FINDINGS),
            ("technique", _TECHNIQUE),
            ("indication", _INDICATION),
        ]
    ) + r")\s*:",
    re.IGNORECASE,
)

_EXCLUDED_SECTIONS = ("technique", "indication")


def split_sections(text: str) -> Dict[str, str]:
    """Split a report into technique/indication/findings/impression/other.

    Text before the first recognized header goes to 'other'. Reports with no
    headers at all put the entire text in 'findings'.
    """
    normalized = strip_accents(text)
    matches = list(_HEADER_RE.finditer(normalized))
    if not matches:
        return {"findings": text}
    sections: Dict[str, List[str]] = {}
    if matches[0].start() > 0:
        sections.setdefault("other", []).append(text[: matches[0].start()])
    for i, m in enumerate(matches):
        name = next(
            n for n in ("impression", "findings", "technique", "indication")
            if m.group(n)
        )
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.setdefault(name, []).append(text[m.end(): end])
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def matchable_sections(text: str) -> List[Tuple[str, str]]:
    """(section_name, content) pairs eligible for label matching."""
    return [
        (name, content)
        for name, content in split_sections(text).items()
        if name not in _EXCLUDED_SECTIONS and content
    ]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.;\n]+", text)
    return [p.strip() for p in parts if p.strip()]
