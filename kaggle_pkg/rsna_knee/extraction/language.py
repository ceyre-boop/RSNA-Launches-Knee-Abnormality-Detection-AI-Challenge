from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

# Stage 1: script detection uniquely identifies these in this corpus.
_SCRIPT_RANGES = {
    "bg": re.compile(r"[Ѐ-ӿ]"),  # Cyrillic
    "el": re.compile(r"[Ͱ-Ͽ]"),  # Greek
    "th": re.compile(r"[฀-๿]"),  # Thai
}

# Stage 2: distinctive marker words per Latin-script language.
# Section headers are the most reliable signal in radiology reports.
_MARKERS: Dict[str, List[str]] = {
    "en": ["the", "findings", "impression", "there is", "no evidence", "knee", "with"],
    "es": ["hallazgos", "impresion", "tecnica", "rodilla", "derrame", "sin", "rotura"],
    "tr": ["bulgular", "sonuc", "izlenmektedir", "menisküs", "menikus", "saptanmadi", "diz", "yirtik", "duzeyinde"],
    "it": ["reperti", "conclusioni", "versamento", "ginocchio", "menisco mediale", "non si"],
    "pt": ["achados", "impressao", "joelho", "nao", "ruptura do", "conclusao"],
    "de": ["befund", "beurteilung", "kein", "erguss", "kniegelenk", "vorderes", "riss"],
    "fr": ["resultats", "conclusion", "epanchement", "menisque", "pas de", "genou"],
    "hr": ["nalaz", "misljenje", "koljena", "prednja", "ukrstena", "krizni", "izljev"],
    "nl": ["bevindingen", "besluit", "knie", "geen", "voorste", "scheur", "meniscusscheur"],
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def detect_language(report: str) -> str:
    if not report:
        return "unk"
    alpha = [c for c in report if c.isalpha()]
    if alpha:
        for lang, pattern in _SCRIPT_RANGES.items():
            hits = len(pattern.findall(report))
            if hits / len(alpha) > 0.30:
                return lang
    normalized = " " + strip_accents(report.lower()) + " "
    scores = {}
    for lang, markers in _MARKERS.items():
        scores[lang] = sum(
            1
            for m in markers
            if re.search(r"(?<![a-z])" + re.escape(strip_accents(m)) + r"(?![a-z])", normalized)
        )
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    if scores[best] == 0 or (len(ranked) > 1 and ranked[0] == ranked[1]):
        return "unk"
    return best
