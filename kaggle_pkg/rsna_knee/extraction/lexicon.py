from __future__ import annotations

import re
from typing import Dict, List, NamedTuple

# All patterns are matched against accent-stripped, lowercased text
# (see language.strip_accents). Cyrillic/Greek text is lowercased only.


class Term(NamedTuple):
    label: str
    lang: str  # "*" = language-independent (Latin abbreviations etc.)
    pattern: str


TERMS: List[Term] = [
    # ---- ACL ----
    Term("ACL", "*", r"\bacl\b|\blca\b(?! externo)|\bocb\b|\bvkb\b"),
    Term("ACL", "en", r"anterior cruciate"),
    Term("ACL", "es", r"(ligamento )?cruzado anterior"),
    Term("ACL", "tr", r"on capraz bag"),
    Term("ACL", "it", r"crociato anteriore"),
    Term("ACL", "pt", r"cruzado anterior"),
    Term("ACL", "de", r"vordere[sn]? kreuzband"),
    Term("ACL", "fr", r"(ligament )?croise anterieur"),
    Term("ACL", "hr", r"prednj(?:a|e|i|eg)\s+(?:ukrsten|kriz)\w*"),
    Term("ACL", "nl", r"voorste kruisband"),
    Term("ACL", "el", r"προσθι\w+ χιαστ\w+"),
    Term("ACL", "bg", r"предна\w* кръстн\w* връзк\w*|пкв"),
    # ---- MCL ----
    Term("MCL", "*", r"\bmcl\b|\bmkl\b|\blli\b|\blcm\b"),
    Term("MCL", "en", r"medial collateral"),
    Term("MCL", "es", r"(ligamento )?(colateral|lateral) (medial|interno)"),
    Term("MCL", "tr", r"ic yan bag|medial kollateral"),
    Term("MCL", "it", r"collaterale mediale"),
    Term("MCL", "pt", r"colateral medial"),
    Term("MCL", "de", r"innenband|mediale[sn]? (seiten|kollateral)band"),
    Term("MCL", "fr", r"(ligament )?collateral (medial|interne)"),
    Term("MCL", "hr", r"medijaln\w+ kolateraln\w+"),
    Term("MCL", "nl", r"mediale collaterale band|binnenband"),
    Term("MCL", "el", r"εσω πλαγι\w+ συνδεσμ\w+"),
    Term("MCL", "bg", r"медиалн\w* колатералн\w*|вътрешн\w* колатералн\w*"),
    # ---- Medial Meniscus ----
    Term("Medial Meniscus", "en", r"medial meniscus"),
    Term("Medial Meniscus", "es", r"menisco (interno|medial)"),
    Term("Medial Meniscus", "tr", r"(ic|medial) menisk?us"),
    Term("Medial Meniscus", "it", r"menisco (interno|mediale)"),
    Term("Medial Meniscus", "pt", r"menisco (interno|medial)"),
    Term("Medial Meniscus", "de", r"innenmeniskus|mediale[rn]? meniskus"),
    Term("Medial Meniscus", "fr", r"menisque (interne|medial)"),
    Term("Medial Meniscus", "hr", r"medijaln\w+ menisk\w*|unutarnj\w+ menisk\w*"),
    Term("Medial Meniscus", "nl", r"mediale meniscus|binnenmeniscus"),
    Term("Medial Meniscus", "el", r"εσω μηνισκ\w+"),
    Term("Medial Meniscus", "bg", r"медиалн\w* мениск\w*|вътрешн\w* мениск\w*"),
    # ---- Lateral Meniscus ----
    Term("Lateral Meniscus", "en", r"lateral meniscus"),
    Term("Lateral Meniscus", "es", r"menisco (externo|lateral)"),
    Term("Lateral Meniscus", "tr", r"(dis|lateral) menisk?us"),
    Term("Lateral Meniscus", "it", r"menisco (esterno|laterale)"),
    Term("Lateral Meniscus", "pt", r"menisco (externo|lateral)"),
    Term("Lateral Meniscus", "de", r"aussenmeniskus|laterale[rn]? meniskus"),
    Term("Lateral Meniscus", "fr", r"menisque (externe|lateral)"),
    Term("Lateral Meniscus", "hr", r"lateraln\w+ menisk\w*|vanjsk\w+ menisk\w*"),
    Term("Lateral Meniscus", "nl", r"laterale meniscus|buitenmeniscus"),
    Term("Lateral Meniscus", "el", r"εξω μηνισκ\w+"),
    Term("Lateral Meniscus", "bg", r"латералн\w* мениск\w*|външн\w* мениск\w*"),
    # ---- Medial OA (medial femorotibial compartment) ----
    Term("Medial OA", "en", r"medial (compartment|femorotibial|tibiofemoral)\s*\w* (osteoarthr|arthr|chondropath|chondral|cartilage)\w*|(osteoarthritis|chondropathy|chondral \w+|cartilage (loss|thinning|defect))[^.;\n]{0,40}medial (compartment|femoral condyle|tibial plateau)|medial femoral condyle[^.;\n]{0,40}(chondral|cartilage|osteophyt)"),
    Term("Medial OA", "es", r"artrosis femorotibial (medial|interna)|(condropatia|condral)[^.;\n]{0,40}(condilo femoral (medial|interno)|platillo tibial (medial|interno)|compartimento (medial|interno))|gonartrosis[^.;\n]{0,30}(medial|interna)"),
    Term("Medial OA", "tr", r"(ic|medial) kompartman\w*[^.;\n]{0,40}(gonartroz|artroz|kondropati|kondral)|(medial|ic) femoral kondil\w*[^.;\n]{0,40}(kondropati|kondral|osteofit)"),
    Term("Medial OA", "it", r"(gonartrosi|artrosi|condropatia)[^.;\n]{0,40}(femoro-?tibiale )?(mediale|interno)|condilo femorale mediale[^.;\n]{0,40}(condropatia|condrale)"),
    Term("Medial OA", "de", r"mediale\w* (gonarthrose|femorotibial\w* arthrose|chondropathie|knorpelschaden)|knorpelschaden[^.;\n]{0,40}medial"),
    Term("Medial OA", "fr", r"arthrose femoro-?tibiale (interne|mediale)|chondropathie[^.;\n]{0,40}(condyle femoral interne|compartiment interne|mediale?)"),
    Term("Medial OA", "*", r"medial\w*[^.;\n]{0,30}(osteoarthr\w+|gonar(?:th)?ro[sz]\w*)|(osteoarthr\w+|gonar(?:th)?ro[sz]\w*)[^.;\n]{0,30}medial"),
    # ---- Lateral OA ----
    Term("Lateral OA", "en", r"lateral (compartment|femorotibial|tibiofemoral)\s*\w* (osteoarthr|arthr|chondropath|chondral|cartilage)\w*|(osteoarthritis|chondropathy|chondral \w+|cartilage (loss|thinning|defect))[^.;\n]{0,40}lateral (compartment|femoral condyle|tibial plateau)|lateral femoral condyle[^.;\n]{0,40}(chondral|cartilage|osteophyt)"),
    Term("Lateral OA", "es", r"artrosis femorotibial (lateral|externa)|(condropatia|condral)[^.;\n]{0,40}(condilo femoral (lateral|externo)|platillo tibial (lateral|externo)|compartimento (lateral|externo))"),
    Term("Lateral OA", "tr", r"(dis|lateral) kompartman\w*[^.;\n]{0,40}(gonartroz|artroz|kondropati|kondral)|lateral femoral kondil\w*[^.;\n]{0,40}(kondropati|kondral|osteofit)"),
    Term("Lateral OA", "it", r"(gonartrosi|artrosi|condropatia)[^.;\n]{0,40}(femoro-?tibiale )?(laterale|esterno)|condilo femorale laterale[^.;\n]{0,40}(condropatia|condrale)"),
    Term("Lateral OA", "de", r"laterale\w* (gonarthrose|femorotibial\w* arthrose|chondropathie|knorpelschaden)|knorpelschaden[^.;\n]{0,40}lateral"),
    Term("Lateral OA", "fr", r"arthrose femoro-?tibiale (externe|laterale)|chondropathie[^.;\n]{0,40}(condyle femoral externe|compartiment externe)"),
    Term("Lateral OA", "*", r"lateral\w*[^.;\n]{0,30}osteoarthr\w+|osteoarthr\w+[^.;\n]{0,30}lateral"),
    # ---- PF OA (patellofemoral) ----
    Term("PF OA", "*", r"(chondromalaci\w+|kondromalazi\w*|condromalaci\w+)"),
    Term("PF OA", "en", r"patellofemoral[^.;\n]{0,40}(osteoarthr|arthr|chondropath|chondral|cartilage)\w*|(chondral \w+|cartilage (loss|thinning|defect)|chondropathy)[^.;\n]{0,40}(patell\w+|trochle\w+)|retropatellar[^.;\n]{0,30}(chondropath|cartilage|arthrosis)\w*"),
    Term("PF OA", "es", r"(artrosis|condropatia|condral)[^.;\n]{0,40}(femoropatelar|rotuliana?|patelar|troclea\w*)|femoropatelar[^.;\n]{0,40}(artrosis|condropatia)"),
    Term("PF OA", "tr", r"patellofemoral[^.;\n]{0,40}(artroz|kondropati|kondral)|(patella\w*|troklea\w*)[^.;\n]{0,40}(kondropati|kondral)"),
    Term("PF OA", "it", r"(artrosi|condropatia)[^.;\n]{0,40}(femoro-?rotulea|rotulea?|troclea\w*)"),
    Term("PF OA", "de", r"retropatellar\w*[^.;\n]{0,40}(arthrose|chondropathie|knorpel)|femoropatellar\w*[^.;\n]{0,40}arthrose"),
    Term("PF OA", "fr", r"(arthrose|chondropathie)[^.;\n]{0,40}(femoro-?patellaire|rotulienne)"),
    # ---- Effusion ----
    Term("Effusion", "en", r"(joint )?effusion"),
    Term("Effusion", "es", r"derrame( articular)?"),
    Term("Effusion", "tr", r"efuzyon|(eklem(?:de|inde)?[^.;\n]{0,30})?(sivi|mayii?) (artisi|artimi)|artmis eklem sivisi"),
    Term("Effusion", "it", r"versamento( articolare)?"),
    Term("Effusion", "pt", r"derrame( articular)?"),
    Term("Effusion", "de", r"erguss|gelenkerguss"),
    Term("Effusion", "fr", r"epanchement( articulaire| intra-?articulaire)?"),
    Term("Effusion", "hr", r"izljev|izliv"),
    Term("Effusion", "nl", r"hydrops|gewrichtsvocht|vochtcollectie"),
    Term("Effusion", "el", r"υγρο[^.;\n]{0,20}αρθρωση|αρθρικ\w+ υγρ\w+|συλλογη υγρου"),
    Term("Effusion", "bg", r"излив|ставен излив"),
    # ---- Synovitis ----
    Term("Synovitis", "*", r"s[iy]novit\w*|συνοβιτιδα|синовит\w*|synovial (thickening|proliferation|hypertroph\w+)"),
    # ---- Baker's ----
    Term("Baker's", "*", r"baker|popliteal cyst|quiste popliteo|kyste poplite|cisto popliteo|poplitealzyste|бейкър|киста на бейкър"),
    # ---- Contusion ----
    Term("Contusion", "en", r"(bone |osseous )?contusion|(bone )?marrow (o?edema|edema)|trabecular (micro)?fracture|bone bruise"),
    Term("Contusion", "es", r"contusion (osea)?|edema oseo|edema de medula osea"),
    Term("Contusion", "tr", r"kontuzyon|kemik iligi odemi"),
    Term("Contusion", "it", r"contusione|edema (della )?spongiosa|edema osseo"),
    Term("Contusion", "pt", r"contusao|edema osseo"),
    Term("Contusion", "de", r"kontusion|knochenmark(s)?odem|bone bruise"),
    Term("Contusion", "fr", r"contusion( osseuse)?|oedeme osseux|oedeme medullaire"),
    Term("Contusion", "hr", r"kontuzij\w+|edem kostane srzi"),
    Term("Contusion", "nl", r"contusie|beenmergoedeem|botcontusie"),
    Term("Contusion", "el", r"οστικο οιδημα|θλαση"),
    Term("Contusion", "bg", r"контузи\w+|костномозъчен о?ток|оток на костния мозък"),
    # ---- Fracture ----
    Term("Fracture", "en", r"fracture"),
    Term("Fracture", "es", r"fractura"),
    Term("Fracture", "tr", r"fraktur|kirik( hatti)?"),
    Term("Fracture", "it", r"frattura"),
    Term("Fracture", "pt", r"fratura"),
    Term("Fracture", "de", r"fraktur|bruch(?!band)"),
    Term("Fracture", "fr", r"fracture"),
    Term("Fracture", "hr", r"prijelom|fraktur\w+"),
    Term("Fracture", "nl", r"fractuur|breuk"),
    Term("Fracture", "el", r"καταγμα"),
    Term("Fracture", "bg", r"фрактура|счупване"),
]

# Pathology words that flip a structure mention into an affirmed finding.
# A bare anatomical mention with none of these nearby is neutral, not affirmed.
PATHOLOGY_CUES: Dict[str, List[str]] = {
    "en": [r"tear", r"torn", r"rupture", r"sprain", r"injur\w+", r"lesion", r"degenerat\w+",
           r"maceration", r"extrusion", r"grade (ii|iii|2|3)", r"partial", r"complete",
           r"discontinuity", r"disrupt\w+", r"abnormal", r"o?edema", r"thicken\w+"],
    "es": [r"rotura", r"ruptura", r"desgarro", r"lesion", r"degenerat\w+", r"fisura",
           r"extrusion", r"grado (ii|iii|2|3)", r"parcial", r"completa", r"esguince",
           r"afectacion", r"engrosamiento", r"edema"],
    "tr": [r"yirtik\w*", r"yirtilma", r"rupture?", r"ruptur\w*", r"lezyon", r"dejenerasyon",
           r"dejenere", r"hasar", r"grade? (2|3|ii|iii)", r"parsiyel", r"tam kat",
           r"odem\w*", r"kalinlasma", r"ekstruzyon"],
    "it": [r"lesione", r"rottura", r"lacerazione", r"degenerat\w+", r"fissurazione",
           r"estrusione", r"edema", r"ispessimento"],
    "pt": [r"ruptura", r"rotura", r"lesao", r"degenera\w+", r"fissura", r"extrusao",
           r"edema", r"espessamento"],
    "de": [r"riss", r"ruptur", r"lasion", r"laesion", r"degenerat\w+", r"verletzung",
           r"odem", r"verdickung", r"extrusion", r"zerrung"],
    "fr": [r"dechirure", r"rupture", r"lesion", r"fissure", r"degenerat\w+", r"entorse",
           r"oedeme", r"epaississement", r"extrusion"],
    "hr": [r"ruptur\w+", r"lezij\w+", r"ostecen\w+", r"degenerativn\w+", r"fisur\w+",
           r"edem\w*"],
    "nl": [r"scheur", r"ruptuur", r"laesie", r"degenerat\w+", r"letsel", r"oedeem",
           r"verdikking"],
    "el": [r"ρηξη", r"ρικνωση", r"εκφυλιση", r"εκφυλιστικ\w+", r"βλαβη", r"οιδημα",
           r"ρωγμη"],
    "bg": [r"руптура", r"разкъсване", r"лезия", r"увреждане", r"дегенерат\w+", r"оток",
           r"фисура"],
}

# Labels whose terms already name pathology; a match alone is affirmative.
SELF_EVIDENT_LABELS = {
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    "Medial OA", "Lateral OA", "PF OA",
}

NEGATION_CUES: Dict[str, List[str]] = {
    "en": [r"\bno\b", r"\bnot\b", r"no evidence", r"without", r"\bintact\b", r"unremarkable",
           r"\bnormal\b", r"negative for", r"is preserved", r"are preserved", r"absen(t|ce)"],
    "es": [r"\bsin\b", r"sin evidencia", r"no se (observa|identifica|aprecia|evidencia|objetiva)\w*",
           r"dentro de (los )?limites normales", r"conservad\w+", r"integr\w+", r"normal\w*",
           r"\bno\b", r"ausencia de", r"indemne\w*"],
    "tr": [r"izlenmedi", r"izlenmemektedir", r"izlenmemistir", r"saptanmadi", r"saptanmamistir",
           r"gorulmedi", r"gozlenmedi", r"mevcut degil\w*", r"dogal\w*", r"normal\w*",
           r"intakt\w*", r"butunlugu korunmus\w*", r"tabii\w*", r"olagan\w*", r"yoktur", r"yok\b"],
    "it": [r"non si (osserva|apprezza|evidenzia|rileva)\w*", r"\bnon\b", r"assenza di",
           r"nei limiti( della norma)?", r"regolar\w+", r"conservat\w+", r"integr\w+",
           r"normal\w+", r"indenne"],
    "pt": [r"\bsem\b", r"\bnao\b", r"ausencia de", r"dentro dos limites", r"preservad\w+",
           r"integr\w+", r"normal\w*", r"indene"],
    "de": [r"\bkein\w*", r"\bohne\b", r"unauffallig\w*", r"\bintakt\w*", r"regelrecht\w*",
           r"\bnormal\w*", r"erhalten\w*", r"nicht (nachweisbar|abgrenzbar)", r"ausschluss"],
    "fr": [r"pas d", r"\bsans\b", r"absence d", r"\bnormal\w*", r"\bintact\w*",
           r"\bintegre\w*", r"respect\w+", r"dans les limites"],
    "hr": [r"\bbez\b", r"nema\b", r"ne vidi se", r"uredn\w+", r"normaln\w+", r"ocuvan\w+",
           r"intaktan|intaktn\w+"],
    "nl": [r"\bgeen\b", r"\bzonder\b", r"\bintact\w*", r"\bnormaal\w*", r"\bnormale\b",
           r"niet aangetoond", r"binnen de norm"],
    "el": [r"χωρις", r"δεν (παρατηρειται|διαπιστωνεται|αναδεικνυεται|υπαρχει)",
           r"φυσιολογικ\w+", r"ακεραι\w+", r"ανευ"],
    "bg": [r"\bбез\b", r"\bне се (установява|визуализира|открива)\w*", r"липс(а|ва)\w*",
           r"нормал\w+", r"запазен\w*", r"интактн\w*", r"\bняма\b"],
}

UNCERTAINTY_CUES: Dict[str, List[str]] = {
    "en": [r"cannot (be )?exclude\w*", r"possibl\w+", r"suspicious", r"suspect\w*",
           r"may represent", r"\bvs\b", r"equivocal", r"question\w*", r"borderline",
           r"probable", r"likely"],
    "es": [r"no impresiona", r"no se puede descartar", r"posible\w*", r"sugestiv\w+",
           r"dudos\w+", r"sospech\w+", r"probable\w*", r"a descartar"],
    "tr": [r"olabilir", r"kuskulu", r"supheli", r"dusundurmektedir", r"dusundur\w+",
           r"ayirt edilemem\w+", r"acisindan (kuskulu|supheli)", r"\?"],
    "it": [r"non si puo escludere", r"possibile", r"sospett\w+", r"dubbi\w+",
           r"compatibile con", r"verosimil\w+"],
    "pt": [r"nao se pode excluir", r"possivel", r"suspeit\w+", r"duvidos\w+",
           r"provavel", r"sugestiv\w+"],
    "de": [r"nicht (sicher )?auszuschliessen", r"moglich\w*", r"verdacht\w*", r"fraglich\w*",
           r"\bdd\b", r"vereinbar mit"],
    "fr": [r"ne (peut|pouvant) etre exclu\w*", r"possible\w*", r"suspicion", r"douteu\w+",
           r"evocateur", r"probable"],
    "hr": [r"moguc\w+", r"suspektn\w+", r"sumnj\w+", r"ne moze se iskljuciti"],
    "nl": [r"kan niet worden uitgesloten", r"mogelijk\w*", r"verdacht", r"twijfelachtig",
           r"suspect\w*"],
    "el": [r"δεν (μπορει να )?αποκλει\w+", r"πιθαν\w+", r"υποπτ\w+", r"αμφιβολ\w+"],
    "bg": [r"не (може да )?се изключ\w+", r"възможн\w+", r"съмнени\w+", r"суспектн\w+",
           r"вероятн\w+"],
}


def compiled_terms(lang: str) -> List[Term]:
    """Terms applicable to a language; 'unk' gets everything."""
    if lang == "unk":
        return TERMS
    return [t for t in TERMS if t.lang in ("*", lang)]


def cues_for(lang: str, table: Dict[str, List[str]]) -> List[re.Pattern]:
    langs = list(table.keys()) if lang == "unk" else [lang]
    patterns: List[re.Pattern] = []
    for lg in langs:
        for cue in table.get(lg, []):
            patterns.append(re.compile(cue, re.IGNORECASE))
    return patterns
