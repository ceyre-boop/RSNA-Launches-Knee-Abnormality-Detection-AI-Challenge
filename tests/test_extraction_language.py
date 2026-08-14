from rsna_knee.extraction.language import detect_language


def test_script_detection():
    assert detect_language("Ставен излив. Медиален мениск без руптура.") == "bg"
    assert detect_language("Αρθρικό υγρό. Έσω μηνίσκος χωρίς ρήξη.") == "el"
    assert detect_language("ข้อเข่ามีน้ำในข้อ") == "th"


def test_latin_marker_vote():
    assert detect_language("Findings: There is a tear of the ACL. Impression: ACL tear.") == "en"
    assert detect_language("Hallazgos: rotura del menisco interno. Sin derrame. Técnica: RMN rodilla.") == "es"
    assert detect_language("Bulgular: Ön çapraz bağ normaldir. Menisküs saptanmadı. Sonuç: diz normal.") == "tr"
    assert detect_language("Bevindingen: geen meniscusscheur. Besluit: normale knie.") == "nl"
    assert detect_language("Befund: Kein Erguss. Beurteilung: Kniegelenk unauffällig, kein Riss.") == "de"


def test_unknown_fallback():
    assert detect_language("") == "unk"
    assert detect_language("xyzzy 12345 qqq") == "unk"
