from rsna_knee.extraction.rules_engine import RulesExtractor

E = RulesExtractor()


def status(report, label):
    return E.extract("uid", report).labels[label].status


def test_english_affirmed_negated_notmentioned():
    assert status("Findings: Complete tear of the ACL.", "ACL") == "affirmed"
    assert status("Findings: The ACL is intact. No joint effusion.", "ACL") == "negated"
    assert status("Findings: The ACL is intact. No joint effusion.", "Effusion") == "negated"
    assert status("Findings: Complete tear of the ACL.", "Baker's") == "not_mentioned"


def test_spanish_negation_postposed():
    assert status("Hallazgos: Rotura del ligamento cruzado anterior.", "ACL") == "affirmed"
    assert status("Hallazgos: Sin evidencia de rotura del ligamento cruzado anterior.", "ACL") == "negated"
    assert (
        status("Hallazgos: Menisco lateral de morfologia conservada, sin signos de rotura.", "Lateral Meniscus")
        == "negated"
    )


def test_spanish_uncertainty():
    assert (
        status("Hallazgos: Posible rotura del menisco interno, no se puede descartar.", "Medial Meniscus")
        == "uncertain"
    )


def test_turkish_suffixal_negation():
    assert status("Bulgular: On capraz bagda tam kat yirtik izlenmektedir.", "ACL") == "affirmed"
    assert status("Bulgular: On capraz bag intakt olup yirtik saptanmadi.", "ACL") == "negated"
    assert status("Bulgular: Lateral menisküs ve medial menisküs normaldir.", "Medial Meniscus") == "negated"
    assert status("Bulgular: Diz ekleminde minimal mayii artisi görüldü.", "Effusion") == "affirmed"


def test_impression_overrides_findings():
    report = (
        "Findings: No definite tear of the medial meniscus. "
        "Impression: Tear of the medial meniscus."
    )
    assert status(report, "Medial Meniscus") == "affirmed"
    result = E.extract("uid", report)
    assert result.labels["Medial Meniscus"].score > 0.9


def test_clinical_question_excluded():
    report = "Diagnostische vraagstelling: meniscusscheur? Bevindingen:"
    assert status(report, "Medial Meniscus") == "not_mentioned"
    assert status(report, "Lateral Meniscus") == "not_mentioned"


def test_bare_structural_mention_is_not_affirmed():
    assert status("Findings: The medial meniscus is visualized.", "Medial Meniscus") == "not_mentioned"


def test_self_evident_labels_affirm_on_mention():
    assert status("Findings: Moderate joint effusion. Baker's cyst present.", "Effusion") == "affirmed"
    assert status("Findings: Moderate joint effusion. Baker's cyst present.", "Baker's") == "affirmed"


def test_score_ordering():
    r = E.extract(
        "uid",
        "Findings: ACL tear. MCL is intact. Possible lateral meniscus tear.",
    )
    labels = r.labels
    assert labels["ACL"].score > labels["Lateral Meniscus"].score
    assert labels["Lateral Meniscus"].score > labels["Baker's"].score  # uncertain > not_mentioned
    assert labels["Baker's"].score > labels["MCL"].score  # not_mentioned > negated
