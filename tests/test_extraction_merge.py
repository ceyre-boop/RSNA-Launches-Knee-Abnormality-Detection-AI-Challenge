from rsna_knee.constants import TARGET_LABELS
from rsna_knee.extraction.merge import merge_results
from rsna_knee.extraction.types import ExtractionResult, LabelEvidence, score_for


def make_result(uid, engine, statuses=None, error=""):
    labels = {
        label: LabelEvidence(
            score=score_for((statuses or {}).get(label, "not_mentioned")),
            status=(statuses or {}).get(label, "not_mentioned"),
        )
        for label in TARGET_LABELS
    }
    return ExtractionResult(
        study_uid=uid, engine=engine, language="en",
        labels=labels if not error else {}, error=error,
    )


def test_llm_wins_when_present():
    rules = {"u1": make_result("u1", "rules", {"ACL": "negated"})}
    llm = {"u1": make_result("u1", "llm", {"ACL": "affirmed"})}
    merged, audit = merge_results(rules, llm)
    assert merged[0].engine == "llm"
    assert merged[0].labels["ACL"].status == "affirmed"
    assert len(audit) == 1
    assert audit[0]["label"] == "ACL"
    assert audit[0]["rules_status"] == "negated"


def test_rules_fallback_on_llm_error():
    rules = {"u1": make_result("u1", "rules", {"ACL": "affirmed"})}
    llm = {"u1": make_result("u1", "llm", error="refusal")}
    merged, audit = merge_results(rules, llm)
    assert merged[0].engine == "rules"
    assert audit == []


def test_no_audit_rows_on_agreement():
    rules = {"u1": make_result("u1", "rules", {"ACL": "affirmed"})}
    llm = {"u1": make_result("u1", "llm", {"ACL": "affirmed"})}
    _, audit = merge_results(rules, llm)
    assert audit == []
