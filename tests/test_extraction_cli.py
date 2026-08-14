import csv

from rsna_knee.cli import build_parser
from rsna_knee.constants import STUDY_ID_COLUMN, TARGET_LABELS


def make_train_csv(path):
    rows = [
        {
            STUDY_ID_COLUMN: "g1",
            "Report": "Findings: ACL tear. Moderate joint effusion. Impression: ACL rupture.",
            **{label: "" for label in TARGET_LABELS},
            "ACL": "1", "Effusion": "1", "MCL": "0", "Medial Meniscus": "0",
            "Lateral Meniscus": "0", "Medial OA": "0", "Lateral OA": "0", "PF OA": "0",
            "Synovitis": "0", "Baker's": "0", "Contusion": "0", "Fracture": "0",
        },
        {
            STUDY_ID_COLUMN: "g2",
            "Report": "Findings: The ACL is intact. No joint effusion. No Baker's cyst.",
            "ACL": "0", "Effusion": "0", "MCL": "0", "Medial Meniscus": "1",
            "Lateral Meniscus": "0", "Medial OA": "0", "Lateral OA": "0", "PF OA": "0",
            "Synovitis": "0", "Baker's": "0", "Contusion": "0", "Fracture": "0",
        },
        {
            STUDY_ID_COLUMN: "u1",
            "Report": "Hallazgos: Sin evidencia de rotura del LCA. Derrame articular.",
            **{label: "" for label in TARGET_LABELS},
        },
    ]
    fieldnames = [STUDY_ID_COLUMN, "Report", *TARGET_LABELS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_extract_and_eval_pipeline(tmp_path, capsys):
    train_csv = tmp_path / "train.csv"
    pseudo_csv = tmp_path / "pseudo.csv"
    tracker_csv = tmp_path / "runs.csv"
    make_train_csv(train_csv)

    parser = build_parser()
    args = parser.parse_args([
        "extract-labels", "--train-csv", str(train_csv),
        "--engine", "rules", "--output-csv", str(pseudo_csv),
    ])
    args.func(args)

    with open(pseudo_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0].keys() >= {STUDY_ID_COLUMN, *TARGET_LABELS, "engine", "language"}
    for row in rows:
        for label in TARGET_LABELS:
            assert 0.0 <= float(row[label]) <= 1.0
    by_uid = {r[STUDY_ID_COLUMN]: r for r in rows}
    assert float(by_uid["g1"]["ACL"]) > 0.8
    assert float(by_uid["g2"]["ACL"]) < 0.1
    assert float(by_uid["u1"]["ACL"]) < 0.1  # sin evidencia de rotura
    assert float(by_uid["u1"]["Effusion"]) > 0.8  # derrame articular

    args = parser.parse_args([
        "eval-extraction", "--pseudo-csv", str(pseudo_csv),
        "--train-csv", str(train_csv), "--tracker-csv", str(tracker_csv),
        "--model-name", "rules-test",
    ])
    args.func(args)
    out = capsys.readouterr().out
    assert "macro_auc" in out
    assert "n/a(single-class)" in out  # e.g. MCL is all-zero in gold rows

    with open(tracker_csv, newline="", encoding="utf-8") as fh:
        tracked = list(csv.DictReader(fh))
    assert tracked[0]["model"] == "rules-test"
