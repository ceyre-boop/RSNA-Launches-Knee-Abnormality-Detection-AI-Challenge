import csv
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import TARGET_LABELS  # noqa: E402
from rsna_knee.submission import build_submission  # noqa: E402


class SubmissionTests(unittest.TestCase):
    def test_build_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            ids_csv = temp_path / "ids.csv"
            preds_csv = temp_path / "preds.csv"
            out_csv = temp_path / "submission.csv"

            with ids_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID"])
                writer.writeheader()
                writer.writerow({"StudyInstanceUID": "a"})

            with preds_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID"] + TARGET_LABELS)
                writer.writeheader()
                writer.writerow({"StudyInstanceUID": "a", **{label: "0.5" for label in TARGET_LABELS}})

            build_submission(ids_csv, preds_csv, out_csv)

            with out_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["StudyInstanceUID"], "a")
                for label in TARGET_LABELS:
                    self.assertEqual(rows[0][label], "0.5")


if __name__ == "__main__":
    unittest.main()

