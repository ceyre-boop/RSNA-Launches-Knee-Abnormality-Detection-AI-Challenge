import csv
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.splits import build_cv_splits, validate_no_group_leakage  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_patient_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            train_csv = temp_path / "train.csv"
            split_csv = temp_path / "split.csv"

            with train_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["StudyInstanceUID", "PatientID", "SiteID", "Language"],
                )
                writer.writeheader()
                writer.writerow({"StudyInstanceUID": "s1", "PatientID": "p1", "SiteID": "A", "Language": "en"})
                writer.writerow({"StudyInstanceUID": "s2", "PatientID": "p1", "SiteID": "A", "Language": "en"})
                writer.writerow({"StudyInstanceUID": "s3", "PatientID": "p2", "SiteID": "B", "Language": "fr"})

            build_cv_splits(train_csv, split_csv, n_folds=3, seed=7)
            self.assertTrue(validate_no_group_leakage(split_csv, train_csv))


if __name__ == "__main__":
    unittest.main()

