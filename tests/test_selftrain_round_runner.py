import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import STUDY_ID_COLUMN, TARGET_LABELS  # noqa: E402
from rsna_knee.selftrain import gold_split, round_runner  # noqa: E402

N_CORPUS = 120
N_GOLD = 58


def build_fixture(workdir: pathlib.Path) -> dict:
    """A miniature corpus: labels csv, teacher OOF, gold train.csv, split, weights."""
    study_ids = [f"s-{index:04d}" for index in range(N_CORPUS)]

    labels = {STUDY_ID_COLUMN: study_ids, "engine": ["sonnet"] * N_CORPUS}
    for offset, label in enumerate(TARGET_LABELS):
        labels[label] = [1.0 if (index + offset) % 4 == 0 else 0.0 for index in range(N_CORPUS)]
    labels_frame = pd.DataFrame(labels)
    labels_csv = workdir / "labels_in.csv"
    labels_frame.to_csv(labels_csv, index=False)

    oof = {STUDY_ID_COLUMN: study_ids}
    for label in TARGET_LABELS:
        current = labels_frame[label].to_numpy()
        oof[label + "_pred"] = np.where(current >= 0.5, 0.05, 0.95)
    teacher_oof = workdir / "teacher_oof.csv"
    pd.DataFrame(oof).to_csv(teacher_oof, index=False)

    gold_rows = labels_frame.iloc[:N_GOLD][[STUDY_ID_COLUMN, *TARGET_LABELS]].copy()
    train_csv = workdir / "train.csv"
    gold_rows.to_csv(train_csv, index=False)

    table = gold_split.gold_studies(train_csv)
    split = gold_split.stratified_split(table)
    split_path = workdir / "gold_split_v1.json"
    gold_split.write_split(split_path, gold_split.split_payload(split, table))

    weights_path = workdir / "label_weights.json"
    weights_path.write_text(json.dumps({label: 0.85 for label in TARGET_LABELS}))

    return {
        "labels_csv": labels_csv,
        "teacher_oof": teacher_oof,
        "train_csv": train_csv,
        "split_path": split_path,
        "weights_path": weights_path,
        "gold_ids": set(gold_rows[STUDY_ID_COLUMN]),
    }


class RoundRunnerTests(unittest.TestCase):
    def test_teacher_oof_requires_every_prediction_column(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = pathlib.Path(workdir) / "bad.csv"
            pd.DataFrame({STUDY_ID_COLUMN: ["a"], "ACL_pred": [0.5]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                round_runner.load_teacher_oof(path)

    def test_identity_round_writes_unchanged_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = pathlib.Path(tmp)
            fixture = build_fixture(workdir)
            out_dir = workdir / "r0"
            result = round_runner.run_round(
                0,
                fixture["teacher_oof"],
                fixture["labels_csv"],
                out_dir,
                train_csv=fixture["train_csv"],
                split_path=fixture["split_path"],
                label_weights_path=fixture["weights_path"],
                max_corrections=0,
            )
            written = pd.read_csv(out_dir / "labels_r0.csv")
            original = pd.read_csv(fixture["labels_csv"])
            self.assertEqual(len(written), len(original))
            pd.testing.assert_frame_equal(
                written[TARGET_LABELS].astype(float), original[TARGET_LABELS].astype(float)
            )
            self.assertTrue(all(item.accepted == 0 for item in result.per_label.values()))
            self.assertIn("engine", written.columns)

    def test_report_has_the_required_shape(self) -> None:
        pytest.importorskip("cleanlab")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = pathlib.Path(tmp)
            fixture = build_fixture(workdir)
            out_dir = workdir / "r1"
            round_runner.run_round(
                1,
                fixture["teacher_oof"],
                fixture["labels_csv"],
                out_dir,
                train_csv=fixture["train_csv"],
                split_path=fixture["split_path"],
                label_weights_path=fixture["weights_path"],
            )
            report = json.loads((out_dir / "corrections_report.json").read_text())
            self.assertEqual(report["round"], 1)
            self.assertEqual(set(report["per_label"]), set(TARGET_LABELS))
            for entry in report["per_label"].values():
                for key in ("candidates", "accepted", "capped", "threshold", "threshold_source"):
                    self.assertIn(key, entry)
            self.assertEqual(len(report["codrift"]), 3)
            self.assertEqual(report["n_skipped_gold"], N_GOLD)

    def test_gold_studies_are_protected_from_correction(self) -> None:
        pytest.importorskip("cleanlab")
        with tempfile.TemporaryDirectory() as tmp:
            workdir = pathlib.Path(tmp)
            fixture = build_fixture(workdir)
            out_dir = workdir / "r1"
            result = round_runner.run_round(
                1,
                fixture["teacher_oof"],
                fixture["labels_csv"],
                out_dir,
                train_csv=fixture["train_csv"],
                split_path=fixture["split_path"],
                label_weights_path=fixture["weights_path"],
            )
            changed = result.flags.set_index(STUDY_ID_COLUMN)
            for study in fixture["gold_ids"]:
                self.assertFalse(bool(changed.loc[study].any()), msg=study)

    def test_cli_runs_the_identity_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = pathlib.Path(tmp)
            fixture = build_fixture(workdir)
            out_dir = workdir / "cli"
            code = round_runner.main(
                [
                    "--round",
                    "0",
                    "--teacher-oof",
                    str(fixture["teacher_oof"]),
                    "--labels-csv",
                    str(fixture["labels_csv"]),
                    "--out-dir",
                    str(out_dir),
                    "--train-csv",
                    str(fixture["train_csv"]),
                    "--split",
                    str(fixture["split_path"]),
                    "--label-weights",
                    str(fixture["weights_path"]),
                    "--max-corrections",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "labels_r0.csv").exists())
            self.assertTrue((out_dir / "corrections_report.json").exists())


class TriageTests(unittest.TestCase):
    def test_triage_constants(self) -> None:
        self.assertEqual(round_runner.TRIAGE_STUDIES, 20)
        self.assertEqual(round_runner.TRIAGE_MIN_VARIANCE, 0.01)

    def test_cached_study_loader_builds_slot_stacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "study.npz"
            np.savez(
                path,
                **{
                    "series-a": np.zeros((8, 224, 224), dtype=np.float16),
                    "series-b": np.ones((5, 224, 224), dtype=np.float16),
                },
            )
            stack = round_runner._load_cached_study(path)
        self.assertEqual(stack.shape, (2, 3, 224, 224))

    def test_triage_checkpoint_is_importable_and_needs_torch(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("timm")
        self.assertTrue(callable(round_runner.triage_checkpoint))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                round_runner.triage_checkpoint(pathlib.Path(tmp) / "missing.pt", tmp)


if __name__ == "__main__":
    unittest.main()
