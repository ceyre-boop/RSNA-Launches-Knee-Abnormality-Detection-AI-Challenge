import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.constants import STUDY_ID_COLUMN, TARGET_LABELS  # noqa: E402
from rsna_knee.selftrain import gold_split  # noqa: E402
from rsna_knee.selftrain.referee import LockedSliceError, assert_locked_excluded  # noqa: E402


def synthetic_gold(n: int = 58) -> pd.DataFrame:
    rows = []
    for index in range(n):
        row = {STUDY_ID_COLUMN: f"study-{index:03d}"}
        for offset, label in enumerate(TARGET_LABELS):
            row[label] = float((index + offset) % (offset + 2) == 0)
        rows.append(row)
    return pd.DataFrame(rows)


class GoldSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = synthetic_gold()

    def test_split_sizes(self) -> None:
        split = gold_split.stratified_split(self.table)
        self.assertEqual(len(split.working), gold_split.WORKING_SIZE)
        self.assertEqual(len(split.locked), gold_split.LOCKED_SIZE)
        self.assertFalse(split.working_set & split.locked_set)

    def test_split_is_deterministic(self) -> None:
        first = gold_split.stratified_split(self.table)
        second = gold_split.stratified_split(self.table.sample(frac=1.0, random_state=7))
        self.assertEqual(first.working, second.working)
        self.assertEqual(first.locked, second.locked)

    def test_split_is_seed_sensitive(self) -> None:
        default = gold_split.stratified_split(self.table)
        other = gold_split.stratified_split(self.table, seed=1)
        self.assertNotEqual(default.locked, other.locked)

    def test_stratification_keeps_locked_positives_proportional(self) -> None:
        split = gold_split.stratified_split(self.table)
        indexed = self.table.set_index(STUDY_ID_COLUMN)
        share = gold_split.LOCKED_SIZE / len(self.table)
        for label in TARGET_LABELS:
            total = float((indexed[label] >= 0.5).sum())
            locked = float((indexed.loc[split.locked, label] >= 0.5).sum())
            self.assertLessEqual(abs(locked - total * share), 1.5, msg=label)

    def test_gold_studies_selects_only_complete_rows(self) -> None:
        frame = self.table.copy()
        frame.loc[0, TARGET_LABELS[0]] = None
        with tempfile.TemporaryDirectory() as workdir:
            path = pathlib.Path(workdir) / "train.csv"
            frame.to_csv(path, index=False)
            selected = gold_split.gold_studies(path)
        self.assertEqual(len(selected), len(frame) - 1)

    def test_write_split_refuses_to_overwrite(self) -> None:
        split = gold_split.stratified_split(self.table)
        payload = gold_split.split_payload(split, self.table)
        with tempfile.TemporaryDirectory() as workdir:
            path = pathlib.Path(workdir) / "gold_split_v1.json"
            gold_split.write_split(path, payload)
            with self.assertRaises(gold_split.SplitExistsError):
                gold_split.write_split(path, payload)
            gold_split.write_split(path, payload, overwrite=True)
            reloaded = gold_split.load_split(path)
        self.assertEqual(reloaded.working, split.working)
        self.assertEqual(reloaded.locked, split.locked)
        self.assertEqual(reloaded.seed, gold_split.GOLD_SPLIT_SEED)

    def test_cli_writes_split_once(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            train_csv = pathlib.Path(workdir) / "train.csv"
            self.table.to_csv(train_csv, index=False)
            out = pathlib.Path(workdir) / "gold_split_v1.json"
            code = gold_split.main(["--train-csv", str(train_csv), "--out", str(out)])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["n_working"], gold_split.WORKING_SIZE)
            self.assertEqual(payload["n_locked"], gold_split.LOCKED_SIZE)
            with self.assertRaises(gold_split.SplitExistsError):
                gold_split.main(["--train-csv", str(train_csv), "--out", str(out)])

    def test_locked_slice_enforcement(self) -> None:
        split = gold_split.stratified_split(self.table)
        assert_locked_excluded(split.working, split)
        with self.assertRaises(LockedSliceError):
            assert_locked_excluded(split.working + split.locked[:1], split)
        assert_locked_excluded(split.locked, split, final_unlock=True)

    def test_committed_split_matches_generator(self) -> None:
        committed = REPO_ROOT / gold_split.DEFAULT_SPLIT_PATH
        if not committed.exists():
            self.skipTest("gold split not generated yet")
        table = gold_split.gold_studies(REPO_ROOT / gold_split.DEFAULT_TRAIN_CSV)
        expected = gold_split.stratified_split(table)
        actual = gold_split.load_split(committed)
        self.assertEqual(actual.working, expected.working)
        self.assertEqual(actual.locked, expected.locked)


if __name__ == "__main__":
    unittest.main()
