import pathlib
import sys
import unittest

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsna_knee.imaging.slots import (  # noqa: E402
    SLOT_NAMES,
    build_slot_table,
    select_slots,
)

COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]


def frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


class SelectSlotsTests(unittest.TestCase):
    def test_full_study_routes_every_slot(self) -> None:
        df = frame(
            [
                ["s1", "sag_fs", 1, 1, "Sagittal"],
                ["s1", "cor_fs", 1, 1, "Coronal"],
                ["s1", "ax_fs", 1, 1, "Axial"],
                ["s1", "sag_nofs", 1, 0, "Sagittal"],
                ["s1", "cor_t1", 0, 0, "Coronal"],
                ["s1", "sag_t1", 0, 0, "Sagittal"],
            ]
        )
        selected = select_slots(df, "s1")
        self.assertEqual(
            selected,
            {
                "SAG_FLUID_FS": "sag_fs",
                "COR_FLUID_FS": "cor_fs",
                "AX_FLUID_FS": "ax_fs",
                "SAG_NOFS": "sag_nofs",
                "COR_T1": "cor_t1",
                "SAG_T1": "sag_t1",
            },
        )

    def test_missing_slots_are_none(self) -> None:
        df = frame([["s1", "only", 1, 1, "Sagittal"]])
        selected = select_slots(df, "s1")
        self.assertEqual(selected["SAG_FLUID_FS"], "only")
        for slot in SLOT_NAMES[1:]:
            self.assertIsNone(selected[slot])

    def test_unknown_study_is_all_none(self) -> None:
        df = frame([["s1", "only", 1, 1, "Sagittal"]])
        self.assertEqual(set(select_slots(df, "nope").values()), {None})

    def test_other_studies_ignored(self) -> None:
        df = frame(
            [
                ["s1", "mine", 1, 1, "Sagittal"],
                ["s2", "theirs", 1, 1, "Sagittal"],
            ]
        )
        self.assertEqual(select_slots(df, "s1")["SAG_FLUID_FS"], "mine")
        self.assertEqual(select_slots(df, "s2")["SAG_FLUID_FS"], "theirs")

    def test_tie_break_prefers_more_slices_via_column(self) -> None:
        df = frame(
            [
                ["s1", "short", 1, 1, "Sagittal"],
                ["s1", "long", 1, 1, "Sagittal"],
            ]
        )
        df["n_slices"] = [12, 40]
        self.assertEqual(select_slots(df, "s1", slice_counts="n_slices")["SAG_FLUID_FS"], "long")

    def test_tie_break_accepts_callable(self) -> None:
        df = frame(
            [
                ["s1", "a", 1, 1, "Coronal"],
                ["s1", "b", 1, 1, "Coronal"],
            ]
        )
        counts = {"a": 5, "b": 25}
        selected = select_slots(df, "s1", slice_counts=lambda uid: counts[uid])
        self.assertEqual(selected["COR_FLUID_FS"], "b")

    def test_tie_break_accepts_dict(self) -> None:
        df = frame([["s1", "a", 1, 1, "Axial"], ["s1", "b", 1, 1, "Axial"]])
        self.assertEqual(select_slots(df, "s1", slice_counts={"a": 30, "b": 3})["AX_FLUID_FS"], "a")

    def test_equal_counts_break_deterministically_on_uid(self) -> None:
        df = frame([["s1", "zzz", 1, 1, "Sagittal"], ["s1", "aaa", 1, 1, "Sagittal"]])
        selected = select_slots(df, "s1", slice_counts={"zzz": 20, "aaa": 20})
        self.assertEqual(selected["SAG_FLUID_FS"], "aaa")

    def test_no_counts_still_deterministic(self) -> None:
        df = frame([["s1", "b", 1, 1, "Sagittal"], ["s1", "a", 1, 1, "Sagittal"]])
        self.assertEqual(select_slots(df, "s1")["SAG_FLUID_FS"], "a")

    def test_t1_slots_ignore_fat_suppression(self) -> None:
        df = frame([["s1", "cor_t1_fs", 0, 1, "Coronal"]])
        self.assertEqual(select_slots(df, "s1")["COR_T1"], "cor_t1_fs")

    def test_fluid_fs_does_not_capture_nofs(self) -> None:
        df = frame([["s1", "nofs", 1, 0, "Sagittal"]])
        selected = select_slots(df, "s1")
        self.assertIsNone(selected["SAG_FLUID_FS"])
        self.assertEqual(selected["SAG_NOFS"], "nofs")

    def test_plane_matching_is_case_insensitive(self) -> None:
        df = frame([["s1", "x", 1, 1, " sagittal "]])
        self.assertEqual(select_slots(df, "s1")["SAG_FLUID_FS"], "x")

    def test_missing_columns_raise(self) -> None:
        df = pd.DataFrame({"StudyInstanceUID": ["s1"]})
        with self.assertRaises(ValueError):
            select_slots(df, "s1")

    def test_bad_slice_counts_type(self) -> None:
        df = frame([["s1", "a", 1, 1, "Sagittal"]])
        with self.assertRaises(TypeError):
            select_slots(df, "s1", slice_counts=17)

    def test_unknown_count_column(self) -> None:
        df = frame([["s1", "a", 1, 1, "Sagittal"]])
        with self.assertRaises(KeyError):
            select_slots(df, "s1", slice_counts="nope")


class SlotTableTests(unittest.TestCase):
    def test_one_row_per_study(self) -> None:
        df = frame(
            [
                ["s1", "a", 1, 1, "Sagittal"],
                ["s1", "b", 1, 1, "Axial"],
                ["s2", "c", 0, 0, "Coronal"],
            ]
        )
        table = build_slot_table(df)
        self.assertEqual(list(table.columns), ["StudyInstanceUID"] + SLOT_NAMES)
        self.assertEqual(len(table), 2)
        row = table.set_index("StudyInstanceUID").loc["s2"]
        self.assertEqual(row["COR_T1"], "c")
        self.assertTrue(pd.isna(row["SAG_FLUID_FS"]))


if __name__ == "__main__":
    unittest.main()
