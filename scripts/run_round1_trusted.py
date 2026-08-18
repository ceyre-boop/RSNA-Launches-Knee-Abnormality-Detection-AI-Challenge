import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rsna_knee.selftrain.round_runner import run_round, format_summary
from rsna_knee.selftrain.referee import TRUSTED_LABELS

if __name__ == "__main__":
    result = run_round(
        round_index=1,
        teacher_oof="artifacts/full_corpus_oof.csv",
        labels_csv="data/pseudo_labels_sonnet_v1_1.csv",
        out_dir="artifacts/selftrain/round1_trusted",
        labels=TRUSTED_LABELS,
    )
    print(format_summary(1, result))
