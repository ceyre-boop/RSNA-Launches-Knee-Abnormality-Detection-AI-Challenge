# RSNA Knee — cached fold training kernel (TABOOST solo).
# Reads the three preprocessed uint8 shard caches (mounted as kernel outputs) and
# trains one fold end-to-end. No PatientID leak check (already done, no leakage),
# no smoke phase — the cache makes a full run cheap.
#
# CONFIG: env vars do NOT survive `kaggle kernels push`, so the pusher seds these.
FOLD = 1
LABELS = "v1_1"
# ---------------------------------------------------------------------------
import subprocess, sys, os, json, pathlib

# Kaggle's preinstalled torch 2.10 dropped sm_60 (P100). Pin a build that supports
# both P100 (sm_60) and T4 (sm_75), BEFORE torch is ever imported.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.5.1", "torchvision==0.20.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "timm"], check=True)
import torch
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
x = (torch.ones(4, device="cuda") * 2).sum().item()
print("gpu_op_ok", x == 8.0)

LABELS_CSV_NAME = f"pseudo_labels_sonnet_{LABELS}.csv"
print(f"CONFIG: FOLD={FOLD} LABELS={LABELS} -> {LABELS_CSV_NAME}")

# ---- Resolve mounts defensively ----
print("INPUT DIRS:", os.listdir("/kaggle/input"))
_hits = list(pathlib.Path("/kaggle/input").rglob(LABELS_CSV_NAME))
assert _hits, f"{LABELS_CSV_NAME} not found under /kaggle/input"
ASSETS = str(_hits[0].parent)
print("ASSETS resolved to:", ASSETS)

COMP = None
for d in pathlib.Path("/kaggle/input").iterdir():
    if (d / "train_series.csv").exists():
        COMP = str(d)
if COMP is None:
    hits = list(pathlib.Path("/kaggle/input").rglob("train_series.csv"))
    assert hits, "train_series.csv not found anywhere under /kaggle/input"
    COMP = str(hits[0].parent)
print("COMP resolved to:", COMP)

# The three fullcache shard kernels mount their outputs at
# /kaggle/input/rsna-knee-fullcache-s{0,1,2}/cache224u8 (path shape varies by
# mount style, so find every dir literally named cache224u8).
CACHE_DIRS = sorted({str(p) for p in pathlib.Path("/kaggle/input").rglob("cache224u8") if p.is_dir()})
assert CACHE_DIRS, "no cache224u8 dirs found under /kaggle/input"
for c in CACHE_DIRS:
    print(f"  cache dir: {c} ({len(list(pathlib.Path(c).glob('*.npz')))} npz)")

sys.path.insert(0, ASSETS)
os.makedirs("/kaggle/working/artifacts", exist_ok=True)

# Splits: PatientID leak check already ran and found no regrouping need, so use the
# report-hash splits straight from assets (prefer a pre-finalised file if shipped).
_final = list(pathlib.Path(ASSETS).rglob("splits_final.csv"))
SPLITS = str(_final[0]) if _final else f"{ASSETS}/splits_reporthash_5fold.csv"
print("SPLITS resolved to:", SPLITS)

# ---- Per-label noise-aware weights: w = (AUC-0.5)/(AUC_best-0.5) from gold-58 ----
AUCS = {"ACL": .9062, "MCL": .9093, "Medial Meniscus": .8990, "Lateral Meniscus": .8255,
        "Medial OA": .8736, "Lateral OA": .8095, "PF OA": .8243, "Effusion": .6522,
        "Synovitis": .6762, "Baker's": .9130, "Contusion": .7551, "Fracture": .8319}
best = max(AUCS.values())
weights = {k: round(max(0.0, (v - 0.5) / (best - 0.5)), 3) for k, v in AUCS.items()}
json.dump(weights, open("/kaggle/working/artifacts/label_weights.json", "w"))
print("LABEL_WEIGHTS:", json.dumps(weights))

# ---- Train via package CLI ----
from rsna_knee.imaging import train as train_mod

argv = ["--fold", str(FOLD),
        "--labels-csv", f"{ASSETS}/{LABELS_CSV_NAME}",
        "--gold-csv", f"{COMP}/train.csv",
        "--series-csv", f"{COMP}/train_series.csv",
        "--splits-csv", SPLITS,
        "--data-root", f"{COMP}/train_series",
        "--cache-dirs", ",".join(CACHE_DIRS),
        "--out", "/kaggle/working/artifacts",
        "--epochs", "14",
        "--batch-size", "16",
        "--num-workers", "4"]
print(f"=== FULL FOLD-{FOLD} (14 epochs, cached) ===", flush=True)
print("ARGV:", " ".join(argv), flush=True)
rc = train_mod.main(argv)
print("FULL_RC:", rc)
