# RSNA Knee — fold-0 training (TABOOST solo). Steps:
# 1) PatientID leak check from DICOM headers (blocks CV trust decision)
# 2) Smoke train (--limit 60) to validate the pipeline on Kaggle hardware
# 3) Full fold-0 training with per-label noise-aware loss weights
import subprocess, sys, os, json, glob, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

import pathlib
_hits = list(pathlib.Path("/kaggle/input").rglob("pseudo_labels_sonnet_v1_1.csv"))
assert _hits, "assets dataset not found under /kaggle/input"
ASSETS = str(_hits[0].parent)
print("ASSETS resolved to:", ASSETS)
# Resolve competition mount defensively: find the dir containing train_series.csv
print("INPUT DIRS:", os.listdir("/kaggle/input"))
COMP = None
for d in pathlib.Path("/kaggle/input").iterdir():
    print(f"  {d.name}: {[f.name for f in list(d.iterdir())[:8]]}")
    if (d / "train_series.csv").exists():
        COMP = str(d)
if COMP is None:
    hits = list(pathlib.Path("/kaggle/input").rglob("train_series.csv"))
    assert hits, "train_series.csv not found anywhere under /kaggle/input"
    COMP = str(hits[0].parent)
print("COMP resolved to:", COMP)
sys.path.insert(0, ASSETS)
os.makedirs("/kaggle/working/artifacts", exist_ok=True)

# ---- Step 1: PatientID leak check ----
import pydicom, pandas as pd
series = pd.read_csv(f"{COMP}/train_series.csv")
pat_of_study = {}
for study, grp in series.groupby("StudyInstanceUID"):
    sdir = f"{COMP}/train_series/{study}/{grp.SeriesInstanceUID.iloc[0]}"
    files = glob.glob(f"{sdir}/*.dcm")
    if not files:
        continue
    ds = pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=["PatientID"])
    pat_of_study[study] = str(getattr(ds, "PatientID", "") or "")
pats = collections.Counter(pat_of_study.values())
multi = {p: c for p, c in pats.items() if p and c > 1}
report = {
    "studies_checked": len(pat_of_study),
    "unique_patients": len(pats),
    "patients_with_multiple_studies": len(multi),
    "max_studies_per_patient": max(pats.values()) if pats else 0,
    "empty_patient_ids": sum(1 for v in pat_of_study.values() if not v),
}
print("PATIENT_ID_CHECK:", json.dumps(report))
pd.DataFrame(pat_of_study.items(), columns=["StudyInstanceUID", "PatientID"]).to_csv(
    "/kaggle/working/artifacts/patient_ids.csv", index=False)

# Regroup splits on PatientID if leakage is possible
splits = pd.read_csv(f"{ASSETS}/splits_reporthash_5fold.csv")
if multi:
    print(f"LEAK RISK: {len(multi)} patients have >1 study -> regrouping folds on PatientID")
    splits["patient"] = splits.StudyInstanceUID.map(pat_of_study).fillna(splits.StudyInstanceUID)
    pf = {p: i % 5 for i, p in enumerate(sorted(splits.patient.unique()))}
    splits["fold"] = splits.patient.map(pf)
splits.to_csv("/kaggle/working/artifacts/splits_final.csv", index=False)

# ---- Per-label noise-aware weights: w = (AUC-0.5)/(AUC_best-0.5) from gold-58 ----
AUCS = {"ACL": .9062, "MCL": .9093, "Medial Meniscus": .8990, "Lateral Meniscus": .8255,
        "Medial OA": .8736, "Lateral OA": .8095, "PF OA": .8243, "Effusion": .6522,
        "Synovitis": .6762, "Baker's": .9130, "Contusion": .7551, "Fracture": .8319}
best = max(AUCS.values())
weights = {k: round(max(0.0, (v - 0.5) / (best - 0.5)), 3) for k, v in AUCS.items()}
json.dump(weights, open("/kaggle/working/artifacts/label_weights.json", "w"))
print("LABEL_WEIGHTS:", json.dumps(weights))

# ---- Step 2+3: train via package CLI ----
from rsna_knee.imaging import train as train_mod
common = ["--fold", "0", "--labels-csv", f"{ASSETS}/pseudo_labels_sonnet_v1_1.csv",
          "--gold-csv", f"{COMP}/train.csv", "--series-csv", f"{COMP}/train_series.csv",
          "--splits-csv", "/kaggle/working/artifacts/splits_final.csv",
          "--data-root", f"{COMP}/train_series", "--out", "/kaggle/working/artifacts",
          "--batch-size", "8", "--num-workers", "4"]
print("=== SMOKE (limit 60, 1 epoch) ===", flush=True)
rc = train_mod.main(common + ["--limit", "60", "--epochs", "1"])
print("SMOKE_RC:", rc, flush=True)
if rc == 0:
    print("=== FULL FOLD-0 (10 epochs) ===", flush=True)
    rc = train_mod.main(common + ["--epochs", "10"])
    print("FULL_RC:", rc)
