# RSNA Knee — offline inference/submission kernel (fold-0 single model, sliding-window TTA).
# Internet OFF. Deps installed from the wheels dataset; weights from same dataset.
# Robustness contract: every test study gets a row; any failure -> 0.5 defaults.
import subprocess, sys, os, pathlib, time
T0 = time.time()

def find(name):
    hits = list(pathlib.Path("/kaggle/input").rglob(name))
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]

# Kaggle strips '+' from uploaded filenames; restore PEP440-valid wheel names in /tmp
import shutil, glob
wsrc = find("fold0_best.pt").parent
wtmp = "/tmp/wheels"; os.makedirs(wtmp, exist_ok=True)
for whl in glob.glob(f"{wsrc}/*.whl"):
    base = os.path.basename(whl)
    fixed = base.replace("2.5.1cu121", "2.5.1+cu121").replace("0.20.1cu121", "0.20.1+cu121")
    shutil.copy(whl, f"{wtmp}/{fixed}")
print("wheels staged:", len(os.listdir(wtmp)))
pkgs = ["nvidia-cuda-nvrtc-cu12==12.1.105", "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cuda-cupti-cu12==12.1.105", "nvidia-cudnn-cu12==9.1.0.70",
        "nvidia-cublas-cu12==12.1.3.1", "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106", "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106", "nvidia-nccl-cu12==2.21.5",
        "nvidia-nvtx-cu12==12.1.105", "nvidia-nvjitlink-cu12==12.1.105",
        "triton==3.1.0", "torch==2.5.1+cu121", "torchvision==0.20.1+cu121", "timm==1.0.28"]
# Pins force pip to actually install our wheels over Kaggle's newer preinstalls
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                "--force-reinstall", f"--find-links={wtmp}", *pkgs], check=True)
# Sanity gate: fail loudly if GPU ops are broken (never emit an all-0.5 submission silently)
import torch as _t
assert _t.cuda.is_available() and ( _t.ones(2, device="cuda") * 2 ).sum().item() == 4.0, "GPU op check failed"

ASSETS = str(find("pseudo_labels_sonnet_v1_3.csv").parent)
sys.path.insert(0, ASSETS)
COMP = str(find("test_series.csv").parent)
CKPT = str(find("fold0_best.pt"))

import numpy as np, pandas as pd, torch
from rsna_knee.constants import TARGET_LABELS, STUDY_ID_COLUMN
from rsna_knee.imaging.slots import build_slot_table, SLOT_NAMES
from rsna_knee.imaging.volume import load_series_volume
from rsna_knee.imaging.model import KneeSlotModel

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = KneeSlotModel(backbone=ckpt["args"].get("backbone", "vit_small_patch14_dinov2"),
                      pretrained=False)
model.load_state_dict(ckpt["model"])
model.to(device).eval()
print(f"model loaded (epoch {ckpt['epoch']}, val macro {ckpt['macro_auc']:.4f}) on {device}")

test = pd.read_csv(f"{COMP}/test.csv")
series = pd.read_csv(f"{COMP}/test_series.csv")
slot_table = build_slot_table(series)
IMG = int(ckpt["args"].get("img_size", 224))
GROUP = 3

# ---- Sliding-window TTA with per-target pooling (round-2 intel item 1) ----
# All consecutive GROUP-slice windows across the central band, capped at MAX_W
# uniformly spaced windows to bound runtime. Focal findings show up in 1-2
# windows, so averaging every window divides their evidence away.
BAND = (0.2, 0.8)
MAX_W = 8
CHUNK = 4  # windows forwarded per batch

POOL = {"Fracture": "max", "Contusion": "max", "Medial Meniscus": "max",
        "Lateral Meniscus": "max", "Baker's": "max",
        "ACL": "top2", "MCL": "top2",
        "Medial OA": "mean", "Lateral OA": "mean", "PF OA": "mean",
        "Effusion": "mean", "Synovitis": "mean"}
BETA = {"ACL": 6.0, "MCL": 6.0,
        "Medial Meniscus": 8.0, "Lateral Meniscus": 8.0, "Baker's": 8.0,
        "Contusion": 8.0, "Fracture": 10.0}
SOFT_MIX = 0.2  # final = (1-SOFT_MIX)*hard + SOFT_MIX*soft


def window_starts(n_slices, group=GROUP, band=BAND):
    """Every consecutive-window start index inside the central band."""
    if n_slices < group:
        return [0]
    low = max(0, min(int(np.floor(band[0] * n_slices)), n_slices - group))
    high = min(int(np.ceil(band[1] * n_slices)), n_slices)
    last = max(low, min(high - group, n_slices - group))
    return list(range(low, last + 1))


def window_indices(n_slices, start, group=GROUP):
    if n_slices < group:
        return [min(i, n_slices - 1) for i in range(group)]
    return list(range(start, start + group))


def pool_windows(probs):
    """probs [W, T] -> [T]: per-target hard pooling blended with softmax-temp soft pooling."""
    out = np.empty(probs.shape[1], dtype=np.float64)
    for j, label in enumerate(TARGET_LABELS):
        col = probs[:, j].astype(np.float64)
        mode = POOL[label]
        if mode == "max":
            hard = float(col.max())
        elif mode == "top2":
            hard = float(np.sort(col)[-2:].mean())
        else:
            hard = float(col.mean())
        if mode == "mean":
            soft = hard
        else:
            z = BETA[label] * col
            e = np.exp(z - z.max())
            soft = float((e / e.sum() * col).sum())
        out[j] = (1.0 - SOFT_MIX) * hard + SOFT_MIX * soft
    return out


rows = []
done = 0
win_counts = []
for uid in test[STUDY_ID_COLUMN]:
    scores = {label: 0.5 for label in TARGET_LABELS}
    try:
        srow = slot_table[slot_table[STUDY_ID_COLUMN] == uid]
        plane_of = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
        vols, starts, mask = [], [], []
        for slot in SLOT_NAMES:
            sid = srow.iloc[0][slot] if len(srow) else None
            if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                vols.append(None); starts.append([0]); mask.append(False); continue
            plane = plane_of.get(sid, "Sagittal")
            vol = load_series_volume(f"{COMP}/test_series/{uid}/{sid}", plane=plane, img_size=IMG)
            vols.append(vol); starts.append(window_starts(vol.shape[0])); mask.append(True)
        if any(mask):
            n_max = max(len(s) for s, keep in zip(starts, mask) if keep)
            W = max(1, min(MAX_W, n_max))
            win_counts.append(W)
            stacks = []
            for w in range(W):
                slots = []
                for vol, st, keep in zip(vols, starts, mask):
                    if not keep:
                        slots.append(torch.zeros(GROUP, IMG, IMG)); continue
                    n_i = len(st)
                    wi = 0 if W == 1 else int(round(w * (n_i - 1) / (W - 1)))
                    idx = window_indices(vol.shape[0], st[min(wi, n_i - 1)])
                    slots.append(torch.from_numpy(vol[idx].astype(np.float32)))
                stacks.append(torch.stack(slots))
            m = torch.tensor([mask]).to(device)
            probs = []
            for i in range(0, W, CHUNK):
                xb = torch.stack(stacks[i:i + CHUNK]).to(device)
                mb = m.expand(xb.shape[0], -1)
                with torch.no_grad():
                    logits = model(xb, mb)
                probs.append(torch.sigmoid(logits).float().cpu().numpy())
            probs = np.concatenate(probs, axis=0)  # [W, T]
            pooled = pool_windows(probs)
            scores = {label: float(p) for label, p in zip(TARGET_LABELS, pooled)}
    except Exception as e:
        print(f"WARN {uid}: {type(e).__name__}: {e}")
    rows.append({STUDY_ID_COLUMN: uid, **scores})
    done += 1
    if done % 50 == 0:
        mw = float(np.mean(win_counts)) if win_counts else 0.0
        print(f"{done}/{len(test)} elapsed={time.time()-T0:.0f}s mean_windows={mw:.1f}", flush=True)

sub = pd.DataFrame(rows, columns=[STUDY_ID_COLUMN, *TARGET_LABELS])
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"submission.csv: {len(sub)} rows, {time.time()-T0:.0f}s total, "
      f"mean_windows={float(np.mean(win_counts)) if win_counts else 0.0:.2f}")
