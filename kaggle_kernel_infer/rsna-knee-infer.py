# RSNA Knee — offline inference/submission kernel (fold-0 single model).
# Internet OFF. Deps installed from the wheels dataset; weights from same dataset.
# Robustness contract: every test study gets a row; any failure -> 0.5 defaults.
import subprocess, sys, os, pathlib, time
T0 = time.time()

def find(name):
    hits = list(pathlib.Path("/kaggle/input").rglob(name))
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]

WHEELS = str(find("torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl").parent)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index",
                f"--find-links={WHEELS}", "torch", "torchvision", "timm"], check=True)

ASSETS = str(find("pseudo_labels_sonnet_v1_3.csv").parent)
sys.path.insert(0, ASSETS)
COMP = str(find("test_series.csv").parent)
CKPT = str(find("fold0_best.pt"))

import numpy as np, pandas as pd, torch
from rsna_knee.constants import TARGET_LABELS, STUDY_ID_COLUMN
from rsna_knee.imaging.slots import build_slot_table, SLOT_NAMES
from rsna_knee.imaging.volume import load_series_volume
from rsna_knee.imaging.dataset import sample_group_indices
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

rows = []
done = 0
for uid in test[STUDY_ID_COLUMN]:
    scores = {label: 0.5 for label in TARGET_LABELS}
    try:
        srow = slot_table[slot_table[STUDY_ID_COLUMN] == uid]
        plane_of = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
        slots, mask = [], []
        for slot in SLOT_NAMES:
            sid = srow.iloc[0][slot] if len(srow) else None
            if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                slots.append(torch.zeros(GROUP, IMG, IMG)); mask.append(False); continue
            plane = plane_of.get(sid, "Sagittal")
            vol = load_series_volume(f"{COMP}/test_series/{uid}/{sid}", plane=plane, img_size=IMG)
            idx = sample_group_indices(vol.shape[0], group=GROUP, rng=None)
            slots.append(torch.from_numpy(vol[idx].astype(np.float32))); mask.append(True)
        if any(mask):
            x = torch.stack(slots).unsqueeze(0).to(device)
            m = torch.tensor([mask]).to(device)
            with torch.no_grad():
                logits = model(x, m)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
            scores = {label: float(p) for label, p in zip(TARGET_LABELS, probs)}
    except Exception as e:
        print(f"WARN {uid}: {type(e).__name__}: {e}")
    rows.append({STUDY_ID_COLUMN: uid, **scores})
    done += 1
    if done % 50 == 0:
        print(f"{done}/{len(test)} elapsed={time.time()-T0:.0f}s", flush=True)

sub = pd.DataFrame(rows, columns=[STUDY_ID_COLUMN, *TARGET_LABELS])
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"submission.csv: {len(sub)} rows, {time.time()-T0:.0f}s total")
