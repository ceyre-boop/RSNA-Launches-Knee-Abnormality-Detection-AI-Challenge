# CPU shard kernel: full-corpus preprocessed cache (uint8, 224px, bbox+130mm crop,
# central-band slices, selected slot series only). SHARD set per push.
import sys, os, pathlib, numpy as np, pandas as pd
SHARD = int(os.environ.get("SHARD", "1")); NSHARDS = 3
_hits = list(pathlib.Path("/kaggle/input").rglob("pseudo_labels_sonnet_v1_3.csv"))
ASSETS = str(_hits[0].parent); sys.path.insert(0, ASSETS)
COMP = str(next(pathlib.Path("/kaggle/input").rglob("train_series.csv")).parent)
from rsna_knee.imaging.slots import build_slot_table, SLOT_NAMES
from rsna_knee.imaging.volume import load_series_volume
series = pd.read_csv(f"{COMP}/train_series.csv")
slot_table = build_slot_table(series)
plane_of = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
uids = sorted(slot_table.StudyInstanceUID.unique())
mine = [u for i, u in enumerate(uids) if i % NSHARDS == SHARD]
out = pathlib.Path("/kaggle/working/cache224u8"); out.mkdir(exist_ok=True)
print(f"shard {SHARD}/{NSHARDS}: {len(mine)} studies", flush=True)

def bbox_crop(vol):
    # content bbox on max-projection, 4px pad (intel item 5)
    m = vol.max(axis=0) > (8.0 / 255.0 if vol.max() <= 1.0 else 8)
    ys, xs = np.where(m)
    if len(ys) < 100: return vol
    y0, y1 = max(0, ys.min()-4), min(vol.shape[1], ys.max()+5)
    x0, x1 = max(0, xs.min()-4), min(vol.shape[2], xs.max()+5)
    return vol[:, y0:y1, x0:x1]

import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
cv2.setNumThreads(1)

def process_study(uid):
    row = slot_table[slot_table.StudyInstanceUID == uid].iloc[0]
    arrs = {}
    for slot in SLOT_NAMES:
        sid = row[slot]
        if sid is None or (isinstance(sid, float) and pd.isna(sid)): continue
        try:
            vol = load_series_volume(f"{COMP}/train_series/{uid}/{sid}",
                                     plane=plane_of.get(sid, "Sagittal"), img_size=336)
            vol = bbox_crop(vol)
            vol = np.stack([cv2.resize(s, (224, 224)) for s in vol])
            lo, hi = int(vol.shape[0]*0.15), max(int(vol.shape[0]*0.85), int(vol.shape[0]*0.15)+3)
            vol = vol[lo:hi]
            arrs[slot] = np.clip(vol*255, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"WARN {uid}/{slot}: {type(e).__name__}", flush=True)
    if not arrs:
        return uid, False
    np.savez_compressed(out/f"{uid}.npz", **arrs)
    return uid, True

ok = fail = done = 0
with ProcessPoolExecutor(max_workers=4) as ex:
    futures = {ex.submit(process_study, u): u for u in mine}
    for fut in as_completed(futures):
        try:
            _, success = fut.result()
            ok += success; fail += (not success)
        except Exception as e:
            fail += 1; print(f"STUDYFAIL {futures[fut]}: {type(e).__name__}", flush=True)
        done += 1
        if done % 100 == 0: print(f"{done}/{len(mine)} ok={ok}", flush=True)
print(f"SHARD_DONE shard={SHARD} ok={ok} fail={fail}")
