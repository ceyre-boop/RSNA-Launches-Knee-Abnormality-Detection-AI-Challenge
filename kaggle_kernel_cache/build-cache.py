# CPU kernel: build 224px preprocessed cache for the 308-study local subset.
import sys, os, pathlib, numpy as np, pandas as pd
_hits = list(pathlib.Path("/kaggle/input").rglob("local_subset_uids.csv"))
ASSETS = str(_hits[0].parent); sys.path.insert(0, ASSETS)
COMP = str(next(pathlib.Path("/kaggle/input").rglob("train_series.csv")).parent)
from rsna_knee.imaging.slots import build_slot_table
from rsna_knee.imaging.volume import load_series_volume
uids = pd.read_csv(f"{ASSETS}/local_subset_uids.csv").StudyInstanceUID.tolist()
series = pd.read_csv(f"{COMP}/train_series.csv")
series = series[series.StudyInstanceUID.isin(uids)]
slot_table = build_slot_table(series)
out = pathlib.Path("/kaggle/working/cache224"); out.mkdir(exist_ok=True)
ok = fail = 0
for i, uid in enumerate(uids):
    rows = slot_table[slot_table.StudyInstanceUID == uid] if hasattr(slot_table, 'StudyInstanceUID') else None
    study_dir = pathlib.Path(COMP) / "train_series" / uid
    if not study_dir.exists():
        fail += 1; continue
    arrs = {}
    for sdir in study_dir.iterdir():
        try:
            vol = load_series_volume(str(sdir), img_size=224)
            arrs[sdir.name] = vol.astype(np.float16)
        except Exception as e:
            print(f"WARN {uid}/{sdir.name}: {e}")
    if arrs:
        np.savez_compressed(out / f"{uid}.npz", **arrs)
        ok += 1
    else:
        fail += 1
    if (i+1) % 25 == 0:
        print(f"progress {i+1}/{len(uids)} ok={ok} fail={fail}", flush=True)
print(f"CACHE_DONE ok={ok} fail={fail}")
