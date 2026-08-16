import pathlib, numpy as np
total = 0
for d in pathlib.Path("/kaggle/input").rglob("cache224u8"):
    n = len(list(d.glob("*.npz")))
    print(f"{d}: {n} studies")
    total += n
print(f"TOTAL_CACHED {total}")
f = next(pathlib.Path("/kaggle/input").rglob("cache224u8/*.npz"))
d = np.load(f)
k = d.files[0]
print(f"sample: {len(d.files)} slots, {k} shape={d[k].shape} dtype={d[k].dtype}")
