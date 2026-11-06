"""Self-contained PLUS ablation (no code from either repository is imported).
Re-implements: PIL grayscale load -> mahotas.features.haralick(return_mean=True) -> GroupKFold(5) by subject -> KNN.
Writes only feature caches into the scratchpad."""
import sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, mahotas, cv2
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
PLUS = Path(sys.argv[1]); OUT = Path(sys.argv[2])
BONA_FIDE, ATTACK = 0, 1

def files(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"})
def read(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)
def feats(paths, cache, pre=None):
    if cache.exists():
        return np.load(cache)
    t = time.time()
    x = np.vstack([mahotas.features.haralick(pre(read(p)) if pre else read(p), return_mean=True).ravel() for p in paths])
    np.save(cache, x); print(f"  extracted {len(paths)} images in {time.time()-t:.0f}s", flush=True)
    return x

rf, sf = files(PLUS / "real"), files(PLUS / "spoof")
key = lambda p: (p.parent.name, p.name)
common = {key(p) for p in rf} & {key(p) for p in sf}
rf_m = [p for p in rf if key(p) in common]; sf_m = [p for p in sf if key(p) in common]
print(f"PLUS: all real={len(rf)} ({len({p.parent.name for p in rf})} subjects), spoof={len(sf)} ({len({p.parent.name for p in sf})} subjects); matched real={len(rf_m)} spoof={len(sf_m)}", flush=True)

xr_all = feats(rf, OUT / "sc_plus_all_real.npy"); xs = feats(sf, OUT / "sc_plus_spoof.npy")
idx_m = [i for i, p in enumerate(rf) if key(p) in common]; xr = xr_all[idx_m]
subj_r_all = np.array([p.parent.name for p in rf]); subj_r = subj_r_all[idx_m]; subj_s = np.array([p.parent.name for p in sf])

def run(x, y, groups, k, scale, scaler_on_all=False, label=""):
    xx = StandardScaler().fit_transform(x) if (scale and scaler_on_all) else x
    ap, bp = [], []
    for tr, te in GroupKFold(n_splits=5).split(xx, y, groups):
        xtr, xte = xx[tr], xx[te]
        if scale and not scaler_on_all:
            sc = StandardScaler().fit(xtr); xtr, xte = sc.transform(xtr), sc.transform(xte)
        p = KNeighborsClassifier(n_neighbors=k).fit(xtr, y[tr]).predict(xte)
        ap.append(np.mean(p[y[te] == ATTACK] == BONA_FIDE)); bp.append(np.mean(p[y[te] == BONA_FIDE] == ATTACK))
    a, b = float(np.mean(ap)), float(np.mean(bp))
    print(f"{label:<62} APCER={a:.4f} BPCER={b:.4f} ACER={(a+b)/2:.4f}   per-fold ACER={[round((u+v)/2,3) for u,v in zip(ap,bp)]}", flush=True)

xm = np.vstack([xr, xs]); ym = np.concatenate([np.full(len(xr), BONA_FIDE), np.full(len(xs), ATTACK)]); gm = np.concatenate([subj_r, subj_s])
print("\n--- matched PLUS set (pad_train's PLUS: pair_by_filename) ---")
run(xm, ym, gm, 6, False, label="[A] matched, unscaled, k=6      (= pad_train baseline recipe)")
run(xm, ym, gm, 5, False, label="[B] matched, unscaled, k=5")
run(xm, ym, gm, 6, True,  label="[C] matched, StandardScaler fit per train fold, k=6")
run(xm, ym, gm, 5, True,  label="[D] matched, StandardScaler fit per train fold, k=5 (old recipe)")
run(xm, ym, gm, 5, True, scaler_on_all=True, label="[E] matched, scaler fit on ALL data (old custom-fold mode), k=5")
run(xm, ym, gm, 1, False, label="[F] matched, unscaled, k=1")
run(xm, ym, gm, 1, True,  label="[G] matched, scaled per fold, k=1")

xa = np.vstack([xr_all, xs]); ya = np.concatenate([np.full(len(xr_all), BONA_FIDE), np.full(len(xs), ATTACK)]); ga = np.concatenate([subj_r_all, subj_s])
print(f"\n--- full PLUS set as last year's notebooks used it ({len(xr_all)} real + {len(xs)} spoof, 60 subjects) ---")
run(xa, ya, ga, 5, True,  label="[H] all 2193, scaled per fold, k=5       (old recipe on old set)")
run(xa, ya, ga, 5, True, scaler_on_all=True, label="[I] all 2193, scaler on ALL, k=5        (old custom-fold mode)")
run(xa, ya, ga, 6, False, label="[J] all 2193, unscaled, k=6              (new recipe on old set)")
run(xa, ya, ga, 5, False, label="[K] all 2193, unscaled, k=5")

print("\n--- per-feature std on the matched set (unscaled Euclidean k-NN is dominated by the largest) ---")
names = ["ASM","Contrast","Correlation","Variance","IDM","SumAvg","SumVar","SumEntropy","Entropy","DiffVar","DiffEntropy","IMC1","IMC2"]
for n, s, mr, ms in zip(names, xm.std(0), xr.mean(0), xs.mean(0)):
    print(f"  {n:<12} std={s:12.4f}   mean real={mr:12.4f}   mean spoof={ms:12.4f}")

cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
xr_c = feats(rf_m, OUT / "sc_plus_matched_real_clahe.npy", pre=cl.apply); xs_c = feats(sf_m, OUT / "sc_plus_spoof_clahe.npy", pre=cl.apply)
xc = np.vstack([xr_c, xs_c])
print("\n--- CLAHE(clip 2.0, 8x8) before Haralick (old code's optional contrast=True), matched set ---")
run(xc, ym, gm, 6, False, label="[L] CLAHE, unscaled, k=6")
run(xc, ym, gm, 5, True,  label="[M] CLAHE, scaled per fold, k=5")
print("\nDONE")
