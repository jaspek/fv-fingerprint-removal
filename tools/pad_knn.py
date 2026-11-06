"""Test D: k-NN presentation attack detection, five-fold by user, APCER / BPCER / ACER.

Experiments:
  baseline : train bona fide vs real PAI, test bona fide vs real PAI          (no --synthetic)
  synthetic: train bona fide vs synthetic PAI, test bona fide vs real PAI     (--synthetic DIR)
The synthetic set (untouched or cleaned) is split into 5 random parts; part f is held out in fold f.

Examples:
  python pad_knn.py --folds folds_plus.csv --feature fourier --out results/pad_plus_baseline.csv
  python pad_knn.py --folds folds_plus.csv --feature fourier --synthetic .../synth --out results/pad_plus_synth.csv
  python pad_knn.py --folds folds_plus.csv --feature lbp --metric wasserstein --synthetic .../cleaned --out ...

Features (settings follow Vorderleitner & Uhl, SAC '25, Sec. 3.2):
  hist (10 bins, k=3) | var (40x40 block variance, k=3) | entropy (50x50 block entropy, k=5)
  lbp (uniform LBP r=1 and r=2 histograms, k=7, --metric wasserstein) | fourier (30 band energies, k=15)
  haralick (needs `pip install mahotas`, k=9, --param d=1)
"""
import argparse, csv, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern
from common import list_images, load_gray, norm_radius

DEFAULT_K = dict(hist=3, var=3, entropy=5, lbp=7, fourier=15, haralick=9)


def f_hist(x, **_):
    h, _ = np.histogram(x, bins=10, range=(0, 256)); return h / h.sum()


def _blocks(x, b):
    H, W = (x.shape[0] // b) * b, (x.shape[1] // b) * b
    return x[:H, :W].reshape(H // b, b, W // b, b).swapaxes(1, 2).reshape(-1, b, b)


def f_var(x, block=40, **_):
    return _blocks(x, int(block)).var(axis=(1, 2))


def f_entropy(x, block=50, **_):
    out = []
    for blk in _blocks(x, int(block)):
        h, _ = np.histogram(blk, bins=32, range=(0, 256)); p = h / h.sum(); p = p[p > 0]
        out.append(-np.sum(p * np.log2(p)))
    return np.array(out)


def f_lbp(x, **_):
    feats = []
    for P, R in [(8, 1), (16, 2)]:
        l = local_binary_pattern(x.astype(np.uint8), P, R, method='uniform')
        h, _ = np.histogram(l, bins=P + 2, range=(0, P + 2)); feats.append(h / h.sum())
    return np.concatenate(feats)


def f_fourier(x, nb=30, **_):
    F = np.abs(np.fft.fft2(x)) ** 2
    r = norm_radius(*x.shape)
    bins = np.minimum((r / (0.5 * np.sqrt(2)) * nb).astype(int), nb - 1)
    e = np.bincount(bins.ravel(), F.ravel(), minlength=nb)
    return np.log(e / e.sum() + 1e-12)


def f_haralick(x, d=1, **_):
    import mahotas
    return mahotas.features.haralick(x.astype(np.uint8), distance=int(d)).mean(axis=0)


FEATS = dict(hist=f_hist, var=f_var, entropy=f_entropy, lbp=f_lbp, fourier=f_fourier, haralick=f_haralick)


def wasserstein_hist(u, v):
    """1-D EMD between concatenated histograms (per histogram: sum of |CDF difference|)."""
    d = 0.0
    for a, b in [(u[:10], v[:10]), (u[10:], v[10:])]:
        d += np.abs(np.cumsum(a) - np.cumsum(b)).sum()
    return d


def apcer_bpcer(y, yhat):
    fn = np.sum((y == 1) & (yhat == 0)); tp = np.sum((y == 1) & (yhat == 1))
    fp = np.sum((y == 0) & (yhat == 1)); tn = np.sum((y == 0) & (yhat == 0))
    ap_ = fn / max(fn + tp, 1); bp_ = fp / max(fp + tn, 1)
    return ap_, bp_, (ap_ + bp_) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', required=True, help='CSV from make_folds.py')
    ap.add_argument('--synthetic', default=None, help='folder with synthetic (or cleaned) PAI samples')
    ap.add_argument('--feature', required=True, choices=list(FEATS))
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--metric', default='euclidean', choices=['euclidean', 'wasserstein'])
    ap.add_argument('--param', action='append', default=[], help='feature parameter k=v (e.g. d=3)')
    ap.add_argument('--limit-synth', type=int, default=0)
    ap.add_argument('--limit-real', type=int, default=0, help='subsample real rows (quick test)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    params = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in a.param}
    k = a.k or DEFAULT_K[a.feature]
    feat = FEATS[a.feature]

    rows = list(csv.DictReader(open(a.folds)))
    if a.limit_real:
        rng = np.random.default_rng(0); rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.limit_real, len(rows)), replace=False))]
    Xr = np.array([feat(load_gray(r['path']), **params) for r in rows])
    yr = np.array([int(r['label']) for r in rows]); fr = np.array([int(r['fold']) for r in rows])

    if a.synthetic:
        sfiles = list_images(a.synthetic, a.limit_synth or None)
        Xs = np.array([feat(load_gray(p), **params) for p in sfiles])
        fs = np.arange(len(sfiles)) % 5
    metric = wasserstein_hist if a.metric == 'wasserstein' else 'euclidean'

    results = []
    for f in range(5):
        bf_tr = (yr == 0) & (fr != f); te = fr == f
        if a.synthetic:
            Xtr = np.vstack([Xr[bf_tr], Xs[fs != f]])
            ytr = np.concatenate([np.zeros(bf_tr.sum()), np.ones((fs != f).sum())])
        else:
            tr = fr != f; Xtr, ytr = Xr[tr], yr[tr]
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm='brute').fit(Xtr, ytr)
        results.append(apcer_bpcer(yr[te], clf.predict(Xr[te]).astype(int)))
    res = np.array(results); mean = res.mean(axis=0)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['fold', 'APCER', 'BPCER', 'ACER'])
        for i, r in enumerate(results): w.writerow([i, f'{r[0]:.4f}', f'{r[1]:.4f}', f'{r[2]:.4f}'])
        w.writerow(['mean', f'{mean[0]:.4f}', f'{mean[1]:.4f}', f'{mean[2]:.4f}'])
    mode = f'synthetic={a.synthetic}' if a.synthetic else 'baseline'
    print(f'{a.feature} k={k} {a.metric} {mode}\n  APCER {100*mean[0]:.1f}%  BPCER {100*mean[1]:.1f}%  ACER {100*mean[2]:.1f}%  -> {a.out}')


if __name__ == '__main__':
    main()
