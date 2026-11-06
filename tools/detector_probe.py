"""Test C (tier 2): retrainable radial-spectrum detector (Durall et al. 2020 style), real vs synthetic.

Trains an SVM on the azimuthally averaged power spectrum of square centre tiles, five random folds,
reports CV accuracy / AUC real-vs-synthetic, then scores extra sets (e.g. cleaned data):
fraction classified as synthetic and AUC against the held-out real tiles.

Example:
  python detector_probe.py --real .../PLUS/real --fake .../synth --eval cleaned=.../cleaned \
      --eval jpeg=.../synth_jpeg --tile 192 --limit 400 --out results/probe_plus.csv
"""
import argparse, csv, os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import list_images, load_gray, center_tile, log_spectrum, radial_profile


def feats(files, tile):
    return np.array([radial_profile(log_spectrum(center_tile(load_gray(p), tile)))[1:tile // 2] for p in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True); ap.add_argument('--fake', required=True)
    ap.add_argument('--eval', action='append', default=[], help='name=DIR (repeatable)')
    ap.add_argument('--tile', type=int, default=0); ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fr, ff = list_images(a.real, a.limit), list_images(a.fake, a.limit)
    tile = a.tile or min(load_gray(fr[0]).shape)
    Xr, Xf = feats(fr, tile), feats(ff, tile)
    X = np.vstack([Xr, Xf]); y = np.r_[np.zeros(len(Xr)), np.ones(len(Xf))]
    folds = np.arange(len(X)) % 5
    rng = np.random.default_rng(0); rng.shuffle(folds)
    evals = {}
    for spec in a.eval:
        n, d = spec.split('=', 1); evals[n] = feats(list_images(d, a.limit), tile)

    accs, aucs, ev_frac, ev_auc = [], [], {n: [] for n in evals}, {n: [] for n in evals}
    for f in range(5):
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, probability=True)).fit(X[folds != f], y[folds != f])
        p = clf.predict_proba(X[folds == f])[:, 1]
        accs.append(np.mean((p > 0.5) == y[folds == f])); aucs.append(roc_auc_score(y[folds == f], p))
        pr = p[y[folds == f] == 0]
        for n, Xe in evals.items():
            pe = clf.predict_proba(Xe)[:, 1]
            ev_frac[n].append(np.mean(pe > 0.5))
            ev_auc[n].append(roc_auc_score(np.r_[np.zeros(len(pr)), np.ones(len(pe))], np.r_[pr, pe]))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['set', 'metric', 'value'])
        w.writerow(['real_vs_fake', 'cv_accuracy', f'{np.mean(accs):.4f}']); w.writerow(['real_vs_fake', 'cv_auc', f'{np.mean(aucs):.4f}'])
        for n in evals:
            w.writerow([n, 'fraction_called_synthetic', f'{np.mean(ev_frac[n]):.4f}'])
            w.writerow([n, 'auc_vs_real', f'{np.mean(ev_auc[n]):.4f}'])
    print(f'radial-spectrum SVM (tile {tile}): real vs fake accuracy {np.mean(accs):.3f}, AUC {np.mean(aucs):.3f}')
    for n in evals:
        print(f'  {n}: called synthetic {100*np.mean(ev_frac[n]):.1f}%, AUC vs real {np.mean(ev_auc[n]):.3f}')
    print('->', a.out)


if __name__ == '__main__':
    main()
