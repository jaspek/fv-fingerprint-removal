"""Follow-up to sweep_analysis2.py (read-only on images, ~1 min):
  (a) k-NN replica floor per band subset (train bona vs half the real PAI, test the other half)
  (b) fixed per-band gain estimated on disjoint PAI halves (2-fold) instead of on the test images
  (c) chain / energy-domain matching with twins drawn ONLY from the 24 real PAI outside the test subset
  (d) white-noise sigma sweep between notch and specmatch (where does the b13-21 excess cross zero)
  (e) absolute sector levels (lobe vs off-axis floor) of the chain relative to real PAI, bands 13-21
Run from this folder: python -I followup.py
"""
import os
import sys
import numpy as np
from scipy.ndimage import median_filter

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import sweep_analysis2 as A                                   # noqa: E402
from common import list_images, load_gray                    # noqa: E402
from band_analysis import knn_split                           # noqa: E402

LIMIT, K, NB, NB64 = 120, 15, 30, 64
MID = list(range(13, 22))


def feats_of(folder):
    return np.array([A.feat_img(load_gray(p), A.geo(load_gray(p).shape)) for p in list_images(folder, LIMIT)])


def main():
    ref_files = list_images(A.POPS['pai'], 500)
    pai_sub = list_images(A.POPS['pai'], LIMIT)
    held = [p for p in ref_files if p not in pai_sub]
    G = A.geo(load_gray(ref_files[0]).shape)
    prof, E64 = {}, {}
    for p in ref_files:
        F = np.fft.fft2(load_gray(p))
        prof[p] = np.bincount(G.b64.ravel(), np.log1p(np.abs(F)).ravel(), minlength=NB64) / np.maximum(G.n64, 1)
        E64[p] = np.bincount(G.b64.ravel(), (np.abs(F) ** 2).ravel(), minlength=NB64)
    Xb, Xp = feats_of(A.POPS['bona']), feats_of(A.POPS['pai'])
    Xc = {c: feats_of(A.POPS[c]) for c in ['pk1.0_s7_sm', 'pk0.6_s7_sm', 'pk0.8_s9_sm']}
    mu_p = Xp.mean(0)
    subsets = {'b0-4': range(0, 5), 'b5-9': range(5, 10), 'b10-14': range(10, 15), 'b15-19': range(15, 20),
               'b20-24': range(20, 25), 'b25-29': range(25, 30), 'b13-21': range(13, 22), 'b1-29': range(1, 30)}
    print('(a) k-NN replica floor per band subset: train bona vs real PAI half A, test half B (and swapped), mean APCER')
    h = len(Xp) // 2
    r_ = []
    for ix in subsets.values():
        ix = list(ix)
        a1, _, _ = knn_split(Xb[:, ix], Xp[:h, ix], Xp[h:, ix])
        a2, _, _ = knn_split(Xb[:, ix], Xp[h:, ix], Xp[:h, ix])
        r_.append((a1 + a2) / 2)
    print('  ' + ' '.join(f'{k:>7s}' for k in subsets))
    print('  ' + ' '.join(f'{100 * x:6.1f}%' for x in r_))

    print('(b) fixed per-band gain, 2-fold: gain from PAI half A (and the chain set), tested on PAI half B, mean APCER')
    for c, X in Xc.items():
        aps = []
        for tr, te in [(slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))]:
            g = np.exp(Xp[tr].mean(0) - X.mean(0))
            E = np.exp(X) * g
            Xf = np.log(E / E.sum(1, keepdims=True) + 1e-12)
            ap, _, _ = knn_split(Xb, Xf, Xp[te])
            aps.append(ap)
        ap0, _, _ = knn_split(Xb, X, Xp)
        print(f'  {c:12s} chain {100 * ap0:5.1f} %  -> fixgain(2-fold) {100 * np.mean(aps):5.1f} %')

    print('(c) notch(1.0,7) + specmatch / energy matching with twins drawn only from the 24 held-out real PAI')
    rng = np.random.default_rng(1)
    synth_files = list_images(A.POPS['synth'], LIMIT)
    Xsm, Xem, Xsm_em, Xnoise = [], [], {}, {s: [] for s in (0.5, 1.0, 1.5, 2.0)}
    sec_chain = []
    for i, p in enumerate(synth_files):
        x = load_gray(p)
        F = np.fft.fft2(x)
        mag, ph = np.abs(F), np.angle(F)
        logm = np.log1p(mag)
        med7 = median_filter(logm, size=7, mode='wrap')
        x10 = A.q8(A.img_from(A.notch_mag(mag, logm, med7, 1.0, G), ph))
        tw = held[rng.integers(len(held))]
        xs = A.q8(A.specmatch_img(x10, prof[tw], G))
        Xsm.append(A.feat_img(xs, G))
        Xem.append(A.feat_img(A.q8(A.energymatch_img(x10, E64[tw], G)), G))
        for s in Xnoise:
            Xnoise[s].append(A.feat_img(A.q8(A.specmatch_img(x10 + rng.normal(0, s, x.shape), prof[tw], G)), G))
        Fs = np.fft.fft2(xs)
        Ps = np.abs(Fs) ** 2
        sec = np.bincount((G.b30 * A.NSEC + G.sec).ravel(), Ps.ravel(), minlength=NB * A.NSEC).reshape(NB, A.NSEC)
        sec_chain.append(sec / Ps.sum())
    Xsm, Xem = np.array(Xsm), np.array(Xem)
    for name, X in [('notch10 + specmatch (held-out twins)', Xsm), ('notch10 + Ematch (held-out twins)', Xem)]:
        ap, db, dv = knn_split(Xb, X, Xp)
        m = X.mean(0)
        print(f'  {name:40s} APCER {100 * ap:5.1f} %  dist15 bona {db:.3f} var {dv:.3f}  centroid {np.sqrt(((m - mu_p) ** 2).sum()):.3f}  '
              f'b13-21 {np.mean([m[b] - mu_p[b] for b in MID]):+.2f}  n|d|>0.8 {(np.abs(A.cohen_d(X, Xp)) > 0.8).sum()}')
    print('(d) white noise sigma between notch(1.0,7) and specmatch (held-out twins):')
    for s, L in Xnoise.items():
        X = np.array(L)
        ap, db, dv = knn_split(Xb, X, Xp)
        m = X.mean(0)
        print(f'  sigma {s:3.1f}: APCER {100 * ap:5.1f} %  centroid {np.sqrt(((m - mu_p) ** 2).sum()):.3f}  '
              f'b13-21 {np.mean([m[b] - mu_p[b] for b in MID]):+.2f}  b22-29 {np.mean([m[b] - mu_p[b] for b in range(22, 30)]):+.2f}  '
              f'b4-12 {np.mean([m[b] - mu_p[b] for b in range(4, 13)]):+.2f}  n|d|>0.8 {(np.abs(A.cohen_d(X, Xp)) > 0.8).sum()}')
    print('(e) sector levels, bands 13-21: log of mean normalised sector energy, chain(notch10+sm) minus real PAI (nats)')
    sec_pai = []
    for p in pai_sub:
        F = np.fft.fft2(load_gray(p))
        P = np.abs(F) ** 2
        sec = np.bincount((G.b30 * A.NSEC + G.sec).ravel(), P.ravel(), minlength=NB * A.NSEC).reshape(NB, A.NSEC)
        sec_pai.append(sec / P.sum())
    sc, sp = np.array(sec_chain).mean(0), np.array(sec_pai).mean(0)
    print('  sector:      ' + ' '.join(f's{i:<5d}' for i in range(A.NSEC)))
    for b in MID:
        print(f'  band {b:2d}:     ' + ' '.join(f'{np.log(sc[b, i]) - np.log(sp[b, i]):+6.2f}' for i in range(A.NSEC)))
    d = np.log(sc[MID]).mean(0) - np.log(sp[MID]).mean(0)
    print('  mean b13-21: ' + ' '.join(f'{v:+6.2f}' for v in d))
    off = [1, 2, 3, 4, 7, 8, 9, 10]
    print(f'  lobe (s5,s6) {d[[5, 6]].mean():+.2f}, fx-axis (s0,s11) {d[[0, 11]].mean():+.2f}, off-axis floor (s1-4,s7-10) {d[off].mean():+.2f}')
    print(f'  share of band energy in the off-axis sectors: PAI {sp[MID][:, off].sum(1).mean():.3f}, chain {sc[MID][:, off].sum(1).mean():.3f}')


if __name__ == '__main__':
    main()
