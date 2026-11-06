"""Extension of band_analysis.py to the t / window sweep of the notch -> specmatch chain.

Read-only on the images. Adds the new chain sets (pk0.6_s7_sm, pk0.8_s5_sm, pk1.0_s5_sm,
pk0.8_s9_sm, pk1.0_s9_sm) and notch-only sets (pk0.6_s7, pk0.8_s5, pk0.8_s9) to the per-band
table, and measures what is needed to decide why lowering t below 1.0 no longer closes the gap
in bands 13-21 and why a 5x5 window is much worse / 9x9 slightly better:
  * flagged-bin counts and flagged energy share per t and per window on the untouched synthetic
  * peakiness (log mean|F|^2 - 2 mean log|F|) of the notched map as a function of t, of the
    median surface itself, of the valley-filled map, of the ring-median capped/filled map
  * within-ring quantiles of log|F| relative to the local 7x7 median and to the ring median
  * fx=0 / fy=0 ridge energy share, ridge cross-section, diagonal-peak neighbourhood, mean 2-D
    peak shape (width in bins along fx and fy), window bias (med_w - med_11) at peak and ridge bins
  * angular-sector energy shares (anisotropy)
  * feature-domain replica of the chain (notch -> uint8 -> specmatch mode 1 with the same twin
    draws -> uint8 -> f_fourier), validated against the actual sets, then candidate fixes on top:
    t=0 cap, ridge removal, ring-median notch, valley fill, ring clamp, white noise, energy-domain
    ring matching (twin / population mean), and a population-level fixed per-band gain.
  * single-split k-NN replica (plain and with the specmatch twins of each test image excluded).

Run from this folder:  python -I sweep_analysis2.py [limit]
Outputs (same folder): band_table2.csv, sim_results.csv, report2.txt
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import median_filter

OUT = os.path.dirname(os.path.abspath(__file__))
TOOLS = r'D:\Studium\Master\MultimediaSecurity\PS\tools'
sys.path.insert(0, TOOLS)
sys.path.insert(0, OUT)
from common import list_images, load_gray                     # noqa: E402
from band_analysis import band_edges, cohen_d, knn_split       # noqa: E402

S = r'D:\Studium\Master\MultimediaSecurity\PS\smoke_subset'
AE = os.path.join(S, 'AE_Res')
POPS = {
    'bona': os.path.join(S, 'PLUS', 'real'),
    'pai': os.path.join(S, 'PLUS', 'spoof'),
    'synth': os.path.join(AE, 'PLUS_spoof'),
    'pk0.6_s7': os.path.join(AE, 'pk0.6_s7'),
    'pk1.0_s7': os.path.join(AE, 'pk1.0_s7'),
    'pk0.8_s5': os.path.join(AE, 'pk0.8_s5'),
    'pk0.8_s9': os.path.join(AE, 'pk0.8_s9'),
    'sm': os.path.join(AE, 'sm'),
    'pk0.6_s7_sm': os.path.join(AE, 'pk0.6_s7_sm'),
    'pk1.0_s7_sm': os.path.join(AE, 'pk1.0_s7_sm'),
    'pk2.0_s7_sm': os.path.join(AE, 'pk2.0_s7_sm'),
    'pk0.8_s5_sm': os.path.join(AE, 'pk0.8_s5_sm'),
    'pk1.0_s5_sm': os.path.join(AE, 'pk1.0_s5_sm'),
    'pk0.8_s9_sm': os.path.join(AE, 'pk0.8_s9_sm'),
    'pk1.0_s9_sm': os.path.join(AE, 'pk1.0_s9_sm'),
}
NOTCH_ONLY = ['synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9']
CHAINS = ['sm', 'pk0.6_s7_sm', 'pk1.0_s7_sm', 'pk2.0_s7_sm', 'pk0.8_s5_sm', 'pk1.0_s5_sm', 'pk0.8_s9_sm', 'pk1.0_s9_sm']
VARIANTS = NOTCH_ONLY + CHAINS
NB, NB64, K, RMIN = 30, 64, 15, 0.1
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 120
T_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
V_LIST = [-0.5, -1.0, -1.5]
Q_LIST = [1, 5, 25, 50, 75, 95, 99]
WIN_LIST = [5, 7, 9, 11]
NSEC = 12
HALF = 8                     # neighbourhood half-width for cross-sections / peak shapes
MID = list(range(13, 22))    # the residual bands
SIM_GRID = [(0.6, 7), (0.8, 7), (1.0, 7), (0.6, 5), (0.8, 5), (1.0, 5), (0.6, 9), (0.8, 9), (1.0, 9)]

_GEO = {}


class Geo:
    def __init__(self, shape):
        h, w = shape
        fy = np.fft.fftfreq(h)[:, None] * np.ones((1, w))
        fx = np.fft.fftfreq(w)[None, :] * np.ones((h, 1))
        self.h, self.w = h, w
        self.r = np.hypot(fy, fx)
        self.fy, self.fx = fy, fx
        self.b30 = np.minimum((self.r / (0.5 * np.sqrt(2)) * NB).astype(int), NB - 1)
        self.b64 = np.minimum((self.r / (0.5 * np.sqrt(2)) * NB64).astype(int), NB64 - 1)
        self.n30 = np.bincount(self.b30.ravel(), minlength=NB)
        self.n64 = np.bincount(self.b64.ravel(), minlength=NB64)
        ky = np.rint(np.fft.fftfreq(h) * h).astype(int)[:, None] * np.ones((1, w), int)
        kx = np.rint(np.fft.fftfreq(w) * w).astype(int)[None, :] * np.ones((h, 1), int)
        self.ky, self.kx = ky, kx
        self.abv = self.r > RMIN
        self.col0 = (np.abs(kx) <= 1) & self.abv          # fx=0 ridge (3 columns)
        self.row0 = (np.abs(ky) <= 1) & self.abv          # fy=0 ridge (3 rows)
        self.offaxis = (np.abs(kx) > 2) & (np.abs(ky) > 2) & self.abv
        ang = np.degrees(np.arctan2(fy, fx)) % 180.0
        self.sec = np.minimum((ang / (180.0 / NSEC)).astype(int), NSEC - 1)
        self.ridge_rows = (np.abs(fy[:, 0]) >= 0.33) & (np.abs(fy[:, 0]) <= 0.5)
        self.mid_rows = (np.abs(fy[:, 0]) >= 0.15) & (np.abs(fy[:, 0]) < 0.33)
        self.ridge_cols = (np.abs(fx[0, :]) >= 0.33) & (np.abs(fx[0, :]) <= 0.5)
        self.offs = np.arange(-HALF, HALF + 1)
        # ring order for ring medians / per-band quantiles
        self.b30_sorted = None


def geo(shape):
    if shape not in _GEO:
        _GEO[shape] = Geo(shape)
    return _GEO[shape]


def q8(y):
    """save_gray + load_gray round trip (uint8)."""
    return np.clip(np.rint(np.clip(y, 0, 255)), 0, 255).astype(np.float32)


def band_feat(P, G):
    E = np.bincount(G.b30.ravel(), P.ravel(), minlength=NB)
    return np.log(E / E.sum() + 1e-12), E


def peakiness_of(lm, G):
    """lm = log1p magnitude map. Returns per-30-band log(mean P) - 2 mean log|F| and the band energies."""
    m = np.expm1(lm)
    E = np.bincount(G.b30.ravel(), (m ** 2).ravel(), minlength=NB)
    ml = np.bincount(G.b30.ravel(), np.log(m + 1e-9).ravel(), minlength=NB) / G.n30
    return np.log(E / G.n30) - 2 * ml, E


def ring_median(v, G):
    order = np.lexsort((v.ravel(), G.b64.ravel()))
    sv = v.ravel()[order]
    starts = np.r_[0, np.cumsum(G.n64)[:-1]]
    return np.array([sv[s + n // 2] if n else 0.0 for s, n in zip(starts, G.n64)])


def band_quantiles(v, G):
    order = np.lexsort((v.ravel(), G.b30.ravel()))
    sv = v.ravel()[order]
    starts = np.r_[0, np.cumsum(G.n30)[:-1]]
    out = np.zeros((len(Q_LIST), NB))
    for b in range(NB):
        s, n = starts[b], G.n30[b]
        for qi, q in enumerate(Q_LIST):
            out[qi, b] = sv[s + int(round(q / 100.0 * (n - 1)))]
    return out


def notch_mag(mag, logm, med, t, G):
    mask = ((logm - med) > t) & G.abv
    nm = mag.copy()
    nm[mask] = np.expm1(med[mask])
    return nm


def img_from(mag, ph):
    return np.clip(np.real(np.fft.ifft2(mag * np.exp(1j * ph))), 0, 255)


def specmatch_img(x, ref, G):
    F = np.fft.fft2(x)
    own = np.bincount(G.b64.ravel(), np.log1p(np.abs(F)).ravel(), minlength=NB64) / np.maximum(G.n64, 1)
    gain = np.exp(ref - own)
    gain[0] = 1.0
    return np.clip(np.real(np.fft.ifft2(F * gain[G.b64])), 0, 255)


def energymatch_img(x, refE, G):
    F = np.fft.fft2(x)
    E = np.bincount(G.b64.ravel(), (np.abs(F) ** 2).ravel(), minlength=NB64)
    gain = np.sqrt(refE / np.maximum(E, 1e-12))
    gain[0] = 1.0
    return np.clip(np.real(np.fft.ifft2(F * gain[G.b64])), 0, 255)


def feat_img(x, G):
    return band_feat(np.abs(np.fft.fft2(x)) ** 2, G)[0]


def chain_feat(x_notched, ref, G):
    """uint8 save -> specmatch (mode 1 twin profile) -> uint8 save -> f_fourier."""
    return feat_img(q8(specmatch_img(q8(x_notched), ref, G)), G)


def ridge_fix(mag, G):
    """Replace the fx=0 (3 columns) and fy=0 (3 rows) ridge bins above rmin by the mean magnitude
    of the bins 3..6 away on the same row / column."""
    nm = mag.copy()
    h, w = mag.shape
    cols = [c % w for c in list(range(-6, -2)) + list(range(3, 7))]
    rowmean = mag[:, cols].mean(1)
    m = G.col0
    nm[m] = np.broadcast_to(rowmean[:, None], mag.shape)[m]
    rows = [r % h for r in list(range(-6, -2)) + list(range(3, 7))]
    colmean = mag[rows, :].mean(0)
    m = G.row0 & ~G.col0
    nm[m] = np.broadcast_to(colmean[None, :], mag.shape)[m]
    return nm


def analyse(task):
    path, kind, ref_prof, ref_E_twin, ref_E_mean, seed = task
    x = load_gray(path)
    G = geo(x.shape)
    F = np.fft.fft2(x)
    mag = np.abs(F)
    ph = np.angle(F)
    P = mag ** 2
    logm = np.log1p(mag)
    res = {}
    feat, E30 = band_feat(P, G)
    res['feat'] = feat
    res['E30'] = E30
    res['E64'] = np.bincount(G.b64.ravel(), P.ravel(), minlength=NB64)
    res['prof64'] = np.bincount(G.b64.ravel(), logm.ravel(), minlength=NB64) / np.maximum(G.n64, 1)
    res['pk'] = peakiness_of(logm, G)[0]
    rm64 = ring_median(logm, G)
    rm = rm64[G.b64]
    med7 = median_filter(logm, size=7, mode='wrap')
    exc = logm - med7
    abv = G.abv
    # flagged counts / energy shares per t (7x7), lower tail
    cnt = np.zeros((len(T_LIST), NB))
    esh = np.zeros((len(T_LIST), NB))
    pk_t = np.zeros((len(T_LIST), NB))
    for ti, t in enumerate(T_LIST):
        m = (exc > t) & abv
        cnt[ti] = np.bincount(G.b30.ravel(), m.ravel(), minlength=NB)
        esh[ti] = np.bincount(G.b30.ravel(), (P * m).ravel(), minlength=NB) / np.maximum(E30, 1e-12)
        pk_t[ti] = peakiness_of(np.where(m, med7, logm), G)[0]
    res['cnt7'], res['esh7'], res['pk_t7'] = cnt, esh, pk_t
    low = np.zeros((len(V_LIST), NB))
    for vi, v in enumerate(V_LIST):
        low[vi] = np.bincount(G.b30.ravel(), ((exc < v) & abv).ravel(), minlength=NB) / G.n30
    res['low7'] = low
    # quantiles of log|F| relative to the local median and to the ring median
    res['q_loc'] = band_quantiles(exc, G)
    res['q_ring'] = band_quantiles(logm - rm, G)
    # peakiness decomposition
    res['pk_fill'] = peakiness_of(np.where(abv, np.maximum(logm, med7), logm), G)[0]
    res['pk_med'] = peakiness_of(np.where(abv, med7, logm), G)[0]
    res['pk_capR'] = peakiness_of(np.where(abv, np.minimum(logm, rm), logm), G)[0]
    res['pk_fillR'] = peakiness_of(np.where(abv, np.maximum(logm, rm), logm), G)[0]
    res['pk_cap_fill_R'] = peakiness_of(np.where(abv, np.clip(logm, rm - 1.0, rm + 1.0), logm), G)[0]
    # ridge / axis / sector energy shares
    res['col0_share'] = np.bincount(G.b30.ravel(), (P * G.col0).ravel(), minlength=NB) / np.maximum(E30, 1e-12)
    res['row0_share'] = np.bincount(G.b30.ravel(), (P * G.row0).ravel(), minlength=NB) / np.maximum(E30, 1e-12)
    res['col0_exc'] = np.bincount(G.b30.ravel(), (exc * G.col0).ravel(), minlength=NB) / np.maximum(
        np.bincount(G.b30.ravel(), G.col0.ravel(), minlength=NB), 1)
    res['col0_flag'] = np.array([(((exc > t) & G.col0).sum() / max(G.col0.sum(), 1)) for t in T_LIST])
    sec = np.bincount((G.b30 * NSEC + G.sec).ravel(), P.ravel(), minlength=NB * NSEC).reshape(NB, NSEC)
    res['sec_share'] = sec / np.maximum(E30[:, None], 1e-12)
    # cross-sections: fx=0 ridge across fx (rows |fy| in [0.33,0.5] and [0.15,0.33)), fy=0 across fy
    cols = [(o) % G.w for o in G.offs]
    res['xs_col0_hi'] = logm[G.ridge_rows][:, cols].mean(0)
    res['xs_col0_mid'] = logm[G.mid_rows][:, cols].mean(0)
    rows = [(o) % G.h for o in G.offs]
    res['xs_row0_hi'] = logm[rows][:, G.ridge_cols].mean(1)
    # diagonal peak (fy=+0.25, fx=-0.25) neighbourhood of exc
    cy, cx = int(round(0.25 * G.h)), (-int(round(0.25 * G.w))) % G.w
    res['diag_nb'] = exc[np.ix_([(cy + o) % G.h for o in G.offs], [(cx + o) % G.w for o in G.offs])]
    # mean 2-D peak shape: strongest off-axis peaks (exc > 1.5), neighbourhood of logm minus the centre median
    cand = np.where((exc > 1.5) & G.offaxis)
    if len(cand[0]):
        order = np.argsort(exc[cand])[::-1][:30]
        acc = np.zeros((2 * HALF + 1, 2 * HALF + 1))
        for i, j in zip(cand[0][order], cand[1][order]):
            acc += logm[np.ix_([(i + o) % G.h for o in G.offs], [(j + o) % G.w for o in G.offs])] - med7[i, j]
        res['peak_shape'] = acc / len(order)
        res['peak_n'] = len(order)
    else:
        res['peak_shape'] = np.zeros((2 * HALF + 1, 2 * HALF + 1))
        res['peak_n'] = 0
    if kind not in ('synth', 'pai'):
        return res
    # ---- windows (synth and real PAI) -------------------------------------------------------
    meds = {7: med7}
    for w in WIN_LIST:
        if w != 7:
            meds[w] = median_filter(logm, size=w, mode='wrap')
    m11 = ((logm - meds[11]) > 1.0) & abv
    cntw = np.zeros((len(WIN_LIST), len(T_LIST), NB))
    eshw = np.zeros((len(WIN_LIST), len(T_LIST), NB))
    pkw = np.zeros((len(WIN_LIST), len(T_LIST), NB))
    bias = np.zeros((len(WIN_LIST), NB))
    bias_ridge = np.zeros((len(WIN_LIST), NB))
    exc_ridge_w = np.zeros((len(WIN_LIST), NB))
    flag_ridge_w = np.zeros((len(WIN_LIST), len(T_LIST)))
    n_ridge = np.maximum(np.bincount(G.b30.ravel(), G.col0.ravel(), minlength=NB), 1)
    n11 = np.maximum(np.bincount(G.b30.ravel(), m11.ravel(), minlength=NB), 1)
    for wi, w in enumerate(WIN_LIST):
        ew = logm - meds[w]
        for ti, t in enumerate(T_LIST):
            m = (ew > t) & abv
            cntw[wi, ti] = np.bincount(G.b30.ravel(), m.ravel(), minlength=NB)
            eshw[wi, ti] = np.bincount(G.b30.ravel(), (P * m).ravel(), minlength=NB) / np.maximum(E30, 1e-12)
            pkw[wi, ti] = peakiness_of(np.where(m, meds[w], logm), G)[0]
            flag_ridge_w[wi, ti] = ((ew > t) & G.col0).sum() / max(G.col0.sum(), 1)
        bias[wi] = np.bincount(G.b30.ravel(), ((meds[w] - meds[11]) * m11).ravel(), minlength=NB) / n11
        bias_ridge[wi] = np.bincount(G.b30.ravel(), ((meds[w] - meds[11]) * G.col0).ravel(), minlength=NB) / n_ridge
        exc_ridge_w[wi] = np.bincount(G.b30.ravel(), (ew * G.col0).ravel(), minlength=NB) / n_ridge
    res.update(cntw=cntw, eshw=eshw, pkw=pkw, bias=bias, bias_ridge=bias_ridge,
               exc_ridge_w=exc_ridge_w, flag_ridge_w=flag_ridge_w)
    if kind != 'synth':
        return res
    # ---- simulations (synth only) --------------------------------------------------------------
    sims = {}
    sims['sm_only'] = chain_feat(x, ref_prof, G)
    for t, w in SIM_GRID:
        sims[f'chain({t},{w})'] = chain_feat(img_from(notch_mag(mag, logm, meds[w], t, G), ph), ref_prof, G)
    nm10 = notch_mag(mag, logm, med7, 1.0, G)
    x10 = q8(img_from(nm10, ph))
    sims['cap_t0_w7'] = chain_feat(img_from(notch_mag(mag, logm, med7, 0.0, G), ph), ref_prof, G)
    sims['cap_t0_w11'] = chain_feat(img_from(notch_mag(mag, logm, meds[11], 0.0, G), ph), ref_prof, G)
    sims['notch10+ridge'] = chain_feat(img_from(ridge_fix(nm10, G), ph), ref_prof, G)
    for tr in (1.0, 0.5):
        mr = ((logm - rm) > tr) & abv
        nmr = mag.copy()
        nmr[mr] = np.expm1(rm[mr])
        sims[f'ringnotch{tr}'] = chain_feat(img_from(nmr, ph), ref_prof, G)
    for v in (0.5, 1.0):
        lm = np.log1p(nm10)
        lm2 = np.where(abv, np.maximum(lm, med7 - v), lm)
        sims[f'notch10+fill{v}'] = chain_feat(img_from(np.expm1(lm2), ph), ref_prof, G)
    lm2 = np.where(abv, np.clip(logm, rm - 1.0, rm + 1.0), logm)
    sims['ringclamp1.0'] = chain_feat(img_from(np.expm1(lm2), ph), ref_prof, G)
    rng = np.random.default_rng(seed)
    for sg in (2.0, 4.0):
        sims[f'notch10+noise{sg:.0f}'] = chain_feat(x10 + rng.normal(0, sg, x.shape), ref_prof, G)
    sims['notch10+Ematch_twin'] = feat_img(q8(energymatch_img(x10, ref_E_twin, G)), G)
    sims['notch10+Ematch_mean'] = feat_img(q8(energymatch_img(x10, ref_E_mean, G)), G)
    sims['notch10+sm+Ematch_twin'] = feat_img(q8(energymatch_img(q8(specmatch_img(x10, ref_prof, G)), ref_E_twin, G)), G)
    res['sims'] = sims
    return res


def stack(results, key):
    return np.array([r[key] for r in results])


def knn_excl(Xb, Xv, Xt, excl, k=K):
    """knn_split with the twin outputs of each test row excluded from its candidate neighbours."""
    Xtr = np.vstack([Xb, Xv])
    ytr = np.r_[np.zeros(len(Xb)), np.ones(len(Xv))]
    D = np.sqrt(((Xt[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    if excl is not None:
        for i, ex in enumerate(excl):
            D[i, len(Xb) + np.asarray(ex, int)] = np.inf
    nn = np.argsort(D, axis=1)[:, :k]
    return 1.0 - ((ytr[nn].mean(1) > 0.5).astype(int)).mean()


def main():
    t0 = time.time()
    lines = []

    def say(s=''):
        lines.append(s)
        print(s)

    # specmatch reference (mode 1): all real PAI, sorted, rng(0) draws in list_images order
    ref_files = list_images(POPS['pai'], 500)
    rng = np.random.default_rng(0)
    synth_files = list_images(POPS['synth'])
    twin = np.array([rng.integers(len(ref_files)) for _ in synth_files])
    synth_sub = list_images(POPS['synth'], LIMIT)
    sub_idx = [synth_files.index(p) for p in synth_sub]
    twin_sub = twin[sub_idx]
    G = geo(load_gray(ref_files[0]).shape)
    ref_prof = np.zeros((len(ref_files), NB64))
    ref_E = np.zeros((len(ref_files), NB64))
    for i, p in enumerate(ref_files):
        F = np.fft.fft2(load_gray(p))
        ref_prof[i] = np.bincount(G.b64.ravel(), np.log1p(np.abs(F)).ravel(), minlength=NB64) / np.maximum(G.n64, 1)
        ref_E[i] = np.bincount(G.b64.ravel(), (np.abs(F) ** 2).ravel(), minlength=NB64)
    ref_E_mean = ref_E.mean(0)
    pai_sub = list_images(POPS['pai'], LIMIT)
    pai_full_idx = np.array([ref_files.index(p) for p in pai_sub])

    tasks, owners = [], []
    for name, folder in POPS.items():
        files = list_images(folder, LIMIT)
        for i, p in enumerate(files):
            if name == 'synth':
                tw = twin_sub[i]
                tasks.append((p, 'synth', ref_prof[tw], ref_E[tw], ref_E_mean, 1000 + i))
            else:
                tasks.append((p, name if name == 'pai' else 'other', None, None, None, 0))
            owners.append(name)
    say(f'{len(tasks)} images in {len(POPS)} populations, limit {LIMIT}, image shape {G.h}x{G.w}, '
        f'{G.h * G.w} bins; ref profiles {len(ref_files)}')
    try:
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            out = list(ex.map(analyse, tasks, chunksize=8))
    except Exception as e:                                     # fall back to serial
        say(f'pool failed ({e!r}), running serially')
        out = [analyse(t) for t in tasks]
    R = {n: [o for o, w in zip(out, owners) if w == n] for n in POPS}
    say(f'per-image pass done in {time.time() - t0:.0f} s')

    edges = band_edges(NB)
    feats = {n: stack(R[n], 'feat') for n in POPS}
    mu = {n: f.mean(0) for n, f in feats.items()}
    sd = {n: f.std(0, ddof=1) for n, f in feats.items()}
    d_pai = {v: cohen_d(feats[v], feats['pai']) for v in VARIANTS}
    d_bona = {v: cohen_d(feats[v], feats['bona']) for v in VARIANTS}
    d_bp = cohen_d(feats['bona'], feats['pai'])
    sq_pai = {v: (mu[v] - mu['pai']) ** 2 for v in VARIANTS}
    esh10 = {n: stack(R[n], 'esh7')[:, T_LIST.index(1.0), :].mean(0) for n in POPS}
    cnt10 = {n: stack(R[n], 'cnt7')[:, T_LIST.index(1.0), :].mean(0) for n in POPS}
    pk = {n: stack(R[n], 'pk').mean(0) for n in POPS}

    # twin replication check on the actual sm set
    prof_sm = stack(R['sm'], 'prof64')
    sm_files = list_images(POPS['sm'], LIMIT)
    sm_all = list_images(POPS['sm'])
    sm_idx = [sm_all.index(p) for p in sm_files]
    d_tw = np.abs(prof_sm[:, 1:] - ref_prof[twin[sm_idx]][:, 1:]).mean()
    d_rnd = np.abs(prof_sm[:, 1:] - ref_prof[(twin[sm_idx] + 7) % len(ref_files)][:, 1:]).mean()
    say(f'twin check (sm set): mean |profile - twin profile| {d_tw:.4f} vs non-twin {d_rnd:.4f}')

    # ===================================================================== task 1
    say()
    say('=' * 110)
    say('TASK 1  Band table (f_fourier, 30 bands). Means per population; then Cohen d vs real PAI / vs bona fide;')
    say('then flagged-peak energy share (re-detected with t=1.0, 7x7) and the share of the squared centroid')
    say('distance to real PAI carried by the band. Full table incl. sd in band_table2.csv.')
    say('=' * 110)
    cols = ['bona', 'pai', 'synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9', 'sm',
            'pk0.6_s7_sm', 'pk1.0_s7_sm', 'pk0.8_s5_sm', 'pk1.0_s5_sm', 'pk0.8_s9_sm', 'pk1.0_s9_sm']
    say('--- means (log normalised band energy) ---')
    say(f"{'b':>2s} {'lo':>5s} {'hi':>5s} | " + ' '.join(f'{c:>11s}' for c in cols))
    for b in range(NB):
        say(f'{b:2d} {edges[b][0]:5.3f} {edges[b][1]:5.3f} | ' + ' '.join(f'{mu[c][b]:11.2f}' for c in cols))
    vcols = cols[2:]
    say('--- Cohen d (variant - real PAI) ---')
    say(f"{'b':>2s} {'d_bo-pai':>8s} | " + ' '.join(f'{c:>11s}' for c in vcols))
    for b in range(NB):
        say(f'{b:2d} {d_bp[b]:8.2f} | ' + ' '.join(f'{d_pai[c][b]:11.2f}' for c in vcols))
    say('--- Cohen d (variant - bona fide) ---')
    say(f"{'b':>2s} {'':>8s} | " + ' '.join(f'{c:>11s}' for c in vcols))
    for b in range(NB):
        say(f'{b:2d} {"":8s} | ' + ' '.join(f'{d_bona[c][b]:11.2f}' for c in vcols))
    say('--- flagged-peak energy share (fraction of band |F|^2 in bins > 1.0 above the 7x7 local median) ---')
    say(f"{'b':>2s} | " + ' '.join(f'{c:>11s}' for c in cols))
    for b in range(4, NB):
        say(f'{b:2d} | ' + ' '.join(f'{esh10[c][b]:11.3f}' for c in cols))
    say('--- share (%) of the squared centroid distance variant -> real PAI carried by the band ---')
    say(f"{'b':>2s} | " + ' '.join(f'{c:>11s}' for c in vcols))
    for b in range(NB):
        say(f'{b:2d} | ' + ' '.join(f'{100 * sq_pai[c][b] / sq_pai[c].sum():11.1f}' for c in vcols))
    say('centroid distance to real PAI / to bona fide, and mean excess over bands 13-21 (nats):')
    for c in vcols:
        say(f'  {c:12s} d_pai {np.sqrt(sq_pai[c].sum()):6.3f}  d_bona {np.sqrt(((mu[c] - mu["bona"]) ** 2).sum()):6.3f}  '
            f'b13-21 excess vs PAI {np.mean([mu[c][b] - mu["pai"][b] for b in MID]):+.3f}  '
            f'b22-29 deficit {np.mean([mu[c][b] - mu["pai"][b] for b in range(22, 30)]):+.3f}  '
            f'n|d|>0.8 vs PAI {(np.abs(d_pai[c]) > 0.8).sum():2d}')
    with open(os.path.join(OUT, 'band_table2.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['band', 'lo_cpp', 'hi_cpp'] + [f'mean_{n}' for n in POPS] + [f'sd_{n}' for n in POPS]
                   + ['d_bona_vs_pai'] + [f'd_pai_{v}' for v in VARIANTS] + [f'd_bona_{v}' for v in VARIANTS]
                   + [f'peakshare_t1.0_{n}' for n in POPS] + [f'npeakbins_t1.0_{n}' for n in POPS]
                   + [f'sqdist_share_pai_{v}' for v in VARIANTS] + [f'peakiness_{n}' for n in POPS])
        for b in range(NB):
            w.writerow([b, f'{edges[b][0]:.4f}', f'{edges[b][1]:.4f}'] + [f'{mu[n][b]:.4f}' for n in POPS]
                       + [f'{sd[n][b]:.4f}' for n in POPS] + [f'{d_bp[b]:.3f}']
                       + [f'{d_pai[v][b]:.3f}' for v in VARIANTS] + [f'{d_bona[v][b]:.3f}' for v in VARIANTS]
                       + [f'{esh10[n][b]:.4f}' for n in POPS] + [f'{cnt10[n][b]:.2f}' for n in POPS]
                       + [f'{sq_pai[v][b] / sq_pai[v].sum():.4f}' for v in VARIANTS] + [f'{pk[n][b]:.3f}' for n in POPS])

    # ===================================================================== task 2
    say()
    say('=' * 110)
    say('TASK 2  Why lowering t below 1.0 no longer closes bands 13-21')
    say('=' * 110)
    say('Feature identity per band: log E = 2*mean(log|F|) + peakiness. Specmatch pins the first term to the')
    say('twin real-PAI profile, so the chain residual = peakiness(after notch) - peakiness(real PAI).')
    say()
    say('(2a) Untouched synthetic, 7x7 window: flagged bins and flagged energy share per band as a function of t')
    cnt7 = stack(R['synth'], 'cnt7').mean(0)
    esh7 = stack(R['synth'], 'esh7').mean(0)
    pkt7 = stack(R['synth'], 'pk_t7').mean(0)
    say(f"{'band':>4s} {'nbins':>6s} | " + ' '.join(f'n>{t:<4.1f}' for t in T_LIST) + ' | ' + ' '.join(f'E>{t:<4.1f}' for t in T_LIST))
    for b in list(range(10, 23)):
        say(f'{b:4d} {G.n30[b]:6d} | ' + ' '.join(f'{cnt7[ti, b]:6.0f}' for ti in range(len(T_LIST)))
            + ' | ' + ' '.join(f'{esh7[ti, b]:6.3f}' for ti in range(len(T_LIST))))
    say()
    say('(2b) Peakiness of the notched map vs t (untouched synthetic, 7x7), against real PAI and bona fide peakiness.')
    say('     t=0 is the limit "every bin above its local median pulled down". Residual = column - PAI.')
    say(f"{'band':>4s} {'PAI':>6s} {'bona':>6s} {'synth':>6s} | " + ' '.join(f't={t:<4.1f}' for t in T_LIST) + ' | resid@t=0')
    for b in list(range(10, 23)):
        say(f"{b:4d} {pk['pai'][b]:6.2f} {pk['bona'][b]:6.2f} {pk['synth'][b]:6.2f} | "
            + ' '.join(f'{pkt7[ti, b]:6.2f}' for ti in range(len(T_LIST))) + f" | {pkt7[0, b] - pk['pai'][b]:+6.2f}")
    say(f"  mean over b13-21: PAI {np.mean([pk['pai'][b] for b in MID]):.2f}, synth {np.mean([pk['synth'][b] for b in MID]):.2f}, "
        + ', '.join(f't={t}: {np.mean([pkt7[ti, b] for b in MID]):.2f}' for ti, t in enumerate(T_LIST)))
    say()
    say('(2c) Peakiness of the ACTUAL sets (what the k-NN residual is made of), bands 10-22:')
    pcols = ['pai', 'bona', 'synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9', 'pk0.6_s7_sm', 'pk1.0_s7_sm', 'pk0.8_s5_sm', 'pk0.8_s9_sm', 'pk1.0_s9_sm']
    say(f"{'band':>4s} | " + ' '.join(f'{c:>11s}' for c in pcols))
    for b in range(10, 23):
        say(f'{b:4d} | ' + ' '.join(f'{pk[c][b]:11.2f}' for c in pcols))
    say('  mean b13-21: ' + ', '.join(f'{c} {np.mean([pk[c][b] for b in MID]):.2f}' for c in pcols))
    say()
    say('(2d) Peakiness decomposition (mean over bands 13-21, nats). cap0 = all bins above the 7x7 median pulled')
    say('     to it (t=0 notch); fill = bins below the 7x7 median raised to it; med = the 7x7 median surface alone;')
    say('     capR / fillR = same with the 64-ring median as reference; clampR = ring median +-1.0.')
    say(f"{'pop':12s} {'raw':>6s} {'cap0':>6s} {'fill':>6s} {'med':>6s} {'capR':>6s} {'fillR':>6s} {'clampR':>6s} | raw-cap0 raw-fill raw-med")
    dec = {}
    for c in ['pai', 'bona', 'synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9', 'pk1.0_s7_sm', 'pk0.6_s7_sm']:
        raw = np.mean([pk[c][b] for b in MID])
        cap0 = np.mean([stack(R[c], 'pk_t7')[:, 0, b].mean() for b in MID])
        fil = np.mean([stack(R[c], 'pk_fill').mean(0)[b] for b in MID])
        med = np.mean([stack(R[c], 'pk_med').mean(0)[b] for b in MID])
        capr = np.mean([stack(R[c], 'pk_capR').mean(0)[b] for b in MID])
        filr = np.mean([stack(R[c], 'pk_fillR').mean(0)[b] for b in MID])
        clr = np.mean([stack(R[c], 'pk_cap_fill_R').mean(0)[b] for b in MID])
        dec[c] = (raw, cap0, fil, med, capr, filr, clr)
        say(f'{c:12s} {raw:6.2f} {cap0:6.2f} {fil:6.2f} {med:6.2f} {capr:6.2f} {filr:6.2f} {clr:6.2f} | {raw - cap0:8.2f} {raw - fil:8.2f} {raw - med:7.2f}')
    say()
    say('(2e) Within-ring spread: quantiles of log1p|F| minus the 7x7 local median (left) and minus the 64-ring')
    say('     median (right), pooled over bands 13-21 (mean of per-band quantiles).')
    say(f"{'pop':12s} | " + ' '.join(f'p{q:<3d}' for q in Q_LIST) + ' | ' + ' '.join(f'p{q:<3d}' for q in Q_LIST) + ' | frac<-0.5 frac<-1.0 frac<-1.5 (local)')
    for c in ['pai', 'bona', 'synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9', 'sm', 'pk1.0_s7_sm', 'pk0.6_s7_sm', 'pk1.0_s9_sm']:
        ql = stack(R[c], 'q_loc').mean(0)[:, MID].mean(1)
        qr = stack(R[c], 'q_ring').mean(0)[:, MID].mean(1)
        lo = stack(R[c], 'low7').mean(0)[:, MID].mean(1)
        say(f'{c:12s} | ' + ' '.join(f'{v:4.2f}' for v in ql) + ' | ' + ' '.join(f'{v:4.2f}' for v in qr)
            + ' | ' + ' '.join(f'{v:9.3f}' for v in lo))
    say()
    say('(2f) fx=0 ridge: energy share of the 3 columns |kx|<=1 (and of the 3 rows |ky|<=1) per band, mean 7x7 excess')
    say('     on the ridge, fraction of ridge bins flagged per t (untouched synthetic / real PAI):')
    say(f"{'band':>4s} | " + ' '.join(f'{c:>9s}' for c in ['col0 syn', 'col0 pai', 'col0 bona', 'col0 pk1.0', 'col0 pk0.6', 'col0 chain', 'row0 syn', 'row0 pai']) + ' | exc syn  exc pai')
    for b in range(10, 23):
        vals = [stack(R[c], 'col0_share').mean(0)[b] for c in ['synth', 'pai', 'bona', 'pk1.0_s7', 'pk0.6_s7', 'pk1.0_s7_sm']]
        vals += [stack(R[c], 'row0_share').mean(0)[b] for c in ['synth', 'pai']]
        say(f'{b:4d} | ' + ' '.join(f'{v:9.3f}' for v in vals)
            + f" | {stack(R['synth'], 'col0_exc').mean(0)[b]:7.2f} {stack(R['pai'], 'col0_exc').mean(0)[b]:7.2f}")
    for c in ['synth', 'pai', 'pk0.6_s7', 'pk1.0_s7']:
        say(f'  fraction of fx=0 ridge bins (r>0.1) flagged, {c:9s}: ' + ' '.join(f't={t}:{v:.2f}' for t, v in zip(T_LIST, stack(R[c], 'col0_flag').mean(0))))
    say('  ridge cross-section: mean log1p|F| at fx offsets -8..8 bins (rows |fy| in [0.33,0.5]):')
    for c in ['synth', 'pai', 'bona', 'pk1.0_s7', 'pk0.6_s7']:
        say(f'    {c:9s} ' + ' '.join(f'{v:5.2f}' for v in stack(R[c], 'xs_col0_hi').mean(0)))
    say('  same for rows |fy| in [0.15,0.33):')
    for c in ['synth', 'pai']:
        say(f'    {c:9s} ' + ' '.join(f'{v:5.2f}' for v in stack(R[c], 'xs_col0_mid').mean(0)))
    say('  fy=0 ridge cross-section across fy (cols |fx| in [0.33,0.5]):')
    for c in ['synth', 'pai']:
        say(f'    {c:9s} ' + ' '.join(f'{v:5.2f}' for v in stack(R[c], 'xs_row0_hi').mean(0)))
    say('  diagonal peak (fy=+0.25, fx=-0.25): mean 7x7 excess, central row (along fx) and central column (along fy), offsets -8..8:')
    for c in ['synth', 'pk1.0_s7', 'pk0.6_s7', 'pai']:
        nb = stack(R[c], 'diag_nb').mean(0)
        say(f'    {c:9s} along fx: ' + ' '.join(f'{v:5.2f}' for v in nb[HALF, :]))
        say(f'    {c:9s} along fy: ' + ' '.join(f'{v:5.2f}' for v in nb[:, HALF]))
    say()
    say('(2g) Anisotropy: energy share per 15-degree sector (angle from the fx axis, folded to 0..180),')
    say('     pooled over bands 13-21. Sector 0 = along fx (vertical stripes), sector 6 = along fy (horizontal stripes).')
    say(f"{'pop':12s} | " + ' '.join(f's{i:<5d}' for i in range(NSEC)) + ' | std over sectors')
    for c in ['pai', 'bona', 'synth', 'pk1.0_s7', 'pk0.6_s7', 'pk1.0_s7_sm', 'pk0.6_s7_sm', 'pk0.8_s5_sm', 'pk1.0_s9_sm']:
        ss = stack(R[c], 'sec_share').mean(0)[MID].mean(0)
        say(f'{c:12s} | ' + ' '.join(f'{v:6.3f}' for v in ss) + f' | {ss.std():.3f}')

    # ===================================================================== task 3
    say()
    say('=' * 110)
    say('TASK 3  Window size: 5x5 much worse, 9x9 slightly better')
    say('=' * 110)
    cw = stack(R['synth'], 'cntw').mean(0)
    ew = stack(R['synth'], 'eshw').mean(0)
    pw = stack(R['synth'], 'pkw').mean(0)
    say('(3a) Untouched synthetic: flagged energy share (bands 13-21 mean) and flagged bins per window and t:')
    say(f"{'window':>6s} | " + ' '.join(f'E>{t:<4.1f}' for t in T_LIST) + ' | ' + ' '.join(f'n>{t:<4.1f}' for t in T_LIST))
    for wi, w in enumerate(WIN_LIST):
        say(f'{w:6d} | ' + ' '.join(f'{ew[wi, ti, MID].mean():6.3f}' for ti in range(len(T_LIST)))
            + ' | ' + ' '.join(f'{cw[wi, ti, MID].sum():6.0f}' for ti in range(len(T_LIST))))
    say('(3b) Peakiness after the notch (bands 13-21 mean) per window and t; real PAI = '
        f"{np.mean([pk['pai'][b] for b in MID]):.2f}, bona = {np.mean([pk['bona'][b] for b in MID]):.2f}:")
    say(f"{'window':>6s} | " + ' '.join(f't={t:<4.1f}' for t in T_LIST))
    for wi, w in enumerate(WIN_LIST):
        say(f'{w:6d} | ' + ' '.join(f'{pw[wi, ti, MID].mean():6.2f}' for ti in range(len(T_LIST))))
    say('(3c) Window bias: mean (med_w - med_11) at bins flagged by the 11x11 median at t=1.0, and on the fx=0 ridge,')
    say('     and the mean excess (logm - med_w) on the ridge, bands 13-21 mean (untouched synthetic | real PAI):')
    bs, br, er = stack(R['synth'], 'bias').mean(0), stack(R['synth'], 'bias_ridge').mean(0), stack(R['synth'], 'exc_ridge_w').mean(0)
    bsp, brp, erp = stack(R['pai'], 'bias').mean(0), stack(R['pai'], 'bias_ridge').mean(0), stack(R['pai'], 'exc_ridge_w').mean(0)
    fr, frp = stack(R['synth'], 'flag_ridge_w').mean(0), stack(R['pai'], 'flag_ridge_w').mean(0)
    say(f"{'window':>6s} | {'bias@peaks':>10s} {'bias@ridge':>10s} {'exc@ridge':>10s} | {'PAI bias@peaks':>14s} {'PAI exc@ridge':>13s} | ridge flagged t=0.6/0.8/1.0 (synth) | (pai)")
    for wi, w in enumerate(WIN_LIST):
        say(f'{w:6d} | {bs[wi, MID].mean():10.3f} {br[wi, MID].mean():10.3f} {er[wi, MID].mean():10.3f} | '
            f'{bsp[wi, MID].mean():14.3f} {erp[wi, MID].mean():13.3f} | '
            + ' '.join(f'{fr[wi, T_LIST.index(t)]:.2f}' for t in (0.6, 0.8, 1.0)) + '  | '
            + ' '.join(f'{frp[wi, T_LIST.index(t)]:.2f}' for t in (0.6, 0.8, 1.0)))
    say('(3d) Mean 2-D shape of the strongest off-axis peaks (excess > 1.5, 7x7): log1p|F| minus the centre median,')
    say('     central row (along fx, 1 bin = 1/736 cpp) and central column (along fy, 1 bin = 1/192 cpp), offsets -8..8:')
    for c in ['synth', 'pai', 'bona']:
        shp = stack(R[c], 'peak_shape')
        n = stack(R[c], 'peak_n')
        m = (shp * n[:, None, None]).sum(0) / max(n.sum(), 1)
        say(f'    {c:6s} (peaks/img {n.mean():5.1f}) along fx: ' + ' '.join(f'{v:5.2f}' for v in m[HALF, :]))
        say(f'    {c:6s} {"":17s} along fy: ' + ' '.join(f'{v:5.2f}' for v in m[:, HALF]))
    say('(3e) Notch-only sets: band mean (nats) relative to real PAI and residual flagged share (t=1.0, 7x7 re-detection):')
    say(f"{'band':>4s} | " + ' '.join(f'{c:>10s}' for c in ['synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9']) + ' | ' + ' '.join(f'{c:>10s}' for c in ['synth', 'pk0.6_s7', 'pk1.0_s7', 'pk0.8_s5', 'pk0.8_s9']))
    for b in range(10, 23):
        say(f'{b:4d} | ' + ' '.join(f"{mu[c][b] - mu['pai'][b]:10.2f}" for c in NOTCH_ONLY) + ' | ' + ' '.join(f'{esh10[c][b]:10.3f}' for c in NOTCH_ONLY))

    # ===================================================================== task 4 / k-NN
    say()
    say('=' * 110)
    say('TASK 4  k-NN replica (k=15, train bona(n) vs variant(n), test real PAI(n)); "noleak" excludes the chain')
    say('outputs whose specmatch twin is the test image. Centroid distance to real PAI; b13-21 / b22-29 excess.')
    say('=' * 110)
    twin_of = {}
    for c in CHAINS:
        files_c = list_images(POPS[c], LIMIT)
        all_c = list_images(POPS[c])
        twin_of[c] = twin[[all_c.index(p) for p in files_c]]
    excl_for = lambda tw: [np.where(tw == pai_full_idx[i])[0] for i in range(len(pai_full_idx))]

    def report_row(name, X, tw=None, fix=False):
        ap, db, dv = knn_split(feats['bona'], X, feats['pai'])
        apn = knn_excl(feats['bona'], X, feats['pai'], excl_for(tw)) if tw is not None else ap
        m = X.mean(0)
        row = dict(name=name, apcer=100 * ap, apcer_noleak=100 * apn, d_bona=db, d_var=dv,
                   centroid=np.sqrt(((m - mu['pai']) ** 2).sum()),
                   mid=np.mean([m[b] - mu['pai'][b] for b in MID]),
                   hi=np.mean([m[b] - mu['pai'][b] for b in range(22, 30)]),
                   n_big=int((np.abs(cohen_d(X, feats['pai'])) > 0.8).sum()))
        say(f"  {name:26s} APCER {row['apcer']:5.1f} %  noleak {row['apcer_noleak']:5.1f} %  dist15 bona {db:.3f} var {dv:.3f}  "
            f"centroid {row['centroid']:.3f}  b13-21 {row['mid']:+.2f}  b22-29 {row['hi']:+.2f}  n|d|>0.8 {row['n_big']:2d}")
        return row

    rows = []
    say('actual sets:')
    for c in VARIANTS:
        rows.append(report_row(c, feats[c], twin_of.get(c)))
    ap0, db0, dv0 = knn_split(feats['bona'], feats['pai'][:len(feats['pai']) // 2], feats['pai'][len(feats['pai']) // 2:])
    say(f'  (sanity: train on half the real PAI, test the other half) APCER {100 * ap0:5.1f} %  dist bona {db0:.3f} pai {dv0:.3f}')
    say('actual chain sets + population-level fixed per-band gain g_b = exp(mu_pai - mu_set) (30 numbers, keeps spread):')
    for c in ['pk0.6_s7_sm', 'pk1.0_s7_sm', 'pk0.8_s9_sm', 'pk1.0_s9_sm', 'pk0.8_s5_sm']:
        E = np.exp(feats[c]) * np.exp(mu['pai'] - mu[c])
        rows.append(report_row(c + '+fixgain', np.log(E / E.sum(1, keepdims=True) + 1e-12), twin_of[c]))
    say('simulated on the untouched synthetic (same twin draws as the actual chain):')
    sims = {k: np.array([r['sims'][k] for r in R['synth']]) for k in R['synth'][0]['sims']}
    for k, X in sims.items():
        rows.append(report_row('sim ' + k, X, twin_sub))
    say('replica validation (mean |sim - actual| per image over bands 1-29, and centroid distance):')
    for k, c in [('sm_only', 'sm'), ('chain(1.0,7)', 'pk1.0_s7_sm'), ('chain(0.6,7)', 'pk0.6_s7_sm'), ('chain(0.8,5)', 'pk0.8_s5_sm'),
                 ('chain(1.0,5)', 'pk1.0_s5_sm'), ('chain(0.8,9)', 'pk0.8_s9_sm'), ('chain(1.0,9)', 'pk1.0_s9_sm')]:
        say(f'  {k:14s} vs {c:12s} mean|diff| {np.abs(sims[k][:, 1:] - feats[c][:, 1:]).mean():.3f}  '
            f'centroid dist {np.sqrt(((sims[k].mean(0) - mu[c]) ** 2).sum()):.3f}  '
            f'b13-21 sim {np.mean([sims[k].mean(0)[b] - mu["pai"][b] for b in MID]):+.2f} actual {np.mean([mu[c][b] - mu["pai"][b] for b in MID]):+.2f}')
    say()
    say('k-NN replica restricted to band subsets (train bona vs variant, test real PAI), noleak:')
    subsets = {'b0-4': range(0, 5), 'b5-9': range(5, 10), 'b10-14': range(10, 15), 'b15-19': range(15, 20),
               'b20-24': range(20, 25), 'b25-29': range(25, 30), 'b13-21': range(13, 22), 'b1-29': range(1, 30)}
    say(f"  {'variant':26s} " + ' '.join(f'{k:>7s}' for k in subsets))
    for c in ['pk0.6_s7_sm', 'pk1.0_s7_sm', 'pk0.8_s5_sm', 'pk0.8_s9_sm', 'pk1.0_s9_sm']:
        r_ = [knn_excl(feats['bona'][:, list(ix)], feats[c][:, list(ix)], feats['pai'][:, list(ix)], excl_for(twin_of[c])) for ix in subsets.values()]
        say(f'  {c:26s} ' + ' '.join(f'{100 * x:6.1f}%' for x in r_))
    for k in ['notch10+Ematch_twin', 'notch10+ridge', 'ringnotch1.0', 'notch10+fill1.0', 'notch10+noise4']:
        r_ = [knn_excl(feats['bona'][:, list(ix)], sims[k][:, list(ix)], feats['pai'][:, list(ix)], excl_for(twin_sub)) for ix in subsets.values()]
        say(f'  {"sim " + k:26s} ' + ' '.join(f'{100 * x:6.1f}%' for x in r_))
    say()
    say('per-band mean excess vs real PAI (nats) of selected simulations, bands 10-29:')
    skeys = ['chain(1.0,7)', 'chain(0.6,7)', 'cap_t0_w7', 'cap_t0_w11', 'chain(1.0,5)', 'chain(1.0,9)', 'notch10+ridge', 'ringnotch1.0',
             'notch10+fill0.5', 'notch10+fill1.0', 'ringclamp1.0', 'notch10+noise2', 'notch10+noise4', 'notch10+Ematch_twin', 'notch10+Ematch_mean']
    say(f"{'band':>4s} | " + ' '.join(f'{k[:12]:>12s}' for k in skeys))
    for b in range(10, 30):
        say(f'{b:4d} | ' + ' '.join(f"{sims[k].mean(0)[b] - mu['pai'][b]:12.2f}" for k in skeys))
    with open(os.path.join(OUT, 'sim_results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    say(f'\ntotal runtime {time.time() - t0:.0f} s')
    with open(os.path.join(OUT, 'report2.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'written: {OUT}: band_table2.csv, sim_results.csv, report2.txt')


if __name__ == '__main__':
    main()
