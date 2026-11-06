"""Per-band mechanism analysis of the notch / matching sweep on the Fourier k-NN feature.

Read-only on the images. Computes the 30-band f_fourier feature of pad_knn.py for seven
populations, per-band means, Cohen's d of every synthetic variant against real PAI and against
bona fide, the share of each band in the squared centroid distance (what the Euclidean k-NN
actually sees), a single-split k-NN replication, and where the periodic peaks of the untouched
synthetic set sit (64-bin radial log-amplitude profile and the 2-D notch mask at t=1.0).

Run from this folder:  python -I band_analysis.py
Outputs (same folder): band_table.csv, band_table.md, peaks.csv, report.txt
"""
import csv
import os
import sys
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, r'D:\Studium\Master\MultimediaSecurity\PS\tools')
from common import list_images, load_gray, radial_log_amplitude, norm_radius   # noqa: E402
from pad_knn import f_fourier                                                     # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
S = r'D:\Studium\Master\MultimediaSecurity\PS\smoke_subset'
POPS = {
    'bona':       os.path.join(S, 'PLUS', 'real'),
    'pai':        os.path.join(S, 'PLUS', 'spoof'),
    'synth':      os.path.join(S, 'AE_Res', 'PLUS_spoof'),
    'pk1.0':      os.path.join(S, 'AE_Res', 'pk1.0_s7'),
    'sm':         os.path.join(S, 'AE_Res', 'sm'),
    'pk1.0_sm':   os.path.join(S, 'AE_Res', 'pk1.0_s7_sm'),
    'pk2.0_sm':   os.path.join(S, 'AE_Res', 'pk2.0_s7_sm'),
}
VARIANTS = ['synth', 'pk1.0', 'sm', 'pk1.0_sm', 'pk2.0_sm']
NB, NB64, LIMIT, K = 30, 64, 120, 15
T_NOTCH, SIZE_NOTCH, RMIN = 1.0, 7, 0.1


def band_edges(nb):
    """Normalised-frequency edges (cycles/px) of the nb bands of f_fourier."""
    w = 0.5 * np.sqrt(2) / nb
    return np.array([(i * w, (i + 1) * w) for i in range(nb)])


def peak_stats(x, t=T_NOTCH, size=SIZE_NOTCH, rmin=RMIN):
    """Same mask as apply_removal.op_peaks. Returns per-30-band: share of the band's |F|^2 that
    sits in flagged peak bins, number of flagged bins, the excess map (logm - local median),
    the 30-band index map and the radius map."""
    F = np.fft.fft2(x)
    mag = np.abs(F)
    P = mag ** 2
    logm = np.log1p(mag)
    med = median_filter(logm, size=size, mode='wrap')
    r = norm_radius(*x.shape)
    mask = ((logm - med) > t) & (r > rmin)
    b30 = np.minimum((r / (0.5 * np.sqrt(2)) * NB).astype(int), NB - 1)
    e_band = np.bincount(b30.ravel(), P.ravel(), minlength=NB)
    e_peak = np.bincount(b30.ravel(), (P * mask).ravel(), minlength=NB)
    n_peak = np.bincount(b30.ravel(), mask.ravel().astype(float), minlength=NB)
    return e_peak / np.maximum(e_band, 1e-12), n_peak, logm - med, b30, r


def cohen_d(a, b):
    sa, sb = a.std(axis=0, ddof=1), b.std(axis=0, ddof=1)
    sp = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(sp, 1e-12)


def knn_split(Xbona, Xvar, Xtest, k=K):
    """Single-split replica of pad_knn: train bona fide (0) vs variant (1), test real PAI.
    Returns APCER (fraction of real PAI called bona fide) and the mean distance of a test sample
    to its k nearest bona fide rows and to its k nearest variant rows."""
    Xtr = np.vstack([Xbona, Xvar])
    ytr = np.r_[np.zeros(len(Xbona)), np.ones(len(Xvar))]
    D = np.sqrt(((Xtest[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    nn = np.argsort(D, axis=1)[:, :k]
    yhat = (ytr[nn].mean(axis=1) > 0.5).astype(int)
    Db = np.sort(D[:, :len(Xbona)], axis=1)[:, :k].mean(1)
    Dv = np.sort(D[:, len(Xbona):], axis=1)[:, :k].mean(1)
    return 1.0 - yhat.mean(), Db.mean(), Dv.mean()


def peakiness(folder):
    """Per band: log(mean |F|^2) - 2*mean(log|F|). Large = energy concentrated in few bins."""
    out = []
    for p in list_images(folder, limit=LIMIT):
        x = load_gray(p)
        F = np.abs(np.fft.fft2(x))
        r = norm_radius(*x.shape)
        b = np.minimum((r / (0.5 * np.sqrt(2)) * NB).astype(int), NB - 1)
        e = np.bincount(b.ravel(), (F ** 2).ravel(), minlength=NB)
        n = np.bincount(b.ravel(), minlength=NB)
        ml = np.bincount(b.ravel(), np.log(F + 1e-9).ravel(), minlength=NB) / np.maximum(n, 1)
        out.append(np.log(e / np.maximum(n, 1)) - 2 * ml)
    return np.array(out).mean(0)


def main():
    lines = []

    def say(s=''):
        lines.append(s)
        print(s)

    feats, prof64, pk_share, pk_n, excess64 = {}, {}, {}, {}, {}
    exc_sum, flag_sum, n_syn = None, None, 0
    for name, folder in POPS.items():
        files = list_images(folder, limit=LIMIT)
        X, P, S30, N30, E64 = [], [], [], [], []
        for p in files:
            x = load_gray(p)
            X.append(f_fourier(x, nb=NB))
            P.append(radial_log_amplitude(x, NB64)[0])
            share, n, exc, b30, r = peak_stats(x)
            S30.append(share)
            N30.append(n)
            b64 = np.minimum((r / (0.5 * np.sqrt(2)) * NB64).astype(int), NB64 - 1)
            pos = np.maximum(exc, 0) * (r > RMIN)
            E64.append(np.bincount(b64.ravel(), pos.ravel(), minlength=NB64)
                       / np.maximum(np.bincount(b64.ravel(), minlength=NB64), 1))
            if name == 'synth':
                m = ((exc > T_NOTCH) & (r > RMIN)).astype(float)
                exc_sum = exc if exc_sum is None else exc_sum + exc
                flag_sum = m if flag_sum is None else flag_sum + m
                n_syn += 1
        feats[name] = np.array(X)
        prof64[name] = np.array(P)
        pk_share[name] = np.array(S30)
        pk_n[name] = np.array(N30)
        excess64[name] = np.array(E64)
        say(f'{name:9s} {len(files):4d} images  {folder}')

    edges = band_edges(NB)
    mu = {n: f.mean(0) for n, f in feats.items()}
    sd = {n: f.std(0, ddof=1) for n, f in feats.items()}
    d_pai = {v: cohen_d(feats[v], feats['pai']) for v in VARIANTS}
    d_bona = {v: cohen_d(feats[v], feats['bona']) for v in VARIANTS}
    d_bona_pai = cohen_d(feats['bona'], feats['pai'])
    sq_pai = {v: (mu[v] - mu['pai']) ** 2 for v in VARIANTS}
    sq_bona = {v: (mu[v] - mu['bona']) ** 2 for v in VARIANTS}

    say()
    say('=' * 100)
    say('Band table: f_fourier (log of normalised |F|^2 per band), 30 bands, radial edges in cycles/px.')
    say('Columns: mean per population | Cohen d bona-PAI | Cohen d (variant - real PAI) | Cohen d (variant - bona)')
    say('| share of the band energy inside the t=1.0 notch mask for synth / pk1.0 / sm')
    say('=' * 100)
    hdr = (f"{'b':>2s} {'lo':>6s} {'hi':>6s} | " + ' '.join(f'{n:>8s}' for n in POPS)
           + f" | {'d_bo-pai':>8s} | " + ' '.join(f'{v:>8s}' for v in VARIANTS)
           + ' | ' + ' '.join(f'{v:>8s}' for v in VARIANTS) + ' | pk_syn pk_pk1 pk_sm')
    say(hdr)
    rows_csv = []
    for b in range(NB):
        lo, hi = edges[b]
        row = (f'{b:2d} {lo:6.3f} {hi:6.3f} | ' + ' '.join(f'{mu[n][b]:8.2f}' for n in POPS)
               + f' | {d_bona_pai[b]:8.2f} | ' + ' '.join(f'{d_pai[v][b]:8.2f}' for v in VARIANTS)
               + ' | ' + ' '.join(f'{d_bona[v][b]:8.2f}' for v in VARIANTS)
               + f" | {pk_share['synth'][:, b].mean():6.3f} {pk_share['pk1.0'][:, b].mean():6.3f} {pk_share['sm'][:, b].mean():5.3f}")
        say(row)
        rows_csv.append([b, f'{lo:.4f}', f'{hi:.4f}'] + [f'{mu[n][b]:.4f}' for n in POPS]
                        + [f'{sd[n][b]:.4f}' for n in POPS] + [f'{d_bona_pai[b]:.3f}']
                        + [f'{d_pai[v][b]:.3f}' for v in VARIANTS] + [f'{d_bona[v][b]:.3f}' for v in VARIANTS]
                        + [f'{pk_share[n][:, b].mean():.4f}' for n in POPS]
                        + [f'{pk_n[n][:, b].mean():.2f}' for n in POPS])
    with open(os.path.join(OUT, 'band_table.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['band', 'lo_cpp', 'hi_cpp'] + [f'mean_{n}' for n in POPS] + [f'sd_{n}' for n in POPS]
                   + ['d_bona_vs_pai'] + [f'd_pai_{v}' for v in VARIANTS] + [f'd_bona_{v}' for v in VARIANTS]
                   + [f'peakshare_t1.0_{n}' for n in POPS] + [f'npeakbins_t1.0_{n}' for n in POPS])
        w.writerows(rows_csv)

    # ---- summaries --------------------------------------------------------------------------
    say()
    say('=' * 100)
    say('Whole-feature summaries (Euclidean over 30 bands, as the k-NN sees it)')
    say('=' * 100)
    say(f"{'variant':10s} {'n|d|>0.8 PAI':>13s} {'n|d|>0.8 bona':>14s} {'max|d| PAI':>12s} {'max|d| bona':>13s} "
        f"{'centroid d PAI':>15s} {'centroid d bona':>16s} {'mean within-sd':>15s}")
    for v in VARIANTS:
        bp = np.abs(d_pai[v]) > 0.8
        bb = np.abs(d_bona[v]) > 0.8
        ip, ib = np.argmax(np.abs(d_pai[v])), np.argmax(np.abs(d_bona[v]))
        say(f'{v:10s} {bp.sum():13d} {bb.sum():14d} {abs(d_pai[v][ip]):6.2f}(b{ip:2d}) {abs(d_bona[v][ib]):7.2f}(b{ib:2d}) '
            f'{np.sqrt(sq_pai[v].sum()):15.3f} {np.sqrt(sq_bona[v].sum()):16.3f} {sd[v].mean():15.3f}')
    say(f"{'bona-pai':10s} {(np.abs(d_bona_pai) > 0.8).sum():13d} {'':14s} {abs(d_bona_pai).max():6.2f}(b{np.argmax(abs(d_bona_pai)):2d})"
        f" {'':13s} {np.sqrt(((mu['bona'] - mu['pai']) ** 2).sum()):15.3f} {'':16s} bona {sd['bona'].mean():.3f} pai {sd['pai'].mean():.3f}")

    say()
    say('Bands with |d| > 0.8 (large effect), per comparison:')
    for v in VARIANTS:
        say(f'  {v:9s} vs real PAI : ' + ', '.join(f'b{b}({d_pai[v][b]:+.1f})' for b in np.where(np.abs(d_pai[v]) > 0.8)[0]))
        say(f'  {v:9s} vs bona fide: ' + ', '.join(f'b{b}({d_bona[v][b]:+.1f})' for b in np.where(np.abs(d_bona[v]) > 0.8)[0]))
    say('  bona fide vs real PAI: ' + ', '.join(f'b{b}({d_bona_pai[b]:+.1f})' for b in np.where(np.abs(d_bona_pai) > 0.8)[0]))

    say()
    say('Share of the squared centroid distance variant -> real PAI carried by each band (top 6):')
    for v in VARIANTS:
        sh = sq_pai[v] / sq_pai[v].sum()
        o = np.argsort(sh)[::-1][:6]
        say(f'  {v:9s} ' + ', '.join(f'b{b}: {100*sh[b]:.0f}% (diff {mu[v][b]-mu["pai"][b]:+.2f})' for b in o))
    say('Share of the squared centroid distance variant -> bona fide carried by each band (top 6):')
    for v in VARIANTS:
        sh = sq_bona[v] / sq_bona[v].sum()
        o = np.argsort(sh)[::-1][:6]
        say(f'  {v:9s} ' + ', '.join(f'b{b}: {100*sh[b]:.0f}% (diff {mu[v][b]-mu["bona"][b]:+.2f})' for b in o))

    # ---- k-NN replica -----------------------------------------------------------------------
    say()
    say('=' * 100)
    say(f'Single-split k-NN replica (k={K}, train bona(120) vs variant(120), test real PAI(120)),')
    say('mean distance of a real-PAI test sample to its 15 nearest bona fide / 15 nearest variant rows')
    say('=' * 100)
    for v in VARIANTS:
        ap, db, dv = knn_split(feats['bona'], feats[v], feats['pai'])
        say(f'  {v:9s} APCER {100*ap:5.1f} %   dist to 15-NN bona {db:.3f}   to 15-NN variant {dv:.3f}   ratio {dv/db:.2f}')
    ap0, db0, dv0 = knn_split(feats['bona'], feats['pai'][:60], feats['pai'][60:])
    say(f'  (sanity: train on 60 real PAI, test the other 60) APCER {100*ap0:5.1f} %  dist bona {db0:.3f} pai {dv0:.3f}')

    say()
    say('k-NN replica restricted to band subsets (train bona vs variant, test real PAI):')
    subsets = {'b0-4': range(0, 5), 'b5-9': range(5, 10), 'b10-14': range(10, 15), 'b15-19': range(15, 20),
               'b20-24': range(20, 25), 'b25-29': range(25, 30), 'b1-29': range(1, 30)}
    say(f"  {'variant':9s} " + ' '.join(f'{k:>8s}' for k in subsets))
    for v in VARIANTS:
        r_ = []
        for k_, idx in subsets.items():
            idx = list(idx)
            ap, _, _ = knn_split(feats['bona'][:, idx], feats[v][:, idx], feats['pai'][:, idx])
            r_.append(ap)
        say(f'  {v:9s} ' + ' '.join(f'{100*x:7.1f}%' for x in r_))

    say()
    say('Leave-band-out: k-NN replica APCER when ONE band is removed from the untouched-synthetic feature')
    say('(bands whose removal lowers APCER most carry the synthetic-vs-PAI mismatch):')
    base, _, _ = knn_split(feats['bona'], feats['synth'], feats['pai'])
    lbo = []
    for b in range(NB):
        idx = [i for i in range(NB) if i != b]
        ap, _, _ = knn_split(feats['bona'][:, idx], feats['synth'][:, idx], feats['pai'][:, idx])
        lbo.append(ap)
    say(f'  all bands {100*base:.1f} %; ' + ' '.join(f'-b{b}:{100*a:.0f}' for b, a in enumerate(lbo)))
    say('Same for pk1.0_sm (chain):')
    base, _, _ = knn_split(feats['bona'], feats['pk1.0_sm'], feats['pai'])
    lbo = []
    for b in range(NB):
        idx = [i for i in range(NB) if i != b]
        ap, _, _ = knn_split(feats['bona'][:, idx], feats['pk1.0_sm'][:, idx], feats['pai'][:, idx])
        lbo.append(ap)
    say(f'  all bands {100*base:.1f} %; ' + ' '.join(f'-b{b}:{100*a:.0f}' for b, a in enumerate(lbo)))

    # ---- peaks of the untouched synthetic set -------------------------------------------------
    say()
    say('=' * 100)
    say('Where the peaks sit (untouched synthetic)')
    say('=' * 100)
    m64 = {n: p.mean(0) for n, p in prof64.items()}
    w64 = 0.5 * np.sqrt(2) / NB64
    say('64-bin mean radial log1p-amplitude profile (what specmatch equalises), differences to real PAI,')
    say('and the mean positive notch excess (logm - 7x7 local median)+ per bin (bins above rmin=0.1 only):')
    say(f"{'bin':>3s} {'cpp':>6s} {'b30':>3s} " + ' '.join(f'{n:>6s}' for n in ['bona', 'pai', 'synth', 'pk1.0', 'sm', 'pk1_sm'])
        + f" | {'syn-pai':>7s} {'pk1-pai':>7s} {'sm-pai':>7s} {'pk1sm-pai':>9s} | {'exc_syn':>7s} {'exc_pk1':>7s} {'exc_sm':>7s} {'exc_pk1sm':>9s} {'exc_pai':>7s}")
    pk_rows = []
    for i in range(NB64):
        c = (i + 0.5) * w64
        b30 = min(int(i / NB64 * NB), NB - 1)
        say(f"{i:3d} {c:6.3f} {b30:3d} " + ' '.join(f"{m64[n][i]:6.2f}" for n in ['bona', 'pai', 'synth', 'pk1.0', 'sm', 'pk1.0_sm'])
            + f" | {m64['synth'][i]-m64['pai'][i]:7.2f} {m64['pk1.0'][i]-m64['pai'][i]:7.2f} {m64['sm'][i]-m64['pai'][i]:7.2f} {m64['pk1.0_sm'][i]-m64['pai'][i]:9.2f}"
            + f" | {excess64['synth'][:, i].mean():7.3f} {excess64['pk1.0'][:, i].mean():7.3f} {excess64['sm'][:, i].mean():7.3f} {excess64['pk1.0_sm'][:, i].mean():9.3f} {excess64['pai'][:, i].mean():7.3f}")
        pk_rows.append([i, f'{c:.4f}', b30] + [f'{m64[n][i]:.4f}' for n in POPS] + [f'{excess64[n][:, i].mean():.4f}' for n in POPS])
    with open(os.path.join(OUT, 'peaks.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['bin64', 'centre_cpp', 'band30'] + [f'meanlogamp_{n}' for n in POPS] + [f'notch_excess_{n}' for n in POPS])
        w.writerows(pk_rows)

    say()
    say('Strongest 2-D peaks of the untouched synthetic set: mean over images of (logm - 7x7 local median),')
    say('bins above rmin, grouped by |fy|,|fx| (symmetric copies dropped), with the fraction of images in')
    say('which the t=1.0 notch flags that exact bin:')
    exc_mean = exc_sum / n_syn
    flag_frac = flag_sum / n_syn
    h, w_ = exc_mean.shape
    fy = np.fft.fftfreq(h)[:, None] * np.ones((1, w_))
    fx = np.fft.fftfreq(w_)[None, :] * np.ones((h, 1))
    r = norm_radius(h, w_)
    cand = np.argwhere((exc_mean > 0.4) & (r > RMIN)).tolist()
    cand.sort(key=lambda ij: -exc_mean[ij[0], ij[1]])
    seen = []
    say(f"{'fy':>7s} {'fx':>7s} {'r_cpp':>6s} {'b30':>3s} {'b64':>3s} {'mean_exc':>8s} {'frac_flagged':>12s}")
    for i, j in cand:
        key = (round(abs(fy[i, j]), 3), round(abs(fx[i, j]), 3))
        if key in seen:
            continue
        seen.append(key)
        b30 = min(int(r[i, j] / (0.5 * np.sqrt(2)) * NB), NB - 1)
        b64 = min(int(r[i, j] / (0.5 * np.sqrt(2)) * NB64), NB64 - 1)
        say(f'{fy[i, j]:7.3f} {fx[i, j]:7.3f} {r[i, j]:6.3f} {b30:3d} {b64:3d} {exc_mean[i, j]:8.2f} {flag_frac[i, j]:12.2f}')
        if len(seen) >= 30:
            break
    say(f'  (candidates with mean excess > 0.4 above rmin: {len(cand)} bins of {h*w_})')
    say()
    say('Per-band number of bins flagged by the t=1.0 notch (mean over images), bands 4..29:')
    for n in ['synth', 'pk1.0', 'sm', 'pk1.0_sm', 'pai', 'bona']:
        say(f'  {n:9s} ' + ' '.join(f'b{b}:{pk_n[n][:, b].mean():.0f}' for b in range(4, NB)))
    say('Per-band share of |F|^2 inside the flagged bins (mean over images), bands 4..29:')
    for n in ['synth', 'pk1.0', 'sm', 'pk1.0_sm', 'pai', 'bona']:
        say(f'  {n:9s} ' + ' '.join(f'b{b}:{pk_share[n][:, b].mean():.2f}' for b in range(4, NB)))

    # ---- mean-of-log vs log-of-mean --------------------------------------------------------------
    say()
    say('=' * 100)
    say('Peakiness per 30-band: log(mean |F|^2 over the band) - 2*mean(log|F|) (nats).')
    say('Radial matching equalises the second term (mean log amplitude); the k-NN feature is the first.')
    say('0 = flat band; large = the band energy sits in a few bins (periodic peaks).')
    say('=' * 100)
    pkn = {n: peakiness(POPS[n]) for n in POPS}
    say(f"{'band':>4s} " + ' '.join(f'{n:>9s}' for n in POPS))
    for b in range(NB):
        say(f'{b:4d} ' + ' '.join(f'{pkn[n][b]:9.2f}' for n in POPS))

    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'band_table.md'), 'w', encoding='utf-8') as fh:
        fh.write('| band | cpp lo-hi | bona | PAI | synth | pk1.0 | sm | pk1.0_sm | pk2.0_sm | d bona-PAI | d synth-PAI | d pk1.0-PAI | d sm-PAI | d pk1.0_sm-PAI | d pk2.0_sm-PAI | d pk1.0_sm-bona | peak share synth (t=1) |\n')
        fh.write('|' + '---|' * 17 + '\n')
        for b in range(NB):
            lo, hi = edges[b]
            fh.write(f'| {b} | {lo:.3f}-{hi:.3f} | ' + ' | '.join(f'{mu[n][b]:.2f}' for n in POPS)
                     + f' | {d_bona_pai[b]:+.2f} | ' + ' | '.join(f'{d_pai[v][b]:+.2f}' for v in VARIANTS)
                     + f" | {d_bona['pk1.0_sm'][b]:+.2f} | {pk_share['synth'][:, b].mean():.2f} |\n")
    print(f'\nwritten: {OUT}: band_table.csv, band_table.md, peaks.csv, report.txt')


if __name__ == '__main__':
    main()
