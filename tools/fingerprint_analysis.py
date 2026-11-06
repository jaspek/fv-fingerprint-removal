"""Test A: model-fingerprint presence in the Fourier and autocorrelation domain.

Example:
  python fingerprint_analysis.py --pop real=.../PLUS/real --pop pai=.../PLUS/spoof \
      --pop synth=.../AE_Res-diff/PLUS_matched/spoof/samples --pop cleaned=.../cleaned \
      --ref real --fp-ref synth --tile 192 --limit 300 --out results/fp_plus

Writes:  <out>/spectra.png (averaged residual log-spectra), <out>/autocorr.png,
         <out>/radial.png (radial profiles), <out>/metrics.csv
Metrics per population: peak ratio at 1/2 and 1/4 cycles/px (grid artefacts), log-spectral distance
of the radial profile to the --ref population, spectral "darkness" (fraction of near-zero
coefficients, the SAC '25 cue), and NCC of each residual to the mean fingerprint of --fp-ref.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from common import list_images, load_gray, center_tile, residual, log_spectrum, radial_profile


def peak_ratio(spec, frac):
    """Peak-to-local-median ratio at the axial grid positions +-frac cycles/px."""
    n = spec.shape[0]; c = n // 2; d = int(round(frac * n))
    lin = np.expm1(spec)
    vals = []
    for (y, x) in [(c - d, c), (c, c - d), (min(c + d, n - 1), c), (c, min(c + d, n - 1))]:
        y0, y1, x0, x1 = max(y - 6, 0), min(y + 7, n), max(x - 6, 0), min(x + 7, n)
        box = lin[y0:y1, x0:x1].copy()
        peak = lin[y, x]
        box[y - y0, x - x0] = np.nan
        vals.append(peak / (np.nanmedian(box) + 1e-8))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', action='append', required=True, help='name=DIR (repeatable)')
    ap.add_argument('--ref', required=True, help='name of the real reference population')
    ap.add_argument('--fp-ref', default=None, help='population whose mean residual is the fingerprint')
    ap.add_argument('--tile', type=int, default=0, help='square tile size (0 = min side of first image)')
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pops = {}
    for spec in a.pop:
        name, d = spec.split('=', 1)
        files = list_images(d, a.limit)
        if not files:
            raise SystemExit(f'no images in {d}')
        pops[name] = files
    tile = a.tile or min(load_gray(pops[a.ref][0]).shape)
    half = min(32, tile // 4)

    stats = {}
    for name, files in pops.items():
        S = np.zeros((tile, tile)); AC = np.zeros((2 * half + 1, 2 * half + 1)); R = None
        dark = []; res_list = []
        for p in files:
            x = center_tile(load_gray(p), tile)
            r = residual(x); res_list.append(r)
            S += log_spectrum(r)
            z = (r - r.mean()) / (r.std() + 1e-8)
            ac = fftconvolve(z, z[::-1, ::-1], mode='same'); cy, cx = np.array(ac.shape) // 2
            AC += ac[cy - half:cy + half + 1, cx - half:cx + half + 1] / (ac[cy, cx] + 1e-8)
            mag = np.abs(np.fft.fft2(x)); dark.append(np.mean(mag < 0.01 * mag.max()))
        n = len(files); S /= n; AC /= n
        stats[name] = dict(S=S, AC=AC, radial=radial_profile(S), dark=float(np.mean(dark)),
                           res=np.stack(res_list), n=n)
        print(f'{name}: {n} images, tile {tile}')

    # NCC to the mean fingerprint of --fp-ref
    if a.fp_ref:
        fp = stats[a.fp_ref]['res'].mean(axis=0); fp = (fp - fp.mean()) / (fp.std() + 1e-8)
        for name, st in stats.items():
            nccs = []
            for r in st['res']:
                z = (r - r.mean()) / (r.std() + 1e-8); nccs.append(float(np.mean(z * fp)))
            st['ncc'] = float(np.mean(nccs))

    ref_rad = stats[a.ref]['radial']
    with open(os.path.join(a.out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['population', 'n', 'peak_ratio_1/2', 'peak_ratio_1/4', 'log_spectral_dist_to_ref',
                    'darkness', 'ncc_to_fingerprint'])
        for name, st in stats.items():
            m = min(len(ref_rad), len(st['radial'])) - 1
            lsd = float(np.mean((np.log(st['radial'][1:m] + 1e-8) - np.log(ref_rad[1:m] + 1e-8)) ** 2))
            row = [name, st['n'], f"{peak_ratio(st['S'], 0.5):.3f}", f"{peak_ratio(st['S'], 0.25):.3f}",
                   f'{lsd:.4f}', f"{st['dark']:.4f}", f"{st.get('ncc', float('nan')):.4f}"]
            w.writerow(row); print(row)

    names = list(stats)
    for key, fname, cmap in [('S', 'spectra.png', 'gray'), ('AC', 'autocorr.png', 'viridis')]:
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
        for ax, name in zip(np.atleast_1d(axes), names):
            ax.imshow(stats[name][key], cmap=cmap); ax.set_title(name, fontsize=18); ax.axis('off')
        fig.tight_layout(); fig.savefig(os.path.join(a.out, fname), dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        ax.plot(np.log(stats[name]['radial'][1:] + 1e-8), label=name)
    ax.set_xlabel('radius (bins)'); ax.set_ylabel('log power (residual)'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'radial.png'), dpi=120); plt.close(fig)
    print('written', a.out)


if __name__ == '__main__':
    main()
