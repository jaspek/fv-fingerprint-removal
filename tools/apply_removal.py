"""Apply a removal / control operator to every image in a folder (output mirrors the folder).

Operators:
  jpeg:q=75            JPEG round trip (control)
  noise:sigma=2        additive Gaussian noise (control; the one baseline that adds energy)
  blur:sigma=1         Gaussian blur (control, subtractive)
  median:size=3        median filter (control)
  lsb                  flip the least significant bit of every pixel (detector-brittleness control)
  peaks:t=1.2,size=7,rmin=0.1   adaptive notch filter (sweep t in 1.0..1.5: t=1.0 was too strong, t>=1.5 did nothing on PLUS): spectral peaks more than t above the local
                       median (log magnitude, size x size window) at radii above rmin cycles/px are
                       pulled down to the median (the "frequency-peaks" idea of Wesselkamp et al.;
                       chain it BEFORE specmatch; sweep t, lower t = more aggressive)
  noisematch:strength=1.0,mode=1,stat=0,scale=1.0   add white noise so the pixel-noise level (stat=0, residual std)
                           or the share of equal-valued neighbour pairs (stat=1) matches a random real image; needs --ref DIR
  specmatch:strength=1.0,mode=0,energy=0   radial spectrum matching to a real reference population
                           (energy=1: match ring energies |F|^2 instead of the mean log amplitude; use after peaks;
                           sectors=8 additionally matches the angular distribution, ring x sector cells;
                           rmin=0.05 leaves rings below 0.05 cycles/px untouched to protect PSNR;
                           rmax=0.4 leaves rings above 0.4 cycles/px untouched)
                           (proposal P1: phase kept, per-frequency gain = exp(strength*(ref - own)));
                           mode=0 matches the population mean profile, mode=1 the profile of a random
                           real image (keeps the natural spread); needs --ref DIR

Example:
  python apply_removal.py --in .../synth --out .../synth_specmatch --op specmatch:strength=1.0 --ref .../PLUS/real
"""
import argparse, io, os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from common import list_images, load_gray, save_gray, radial_log_amplitude, radial_log_energy, cell_log_energy, norm_radius
from scipy.ndimage import gaussian_filter


def parse_op(s):
    name, _, rest = s.partition(':')
    params = {}
    for kv in filter(None, rest.split(',')):
        k, v = kv.split('='); params[k] = float(v)
    return name, params


def op_jpeg(x, q=75.0):
    buf = io.BytesIO(); Image.fromarray(x.astype(np.uint8)).save(buf, 'JPEG', quality=int(q))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('L'), dtype=np.float32)


def op_noise(x, sigma=2.0, rng=np.random.default_rng(0)):
    return x + rng.normal(0, sigma, x.shape)


def op_blur(x, sigma=1.0):
    return gaussian_filter(x, sigma)


def op_median(x, size=3.0):
    return median_filter(x, size=int(size))


def op_lsb(x):
    return (x.astype(np.uint8) ^ 1).astype(np.float32)


def op_peaks(x, t=1.2, size=7, rmin=0.1):
    """Adaptive notch: spectral peaks whose log magnitude exceeds the local median (size x size)
    by more than t are pulled down to that median; only radii above rmin cycles/px are touched
    (the low-frequency image content near DC is left alone); phase is kept."""
    F = np.fft.fft2(x)
    mag = np.abs(F); logm = np.log1p(mag)
    med = median_filter(logm, size=int(size), mode='wrap')
    mask = ((logm - med) > t) & (norm_radius(*x.shape) > rmin)
    newmag = mag.copy(); newmag[mask] = np.expm1(med[mask])
    y = np.real(np.fft.ifft2(newmag * np.exp(1j * np.angle(F))))
    return np.clip(y, 0, 255)


def _resid_std(x):
    """Std of the high-pass residual (image minus Gaussian blur, sigma 1): the pixel-noise level."""
    return float((x - gaussian_filter(x, 1.0)).std())


def _tie_frac(x):
    """Share of horizontally neighbouring pixel pairs with exactly equal 8-bit value (what uniform LBP codes react to)."""
    q = np.clip(np.rint(x), 0, 255)
    return float((np.diff(q, axis=1) == 0).mean())


def make_noisematch(ref_dir, limit):
    """noisematch:strength=1.0,mode=1,stat=0,scale=1.0
    Add white Gaussian noise so that a pixel-level statistic of the image matches a real reference image:
      stat=0: the high-pass residual std (noise level); the needed sigma follows from adding in quadrature
      stat=1: the share of equal-valued neighbour pairs (ties), found by bisection on sigma
    mode=1 draws a random reference image per input (keeps the spread), mode=0 uses the population mean.
    strength scales the gap to close (1 = full), scale multiplies the resulting sigma (for sweeps).
    Noise is only added, never removed: images already noisier than the reference stay as they are."""
    refs = [load_gray(p) for p in list_images(ref_dir, limit)]
    rstd = np.array([_resid_std(im) for im in refs]); rtie = np.array([_tie_frac(im) for im in refs])
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((256, 256))
    c = float((probe - gaussian_filter(probe, 1.0)).std())        # residual std of unit white noise after the high-pass
    stats = {"n": 0, "sigma_sum": 0.0}

    def op(x, strength=1.0, mode=1.0, stat=0.0, scale=1.0):
        i = rng.integers(len(refs))
        if int(stat) == 0:
            target = rstd[i] if int(mode) == 1 else rstd.mean()
            own = _resid_std(x)
            goal = own + strength * (target - own)
            sigma = np.sqrt(max(goal ** 2 - own ** 2, 0.0)) / c
        else:
            target = rtie[i] if int(mode) == 1 else rtie.mean()
            own = _tie_frac(x)
            goal = own + strength * (target - own)
            sigma = 0.0
            if goal < own:                                           # more noise -> fewer ties (monotone), bisection
                lo, hi = 0.0, 6.0
                noise = rng.standard_normal(x.shape)
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    if _tie_frac(x + mid * noise) > goal: lo = mid
                    else: hi = mid
                sigma = 0.5 * (lo + hi)
        sigma *= scale
        stats["n"] += 1; stats["sigma_sum"] += sigma
        if sigma <= 0:
            return x
        return np.clip(x + rng.standard_normal(x.shape) * sigma, 0, 255)
    op.stats = stats
    return op


def make_specmatch(ref_dir, limit, nbins=64):
    refs = [load_gray(p) for p in list_images(ref_dir, limit)]
    shape = refs[0].shape
    if any(im.shape != shape for im in refs):
        raise SystemExit('specmatch: reference images must share one size (ring profiles depend on it)')
    profs = np.array([radial_log_amplitude(im, nbins)[0] for im in refs])    # ring mean of log(1+|F|)
    eprofs = np.array([radial_log_energy(im, nbins)[0] for im in refs])      # ring log mean |F|^2
    cprofs = {}                                                              # (ring, sector) log energies, per sector count, lazily
    ref_mean, eref_mean = profs.mean(axis=0), eprofs.mean(axis=0)            # mode 0 targets: mean log profile,
    rng = np.random.default_rng(0)                                           # i.e. geometric mean of ring energies

    def op(x, strength=1.0, mode=0.0, energy=0.0, sectors=1.0, rmin=0.0, rmax=1.0):
        # rmax: rings above this radius (cycles/px) keep gain 1 (1 = none), so the finest pixel-scale
        #       texture is left as generated (SCUT: matching the corner rings made LBP see noise)
        kmax = min(nbins, int(rmax / (0.5 * np.sqrt(2)) * nbins)) if rmax < 1.0 else nbins
        # rmin: rings below this radius (cycles/px) keep gain 1 (0 = only the DC ring), so coarse
        #       image content is left alone and PSNR is preserved; the fingerprint sits higher up
        kmin = max(1, int(rmin / (0.5 * np.sqrt(2)) * nbins) + 1) if rmin > 0 else 1
        if x.shape != shape:
            raise SystemExit(f'specmatch: input {x.shape} differs from reference size {shape}')
        # mode=0: match the population-mean profile (all outputs get the same radial envelope,
        #         which collapses the class spread; the k-NN PAD still fails on that)
        # mode=1: match the profile of a randomly drawn real image (keeps natural spread)
        # energy=0: match the ring mean of log amplitude (the radial envelope)
        # energy=1: match the ring ENERGY (mean |F|^2; mode 0 uses the geometric mean over the references), which is what band-energy PAD features
        #           measure; gain = (E_ref / E_own)^(strength/2) per ring. Use after the notch:
        #           on a peaky ring it would scale the peaks with the floor.
        i = rng.integers(len(profs))
        if int(energy) == 1 and int(sectors) > 1:
            # sectors>1: match the 2-D envelope (energy per ring x sector cell), so the angular
            # distribution follows the twin as well; needed for band features that are not
            # isotropic in cycles/px (e.g. pixel-radius bands on non-square images)
            S = int(sectors)
            if S not in cprofs:
                cprofs[S] = np.array([cell_log_energy(im, nbins, S)[0] for im in refs])
            ref = cprofs[S][i] if int(mode) == 1 else cprofs[S].mean(axis=0)
            own, cells = cell_log_energy(x, nbins, S)
            gain = np.exp(0.5 * strength * (ref - own)); gain[:kmin * S] = 1.0   # keep rings below rmin (at least DC)
            gain[kmax * S:] = 1.0                                              # keep rings above rmax
            F = np.fft.fft2(x)
            return np.clip(np.real(np.fft.ifft2(F * gain[cells])), 0, 255)
        if int(energy) == 1:
            ref = eprofs[i] if int(mode) == 1 else eref_mean
            own, bins = radial_log_energy(x, nbins)
            gain = np.exp(0.5 * strength * (ref - own))
        else:
            ref = profs[i] if int(mode) == 1 else ref_mean
            own, bins = radial_log_amplitude(x, nbins)
            gain = np.exp(strength * (ref - own))
        gain[:kmin] = 1.0                                               # keep rings below rmin (at least DC)
        gain[kmax:] = 1.0                                               # keep rings above rmax
        F = np.fft.fft2(x)
        y = np.real(np.fft.ifft2(F * gain[bins]))
        return np.clip(y, 0, 255)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--op', required=True)
    ap.add_argument('--ref', default=None, help='real reference folder (specmatch)')
    ap.add_argument('--ref-limit', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    name, params = parse_op(a.op)
    ops = dict(jpeg=op_jpeg, noise=op_noise, blur=op_blur, median=op_median, lsb=op_lsb, peaks=op_peaks)
    if name == 'specmatch':
        if not a.ref:
            raise SystemExit('specmatch needs --ref')
        fn = make_specmatch(a.ref, a.ref_limit)
    elif name == 'noisematch':
        if not a.ref:
            raise SystemExit('noisematch needs --ref')
        fn = make_noisematch(a.ref, a.ref_limit)
    elif name in ops:
        fn = ops[name]
    else:
        raise SystemExit(f'unknown op {name}')
    files = list_images(a.inp, a.limit or None)
    for p in files:
        rel = os.path.splitext(os.path.relpath(p, a.inp))[0] + '.png'
        save_gray(fn(load_gray(p), **params), os.path.join(a.out, rel))
    extra = f' (mean sigma {fn.stats["sigma_sum"] / max(fn.stats["n"], 1):.2f})' if hasattr(fn, 'stats') else ''
    print(f'{name} {params}: {len(files)} images -> {a.out}{extra}')


if __name__ == '__main__':
    main()
