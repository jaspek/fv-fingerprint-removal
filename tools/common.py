"""Shared helpers for the finger-vein fingerprint toolkit (grayscale only)."""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

EXTS = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


def list_images(root, limit=None, seed=0):
    """All image files below root (recursive), sorted; optional random subset."""
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(root)
             for f in fs if f.lower().endswith(EXTS)]
    files.sort()
    if limit and len(files) > limit:
        rng = np.random.default_rng(seed)
        files = sorted(rng.choice(files, limit, replace=False).tolist())
    return files


def load_gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32)


def save_gray(arr, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(path)


def user_id(path, dataset):
    """Subject identifier from the file name, per dataset naming convention."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split('_')
    if dataset == 'plus':      # PLUS-FV3-Laser_PALMAR_001_01_02_01 -> 001 (user before session)
        return parts[2]
    if dataset == 'idiap':     # 001_L_1 (or 001_L_1_spoof after decoding the test set) -> 001
        return parts[0]
    if dataset == 'scut':      # 21_1_1_0_3 -> 21
        return parts[0]
    return os.path.basename(os.path.dirname(path))


def center_tile(img, size):
    """Central size x size crop (size <= min(H, W))."""
    h, w = img.shape
    size = min(size, h, w)
    y0, x0 = (h - size) // 2, (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def residual(img):
    """Noise residual: image minus a 3x3 median-filtered copy (no zero-mean step)."""
    return img - median_filter(img, size=3)


def radial_profile(spec2d):
    """Azimuthally averaged profile of a centred square 2-D spectrum."""
    h, w = spec2d.shape
    y, x = np.indices(spec2d.shape)
    r = np.hypot(y - h // 2, x - w // 2).astype(int)
    n = np.bincount(r.ravel())
    s = np.bincount(r.ravel(), spec2d.ravel())
    return s / np.maximum(n, 1)


def log_spectrum(img):
    """Centred log-magnitude spectrum of an image or residual."""
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))


def norm_radius(h, w):
    """Normalised radial frequency (cycles/pixel) for each FFT bin (non-shifted layout)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.hypot(fy, fx)


def radial_log_energy(img, nbins=64):
    """Log of the mean |F|^2 per normalised-frequency bin (ring energy, what band-energy PAD features measure)."""
    F = np.fft.fft2(img)
    r = norm_radius(*img.shape)
    bins = np.minimum((r / (0.5 * np.sqrt(2)) * nbins).astype(int), nbins - 1)
    e = np.abs(F) ** 2
    s = np.bincount(bins.ravel(), e.ravel(), minlength=nbins)
    n = np.bincount(bins.ravel(), minlength=nbins)
    return np.log(s / np.maximum(n, 1) + 1e-12), bins


def cell_index(shape, nbins=64, sectors=8):
    """Index of the (ring, sector) cell of every FFT bin: ring as in radial_log_amplitude, sector from the
    orientation of (fy, fx) folded to [0, pi) so that bin and mirror bin share a cell (keeps the output real)."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    r = np.hypot(fy, fx)
    ring = np.minimum((r / (0.5 * np.sqrt(2)) * nbins).astype(int), nbins - 1)
    ang = np.mod(np.arctan2(fy, fx), np.pi)
    sec = np.minimum((ang / np.pi * sectors).astype(int), sectors - 1)
    return ring * sectors + sec, nbins * sectors


def cell_log_energy(img, nbins=64, sectors=8):
    """Log of the mean |F|^2 per (ring, sector) cell: the 2-D spectral envelope at ring x sector resolution."""
    F = np.fft.fft2(img)
    cells, n = cell_index(img.shape, nbins, sectors)
    e = np.abs(F) ** 2
    s = np.bincount(cells.ravel(), e.ravel(), minlength=n)
    c = np.bincount(cells.ravel(), minlength=n)
    return np.log(s / np.maximum(c, 1) + 1e-12), cells


def radial_log_amplitude(img, nbins=64):
    """Mean log amplitude per normalised-frequency bin (works for non-square images)."""
    F = np.fft.fft2(img)
    r = norm_radius(*img.shape)
    bins = np.minimum((r / (0.5 * np.sqrt(2)) * nbins).astype(int), nbins - 1)
    la = np.log1p(np.abs(F))
    s = np.bincount(bins.ravel(), la.ravel(), minlength=nbins)
    n = np.bincount(bins.ravel(), minlength=nbins)
    return s / np.maximum(n, 1), bins
