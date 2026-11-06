"""Test B: PSNR and SSIM between a synthetic set and its cleaned version (paired by file name).

Example:
  python quality_metrics.py --a .../synth --b .../cleaned --out results/quality_cleaned.csv
"""
import argparse, csv, os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from common import list_images, load_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='original (synthetic) folder')
    ap.add_argument('--b', required=True, help='processed folder with the same relative file names')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    rows = []
    for pa in list_images(a.a, a.limit or None):
        rel = os.path.relpath(pa, a.a)
        pb = os.path.join(a.b, rel)
        if not os.path.exists(pb):
            pb = os.path.splitext(pb)[0] + '.png'
            if not os.path.exists(pb):
                continue
        x, y = load_gray(pa), load_gray(pb)
        h, w = min(x.shape[0], y.shape[0]), min(x.shape[1], y.shape[1])
        x, y = x[:h, :w], y[:h, :w]
        rows.append((rel, peak_signal_noise_ratio(x, y, data_range=255),
                     structural_similarity(x, y, data_range=255)))
    if not rows:
        raise SystemExit('no paired files found')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['file', 'psnr', 'ssim']); w.writerows(rows)
    ps, ss = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    print(f'{len(rows)} pairs  PSNR {ps.mean():.2f} +- {ps.std():.2f} dB   SSIM {ss.mean():.4f} +- {ss.std():.4f}')


if __name__ == '__main__':
    main()
