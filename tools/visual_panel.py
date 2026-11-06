"""Side-by-side panel for the visual check: untouched image, cleaned image, amplified difference, and a zoomed
centre crop of both. One row per image, images drawn at random (fixed seed) from the untouched folder.

Example:
  python visual_panel.py --a <root>/AE_Res-diff/PLUS_matched/spoof/samples --b <root>/AE_Res-leakfree-v210-fold0/PLUS_matched/spoof/samples --n 4 --out panel_PLUS.png
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio
from common import list_images, load_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='untouched folder')
    ap.add_argument('--b', required=True, help='cleaned folder with the same relative file names')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--crop', type=int, default=96, help='side of the zoomed centre crop in pixels')
    ap.add_argument('--zoom', type=int, default=3)
    ap.add_argument('--gain', type=float, default=8.0, help='amplification of the difference image (grey 128 = no change)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    files = list_images(a.a, a.n, a.seed)
    rows = []
    for p in files:
        rel = os.path.relpath(p, a.a)
        q = os.path.join(a.b, os.path.splitext(rel)[0] + '.png')
        if not os.path.exists(q):
            q = os.path.join(a.b, rel)
        x, y = load_gray(p), load_gray(q)
        d = np.clip(128 + a.gain * (y - x), 0, 255)
        h, w = x.shape
        c = min(a.crop, h, w)
        cy, cx = (h - c) // 2, (w - c) // 2
        zoom = lambda im: np.kron(im[cy:cy + c, cx:cx + c], np.ones((a.zoom, a.zoom)))
        print(f'{rel}: PSNR {peak_signal_noise_ratio(x, y, data_range=255):.2f} dB, max abs change {np.abs(y - x).max():.0f}')
        rows.append([x, y, d, zoom(x), zoom(y)])

    gap, top = 8, 18
    tiles_w = [max(r[k].shape[1] for r in rows) for k in range(5)]
    row_h = [max(t.shape[0] for t in r) for r in rows]
    W = sum(tiles_w) + gap * 6
    H = top + sum(row_h) + gap * (len(rows) + 1)
    canvas = Image.new('L', (W, H), 255)
    draw = ImageDraw.Draw(canvas)
    labels = ['untouched', 'cleaned', f'difference x{a.gain:g} (128 = unchanged)', f'crop {c} px, untouched', f'crop {c} px, cleaned']
    x0 = gap
    for k, lab in enumerate(labels):
        draw.text((x0, 3), lab, fill=0)
        x0 += tiles_w[k] + gap
    y0 = top + gap
    for r, rh in zip(rows, row_h):
        x0 = gap
        for k, t in enumerate(r):
            canvas.paste(Image.fromarray(np.clip(np.rint(t), 0, 255).astype(np.uint8)), (x0, y0 + (rh - t.shape[0]) // 2))
            x0 += tiles_w[k] + gap
        y0 += rh + gap
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    canvas.save(a.out)
    print('->', a.out)


if __name__ == '__main__':
    main()
