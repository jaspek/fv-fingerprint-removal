"""Summarise a sweep run (sweep_peaks.sh): one row per cleaned set with PSNR/SSIM against the
untouched synthetic images and the k-NN PAD numbers (APCER/BPCER in %, mean over the folds).

  python summarize_sweep.py --results .../smoke_subset/results --out .../sweep_summary.csv
"""
import argparse, csv, glob, os, re

FEATS = ["fourier", "lbp", "hist", "var", "entropy"]


def pad_mean(path):
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["fold"] == "mean":
            return 100 * float(r["APCER"]), 100 * float(r["BPCER"]), 100 * float(r["ACER"])
    return None


def quality(path):
    """Mean PSNR/SSIM; an unchanged image has PSNR inf, which is capped at 100 dB so the mean stays readable."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    return (sum(min(float(r["psnr"]), 100.0) for r in rows) / len(rows), sum(float(r["ssim"]) for r in rows) / len(rows))


def sort_key(tag):
    """untouched, then matching-only rows (sm, em), then pk<t>_s<size>[_n<sigma>][_sm|_em] by t, size, variant."""
    if tag == "untouched":
        return (0, 0.0, 0, 0.0, 0)
    if tag in ("sm", "em"):
        return (1, 0.0, 0, 0.0, 0 if tag == "sm" else 1)
    m = re.match(r"pk([0-9.]+)_s(\d+)(?:_n([0-9.]+))?(?:_(sm|em))?$", tag)
    if m:
        variant = {None: 0, "sm": 1, "em": 2}[m.group(4)]
        return (2, float(m.group(1)), int(m.group(2)), float(m.group(3) or 0), variant)
    return (3, 0.0, 0, 0.0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tags = sorted({os.path.basename(p)[len("pad_fourier_"):-4] for p in glob.glob(os.path.join(a.results, "pad_fourier_*.csv"))}, key=sort_key)
    header = ["set", "PSNR_dB", "SSIM"] + [f"{f}_{m}" for f in FEATS for m in ("APCER", "BPCER")]
    rows = []
    for tag in tags:
        q = os.path.join(a.results, f"q_{tag}.csv")
        psnr, ssim = quality(q) if os.path.exists(q) else (float("nan"), float("nan"))
        row = [tag, psnr, ssim]
        for f in FEATS:
            p = os.path.join(a.results, f"pad_{f}_{tag}.csv")
            r = pad_mean(p) if os.path.exists(p) else None
            row += [r[0], r[1]] if r else [float("nan"), float("nan")]
        rows.append(row)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([row[0], f"{row[1]:.2f}", f"{row[2]:.4f}"] + [f"{v:.2f}" for v in row[3:]])

    def fmt(v, nd=1):
        return "   -" if v != v else f"{v:.{nd}f}"

    print()
    print(f"{'set':<16}{'PSNR':>7}{'SSIM':>7}{'Fourier A/B':>14}{'LBP A/B':>12}{'hist A/B':>12}{'var A/B':>12}{'entropy A/B':>14}")
    for row in rows:
        cells = [f"{row[0]:<16}", f"{fmt(row[1]):>7}", f"{fmt(row[2], 3):>7}"]
        for i in range(len(FEATS)):
            a_, b_ = row[3 + 2 * i], row[4 + 2 * i]
            width = 14 if i in (0, 4) else 12
            cells.append(f"{fmt(a_) + '/' + fmt(b_):>{width}}")
        print("".join(cells))
    print(f"\nA = APCER, B = BPCER, in %; PSNR/SSIM against the untouched synthetic images. -> {a.out}")


if __name__ == "__main__":
    main()
