"""Build a CycleGAN dataset folder from the finger-vein data: bona fide images as domain A, real PAI as domain B.

    python prepare_cyclegan_data.py --data-root <data root> --dataset PLUS --out datasets/plus_fv
        [--exclude-subjects 001 002 ...]   subjects left out of trainA/trainB (for a leak-free protocol)
        [--limit-test 2048]                how many bona fide images to put into testA (these are translated to synthetic PAI)

Layout written (what pytorch-CycleGAN-and-pix2pix expects):
    <out>/trainA  bona fide          <out>/trainB  real PAI
    <out>/testA   bona fide images to translate into synthetic PAI (A -> B)
PLUS: real/<subject>/*.png and spoof/<subject>/*.png. IDIAP and SCUT: the train/dev(/test) subsets are merged.
"""
import argparse
import shutil
from pathlib import Path

LAYOUT = {"PLUS": ("PLUS", ("",)), "IDIAP": ("IDIAP/full", ("train", "dev")), "SCUT": ("SCUT/full", ("train", "dev", "test"))}
EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="PLUS", choices=list(LAYOUT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-subjects", nargs="*", default=[])
    ap.add_argument("--limit-test", type=int, default=0, help="0 = all bona fide images into testA")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = all images per class (smaller numbers for smoke tests)")
    a = ap.parse_args()

    real_dir, subsets = LAYOUT[a.dataset]
    base = Path(a.data_root) / "Data_FV_Spoofing_WS2025_26" / real_dir
    out = Path(a.out)
    counts = {}
    for cls, dom in (("real", "A"), ("spoof", "B")):
        files = []
        for sub in subsets:
            d = base / sub / cls
            if not d.is_dir():
                continue
            for subj in sorted(p for p in d.iterdir() if p.is_dir()):
                if subj.name in a.exclude_subjects:
                    continue
                files += sorted(f for f in subj.iterdir() if f.suffix.lower() in EXTS)
        files = files[: a.limit_train or None]
        dst = out / f"train{dom}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f"{f.parent.name}_{f.name}")
        counts[cls] = len(files)
        if cls == "real":
            test = out / "testA"
            test.mkdir(parents=True, exist_ok=True)
            for f in files[: a.limit_test or None]:
                shutil.copy2(f, test / f"{f.parent.name}_{f.name}")
            counts["testA"] = len(files[: a.limit_test or None])
    print(f"{a.dataset}: trainA (bona fide) {counts['real']}, trainB (real PAI) {counts['spoof']}, testA {counts['testA']} -> {out}")


if __name__ == "__main__":
    main()
