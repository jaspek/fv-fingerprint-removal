"""Collect the A->B translations of a CycleGAN test run into the pad_train synthetic-source layout.

    python export_synthetic.py --results pytorch-CycleGAN-and-pix2pix/results/plus_fsc/test_latest/images
                               --data-root <data root> --name CycleGAN-fsc --dataset PLUS

Writes <data-root>/<name>/<synthetic_dir>/spoof/samples/sample_<i>.png (grayscale), where synthetic_dir is
PLUS_matched, IDIAP or SCUT as in pad_train's config. Evaluate afterwards with
    python ../tools/run_pad_train_cleaned.py --data-root <data-root> --source <name> --datasets <dataset> --out ...
"""
import argparse
from pathlib import Path

from PIL import Image

SYN_DIR = {"PLUS": "PLUS_matched", "IDIAP": "IDIAP", "SCUT": "SCUT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="CycleGAN results/<name>/test_latest/images folder")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--name", required=True, help="synthetic source name, e.g. CycleGAN-fsc")
    ap.add_argument("--dataset", default="PLUS", choices=list(SYN_DIR))
    a = ap.parse_args()

    src = sorted(Path(a.results).glob("*_fake.png")) or sorted(Path(a.results).glob("*_fake_B.png"))   # "test" model writes *_fake.png
    if not src:
        raise SystemExit(f"no *_fake.png or *_fake_B.png in {a.results}")
    dst = Path(a.data_root) / a.name / SYN_DIR[a.dataset] / "spoof" / "samples"
    dst.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(src):
        Image.open(f).convert("L").save(dst / f"sample_{i}.png")
    print(f"{len(src)} synthetic PAI -> {dst}")


if __name__ == "__main__":
    main()
