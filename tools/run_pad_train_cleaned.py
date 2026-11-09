"""Evaluate cleaned synthetic PAI sets with the pad_train harness (all three extractors, same folds).

A cleaned set is a plain folder tree <name>/PLUS_matched/spoof/samples/*.png below the pad_train
data root, produced with apply_removal.py. Each --source NAME registers <data_root>/NAME as a
synthetic source next to AE_Res and VAE.

Example (PowerShell):
  python run_pad_train_cleaned.py --data-root <data root> --source AE_Res-clean --datasets PLUS --out <work>/results_pad_train_PLUS_clean.csv
"""
import argparse
from pathlib import Path

import pad_train
from pad_train.config import DataSource, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source", action="append", required=True, help="folder name below the data root (repeatable)")
    ap.add_argument("--datasets", nargs="+", default=["PLUS"])
    ap.add_argument("--extractors", nargs="+", default=["fourier", "haralick_mahotas", "lbp"])
    ap.add_argument("--per-fold", action="store_true")
    ap.add_argument("--scale", action="store_true", help="standardise the features before the k-NN (StandardScaler fitted on the training fold only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    settings.data_root = Path(a.data_root)
    settings.scale_features = True if a.scale else None  # None = the harness default (pad-train >= 2.0: scaled Haralick, raw Fourier and LBP)
    for name in a.source:
        if not (settings.data_root / name).is_dir():
            raise SystemExit(f"missing folder: {settings.data_root / name}")
        settings.synthetic_sources[name] = DataSource(name=name, path=Path(name), url=None)

    table = pad_train.evaluate(datasets=a.datasets, extractors=a.extractors, synthetic=a.source,
                               protocols=["baseline", "synthetic"], per_fold=a.per_fold, download=False)
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
