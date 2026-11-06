"""Leak-free evaluation of the cleaning chain in pad_train: per fold, the spectrum-matching reference
consists only of the real PAI images of that fold's TRAINING subjects, so no test image ever serves
as a matching twin. Folds are pad_train's own (GroupKFold over subjects, same sample order).

Steps per fold i:
  1. copy the training subjects' real PAI images to <data_root>/ref_fold<i>/
  2. apply_removal.py --op specmatch:<OPTS> --ref ref_fold<i> on the (already notched) synthetic set
     -> <data_root>/<out_prefix>-fold<i>/PLUS_matched/spoof/samples
  3. pad_train.evaluate(per_fold=True) with that set as synthetic source, keep the row of fold i
Finally the five fold rows are averaged, next to the baseline rows (trained on real PAI).

Example (PowerShell, from PS/tools):
  python run_pad_train_leakfree.py --data-root <data root> --notched AE_Res-notch --out <work>/results_pad_train_PLUS_leakfree.csv
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import pad_train
from pad_train.config import DataSource, settings

TOOLS = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="PLUS")
    ap.add_argument("--notched", default="AE_Res-notch", help="synthetic source folder that already went through the notch")
    ap.add_argument("--spoof-dir", default=None, help="real PAI folder with one sub-folder per subject (default: PLUS layout)")
    ap.add_argument("--match", default="strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05", help="specmatch options")
    ap.add_argument("--op-name", default="specmatch", help="operator that takes --ref: specmatch or noisematch")
    ap.add_argument("--extractors", nargs="+", default=["fourier", "haralick", "lbp"])
    ap.add_argument("--out-prefix", default="AE_Res-leakfree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="only print the fold split and the planned commands")
    ap.add_argument("--reuse", action="store_true", help="skip the cleaning when the fold folder already exists (evaluate only)")
    ap.add_argument("--scale", action="store_true", help="standardise the features before the k-NN (StandardScaler fitted on the training fold only); applies to the baseline rows too")
    a = ap.parse_args()

    root = Path(a.data_root)
    settings.data_root = root
    settings.scale_features = a.scale
    layout = settings.datasets[a.dataset]
    # real PAI folders: one per subset (PLUS: none, IDIAP: train/dev, SCUT: train/dev/test), subject = sub-folder
    subsets = getattr(layout, "subsets", ("",)) or ("",)
    spoof_dirs = [Path(a.spoof_dir)] if a.spoof_dir else [root / settings.real_source.path / layout.real_dir / sub / "spoof" for sub in subsets]
    spoof_dirs = [d for d in spoof_dirs if d.is_dir()]
    if not spoof_dirs:
        raise SystemExit(f"no real PAI folder found below {root / settings.real_source.path / layout.real_dir}")
    exts = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

    def pai_files(subject):
        return [f for d in spoof_dirs if (d / subject).is_dir() for f in (d / subject).iterdir() if f.suffix.lower() in exts]
    synthetic_rel = Path(layout.synthetic_dir) / settings.synthetic_attack_subdir
    notched_dir = root / a.notched / synthetic_rel
    if not notched_dir.is_dir():
        raise SystemExit(f"notched synthetic folder not found: {notched_dir}")

    # Same fold split as pad_train.cross_validate: GroupKFold over subjects, samples ordered real then spoof.
    data = pad_train.load_data(a.dataset, (), False, settings)
    subjects = np.concatenate([data.real_subjects, data.spoof_subjects])
    splits = list(GroupKFold(n_splits=settings.n_folds).split(np.zeros((len(subjects), 1)), groups=subjects))

    fold_rows = []
    for i, (train_idx, _test_idx) in enumerate(splits):
        train_subjects = sorted(set(subjects[train_idx]))
        ref_dir = root / f"ref_fold{i}"
        src_name = f"{a.out_prefix}-fold{i}"
        out_dir = root / src_name / synthetic_rel
        n_ref = sum(len(pai_files(s)) for s in train_subjects)
        print(f"== fold {i}: {len(train_subjects)} training subjects, {n_ref} reference PAI images -> {src_name}")
        cmd = [sys.executable, str(TOOLS / "apply_removal.py"), "--in", str(notched_dir), "--out", str(out_dir),
               "--op", f"{a.op_name}:{a.match}", "--ref", str(ref_dir)]
        if a.dry_run:
            print("   ", " ".join(cmd))
            continue
        if a.reuse and out_dir.is_dir():
            print(f"    reusing {out_dir}")
        else:
            if ref_dir.exists():
                shutil.rmtree(ref_dir)
            for s in train_subjects:
                for f in pai_files(s):
                    (ref_dir / s).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, ref_dir / s / f.name)
            subprocess.run(cmd, check=True)
        settings.synthetic_sources[src_name] = DataSource(name=src_name, path=Path(src_name), url=None)
        table = pad_train.evaluate(datasets=[a.dataset], extractors=a.extractors, synthetic=[src_name],
                                   protocols=["synthetic"], per_fold=True, download=False)
        row = table[table["fold"] == i].copy()
        row["train_attack"] = "leak-free"
        fold_rows.append(row)
        print(row.to_string(index=False))

    if a.dry_run:
        return
    folds = pd.concat(fold_rows)
    mean = folds.groupby(["dataset", "extractor", "protocol", "train_attack"], as_index=False)[["APCER", "BPCER", "ACER"]].mean()
    baseline = pad_train.evaluate(datasets=[a.dataset], extractors=a.extractors, synthetic=[], protocols=["baseline"], download=False)
    result = pd.concat([baseline, mean]).sort_values(["extractor", "protocol"])
    result.to_csv(a.out, index=False)
    folds.to_csv(str(Path(a.out).with_name(Path(a.out).stem + "_per_fold.csv")), index=False)
    print("\n== leak-free result (mean over the five folds) next to the baseline")
    print(result.to_string(index=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
