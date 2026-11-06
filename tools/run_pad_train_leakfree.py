"""Leak-free evaluation of the cleaning chain in pad_train: per fold, the spectrum-matching reference
consists only of the real PAI images of that fold's TRAINING subjects, so no test image ever serves
as a matching twin. Folds are pad_train's own (GroupKFold over subjects, same sample order).

Steps per fold i:
  1. copy the training subjects' real PAI images to <data_root>/ref_<out_prefix>_fold<i>/
  2. apply_removal.py --op <op_name>:<OPTS> --ref ref_... on the (already notched) synthetic set
     -> <data_root>/<out_prefix>-fold<i>/<synthetic_dir>/spoof/samples
  3. pad_train.cross_validate on precomputed features with that set as the synthetic source, keep the row of fold i
Finally the five fold rows are averaged, next to the baseline rows (trained on real PAI).

The features of the bona fide and real PAI images are extracted once per extractor and reused for every fold and
for the baseline (pad_train.evaluate would extract them again for every fold), which makes a SCUT run about three
times faster. The numbers are identical to evaluate(per_fold=True): same features, same folds, same sub-sampling.

Checkpoints: the per-fold table <out>_per_fold.csv is written after every fold; with --resume, folds already in it
are skipped and complete fold folders are reused (partial ones are cleaned again).

Two leak-free stages chain with --per-fold-input: --notched then names the prefix of the first stage's fold folders and
fold i reads <notched>-fold<i>/<synthetic dir>, so every stage of fold i only ever saw that fold's training subjects.

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
from pad_train.config import settings
from pad_train.data import image_files, parallel_map, read_image
from pad_train.evaluation import ID_COLUMNS, METRIC_COLUMNS, cross_validate, extract_features
from pad_train.features.registry import get_extractor

TOOLS = Path(__file__).resolve().parent
COLUMNS = [*ID_COLUMNS, "fold", *METRIC_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="PLUS")
    ap.add_argument("--notched", default="AE_Res-notch", help="synthetic source folder that already went through the notch")
    ap.add_argument("--per-fold-input", action="store_true", help="--notched is a per-fold prefix: fold i reads <notched>-fold<i>/<synthetic dir>, the output of an earlier leak-free stage")
    ap.add_argument("--spoof-dir", default=None, help="real PAI folder with one sub-folder per subject (default: PLUS layout)")
    ap.add_argument("--match", default="strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05", help="specmatch options")
    ap.add_argument("--op-name", default="specmatch", help="operator that takes --ref: specmatch, specfill or noisematch")
    ap.add_argument("--extractors", nargs="+", default=["fourier", "haralick", "lbp"])
    ap.add_argument("--out-prefix", default="AE_Res-leakfree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="only print the fold split and the planned commands")
    ap.add_argument("--reuse", action="store_true", help="skip the cleaning when the fold folder already exists and is complete (evaluate only)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run: folds with a row in <out>_per_fold.csv are skipped, complete fold folders are reused")
    ap.add_argument("--scale", action="store_true", help="standardise the features before the k-NN (StandardScaler fitted on the training fold only); applies to the baseline rows too")
    a = ap.parse_args()

    root = Path(a.data_root)
    settings.data_root = root
    settings.scale_features = True if a.scale else None  # None = the harness default (pad-train >= 2.0: scaled Haralick, raw Fourier and LBP)
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

    def input_dir(fold):
        return root / (f"{a.notched}-fold{fold}" if a.per_fold_input else a.notched) / synthetic_rel
    notched_dir = input_dir(0)
    if not notched_dir.is_dir():
        raise SystemExit(f"input synthetic folder not found: {notched_dir}")

    # Same fold split as pad_train.cross_validate: GroupKFold over subjects, samples ordered real then spoof.
    data = pad_train.load_data(a.dataset, (), False, settings)
    subjects = np.concatenate([data.real_subjects, data.spoof_subjects])
    splits = list(GroupKFold(n_splits=settings.n_folds).split(np.zeros((len(subjects), 1)), groups=subjects))

    extractors = {name: get_extractor(name, layout) for name in a.extractors}
    scale = {name: settings.scale_features if settings.scale_features is not None else getattr(ex, "requires_scaling", False)
             for name, ex in extractors.items()}
    cache = {}

    def real_features(name):
        """Features of the bona fide and real PAI images, extracted once per extractor."""
        if name not in cache:
            cache[name] = (extract_features(data.real, extractors[name]), extract_features(data.spoof, extractors[name]))
            print(f"    {name}: features of {len(data.real)} bona fide and {len(data.spoof)} real PAI images cached")
        return cache[name]

    def evaluate_fold(src_name, folder, fold):
        images = parallel_map(read_image, image_files(folder), 1)
        frames = []
        for name, ex in extractors.items():
            real_f, spoof_f = real_features(name)
            table = cross_validate(real_f, data.real_subjects, spoof_f, data.spoof_subjects, {src_name: extract_features(images, ex)},
                                   layout.n_neighbors, protocols=["synthetic"], balance_synthetic=True, n_folds=settings.n_folds,
                                   random_state=settings.random_state, scale_features=scale[name])
            frames.append(table[table["fold"] == fold].assign(dataset=a.dataset, extractor=name))
        return pd.concat(frames)[COLUMNS]

    per_fold_path = Path(a.out).with_name(Path(a.out).stem + "_per_fold.csv")
    done = {}
    if a.resume and per_fold_path.exists():
        prev = pd.read_csv(per_fold_path)
        done = {int(f): g for f, g in prev.groupby("fold")}
        print(f"resuming: folds {sorted(done)} already evaluated in {per_fold_path.name}")
    def count_images(d):
        return sum(1 for f in d.rglob("*") if f.suffix.lower() in exts) if d.is_dir() else 0

    def complete(d, n_in):
        return d.is_dir() and count_images(d) == n_in

    fold_rows = []
    for i, (train_idx, _test_idx) in enumerate(splits):
        if i in done:
            fold_rows.append(done[i])
            print(f"== fold {i}: checkpoint found, skipped")
            continue
        train_subjects = sorted(set(subjects[train_idx]))
        ref_dir = root / f"ref_{a.out_prefix}_fold{i}"
        src_name = f"{a.out_prefix}-fold{i}"
        out_dir = root / src_name / synthetic_rel
        n_ref = sum(len(pai_files(s)) for s in train_subjects)
        src_in = input_dir(i)
        n_in = count_images(src_in)
        if n_in == 0:
            raise SystemExit(f"input synthetic folder of fold {i} missing or empty: {src_in}")
        print(f"== fold {i}: {len(train_subjects)} training subjects, {n_ref} reference PAI images, {n_in} inputs from {src_in.parts[len(root.parts)]} -> {src_name}")
        cmd = [sys.executable, str(TOOLS / "apply_removal.py"), "--in", str(src_in), "--out", str(out_dir),
               "--op", f"{a.op_name}:{a.match}", "--ref", str(ref_dir)]
        if a.dry_run:
            print("   ", " ".join(cmd))
            continue
        if (a.reuse or a.resume) and complete(out_dir, n_in):
            print(f"    reusing {out_dir} ({n_in} images)")
        else:
            if out_dir.is_dir():
                shutil.rmtree(out_dir)                                   # a partial folder from an interrupted run
            if ref_dir.exists():
                shutil.rmtree(ref_dir)
            for s in train_subjects:
                for f in pai_files(s):
                    (ref_dir / s).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, ref_dir / s / f.name)
            subprocess.run(cmd, check=True)
            shutil.rmtree(ref_dir)                                       # the twins are only needed for this cleaning
        row = evaluate_fold(src_name, out_dir, i)
        row["train_attack"] = "leak-free"
        fold_rows.append(row)
        print(row.to_string(index=False))
        pd.concat(fold_rows).to_csv(per_fold_path, index=False)         # checkpoint after every fold

    if a.dry_run:
        return
    folds = pd.concat(fold_rows)
    mean = folds.groupby(["dataset", "extractor", "protocol", "train_attack"], as_index=False)[["APCER", "BPCER", "ACER"]].mean()
    frames = []
    for name in extractors:
        real_f, spoof_f = real_features(name)
        table = cross_validate(real_f, data.real_subjects, spoof_f, data.spoof_subjects, {}, layout.n_neighbors, protocols=["baseline"],
                               n_folds=settings.n_folds, random_state=settings.random_state, scale_features=scale[name])
        frames.append(table.assign(dataset=a.dataset, extractor=name))
    baseline = pd.concat(frames).groupby(ID_COLUMNS, sort=False)[METRIC_COLUMNS].mean().reset_index()
    result = pd.concat([baseline, mean]).sort_values(["extractor", "protocol"])
    result.to_csv(a.out, index=False)
    folds.to_csv(str(per_fold_path), index=False)
    print("\n== leak-free result (mean over the five folds) next to the baseline")
    print(result.to_string(index=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
