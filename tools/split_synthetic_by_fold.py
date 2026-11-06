"""Split a synthetic source whose samples derive from known subjects into per-fold sources.

The CycleGAN translations were made from the bona fide images of all subjects, so a sample made
from a test subject's image would sit in the attack class of that fold's training set. This keeps,
per fold, only the samples made from that fold's training subjects (same subject folds as pad_train):

    python split_synthetic_by_fold.py --data-root <root> --dataset PLUS --source CycleGAN-fsc-clean-r002
        --names-from <CycleGAN results folder with the *_fake.png files> --out-prefix CycleGAN-fsc-clean-r002

Writes <root>/<out-prefix>-fold<i>/<synthetic_dir>/spoof/samples/ for i = 0..4.
export_synthetic.py numbered the sorted *_fake.png files as sample_<i>.png, so the i-th sorted name
gives the subject of sample_i: the prefix before the first underscore, added by prepare_cyclegan_data.py.
Evaluate afterwards with
    python run_pad_train_leakfree.py --data-root <root> --dataset PLUS --notched <source> --out-prefix <out-prefix> --reuse ...
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import pad_train
from pad_train.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="PLUS")
    ap.add_argument("--source", required=True, help="synthetic source folder below the data root")
    ap.add_argument("--names-from", required=True, help="folder with the *_fake.png files the source was exported from")
    ap.add_argument("--out-prefix", required=True, help="fold folders are <out-prefix>-fold<i>, as in run_pad_train_leakfree.py")
    a = ap.parse_args()

    root = Path(a.data_root)
    settings.data_root = root
    layout = settings.datasets[a.dataset]
    rel = Path(layout.synthetic_dir) / settings.synthetic_attack_subdir
    src = root / a.source / rel
    names = sorted(Path(a.names_from).glob("*_fake.png")) or sorted(Path(a.names_from).glob("*_fake_B.png"))
    samples = sorted(src.glob("sample_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not names or len(names) != len(samples):
        raise SystemExit(f"{len(names)} names in {a.names_from} but {len(samples)} samples in {src}")
    subject_of = [n.name.split("_")[0] for n in names]

    # Same fold split as pad_train.cross_validate: GroupKFold over subjects, samples ordered real then spoof.
    data = pad_train.load_data(a.dataset, (), False, settings)
    subjects = np.concatenate([data.real_subjects, data.spoof_subjects])
    splits = GroupKFold(n_splits=settings.n_folds).split(np.zeros((len(subjects), 1)), groups=subjects)
    for i, (train_idx, _test_idx) in enumerate(splits):
        train = set(subjects[train_idx])
        out = root / f"{a.out_prefix}-fold{i}" / rel
        out.mkdir(parents=True, exist_ok=True)
        kept = 0
        for sample, subject in zip(samples, subject_of):
            if subject in train:
                shutil.copy2(sample, out / sample.name)
                kept += 1
        print(f"fold {i}: {len(train)} training subjects, {kept} of {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
