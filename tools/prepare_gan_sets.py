"""Extract the GAN samples that pad_train uses from a local copy of Data_prepared.zip.

Same archive members and the same SCUT rotation as `pad-train download`, taken from the zip on
disk instead of a download. Members that already exist below <root>/Data_prepared are skipped, so
the script can be rerun after the package changed its variant table (as in 2.1.0). Run with the
Python 3.12 interpreter that has pad-train installed:

    python prepare_gan_sets.py --data-root <root> --zip <path to Data_prepared.zip>

Leaves the zip untouched.
"""
import argparse
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import pad_train
from pad_train.data import _matches, rotate_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--zip", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = pad_train.settings
    settings.data_root = Path(a.data_root)
    # The GAN sources share one archive, one member list and the SCUT rotation; any of them will do.
    gan_sources = [s for s in settings.synthetic_sources.values() if s.members]
    if not gan_sources:
        raise SystemExit("no synthetic source with archive members found (pad-train >= 1.1.0 needed)")
    source = gan_sources[0]
    target = settings.resolve(source)
    rotate = source.rotate or {}
    missing = [m for m in source.members if not (target / m).is_dir()]
    if not missing:
        print(f"nothing to do, all {len(source.members)} member folders exist below {target}")
        return
    print(f"{len(missing)} of {len(source.members)} member folders missing, extracting from {a.zip}")

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip) as zip_file, tempfile.TemporaryDirectory(dir=target.parent, prefix=".Data_prepared-") as tmp:
        staging = Path(tmp)
        names = [n for n in zip_file.namelist() if _matches(n, missing)]
        if not names:
            raise SystemExit(f"none of {missing} found in the archive")
        zip_file.extractall(staging, members=names)
        for member in missing:
            folder = staging / member
            if not folder.is_dir():
                print(f"  not in archive: {member}")
                continue
            if member in rotate:
                print(f"  turning {rotate_images(folder, rotate[member])} images in {member}")
            (target / member).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), str(target / member))
            n_files = sum(1 for p in (target / member).rglob("*") if p.is_file())
            print(f"  {n_files:5d} files -> {member}")


if __name__ == "__main__":
    main()
