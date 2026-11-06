"""Give the anonymised IDIAP test files their original names (and real/spoof folders).

Example:
  python decode_idiap_test.py --test .../IDIAP/full/test --translator .../IDIAP/translator_full_files.txt --out .../IDIAP_decoded/full
Creates out/real/<name>.png and out/spoof/<name>.png (copies, originals untouched).
Use the matching translator for 'full' vs 'cropped' (they are different mappings).
"""
import argparse, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required=True)
    ap.add_argument('--translator', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    n = 0
    for line in open(a.translator):
        parts = line.split()
        if len(parts) != 2:
            continue
        anon, orig = parts
        cls = 'spoof' if orig.endswith('_spoof.png') else 'real'
        dst = os.path.join(a.out, cls, orig)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(a.test, anon), dst); n += 1
    print(f'decoded {n} files into {a.out}/real and {a.out}/spoof')


if __name__ == '__main__':
    main()
