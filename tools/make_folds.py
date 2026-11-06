"""Build five user-partitioned, balanced folds for one dataset.

Example:
  python make_folds.py --dataset plus --real .../PLUS/real --spoof .../PLUS/spoof --out folds_plus.csv

Output CSV columns: path, label (0 = bona fide, 1 = real PAI), user, fold.
Only users that appear in BOTH real and spoof folders are kept (balanced sets, as on the course page).
"""
import argparse, csv, os, random
from common import list_images, user_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['plus', 'idiap', 'scut'])
    ap.add_argument('--real', required=True)
    ap.add_argument('--spoof', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    real = [(p, 0, user_id(p, a.dataset)) for p in list_images(a.real)]
    spoof = [(p, 1, user_id(p, a.dataset)) for p in list_images(a.spoof)]
    users_real, users_spoof = {u for _, _, u in real}, {u for _, _, u in spoof}
    users = sorted(users_real & users_spoof)
    if not users:
        raise SystemExit('No common users between real and spoof; check --dataset naming')
    random.Random(a.seed).shuffle(users)
    fold_of = {u: i % a.k for i, u in enumerate(users)}

    rows = [(p, lab, u, fold_of[u]) for p, lab, u in real + spoof if u in fold_of]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['path', 'label', 'user', 'fold']); w.writerows(rows)

    n_real = sum(1 for r in rows if r[1] == 0); n_spoof = len(rows) - n_real
    print(f'{a.dataset}: {len(users)} users in both classes -> {n_real} bona fide, {n_spoof} PAI, '
          f'{a.k} folds; dropped {len(users_real - users_spoof)} real-only and '
          f'{len(users_spoof - users_real)} spoof-only users. Written {a.out}')


if __name__ == '__main__':
    main()
