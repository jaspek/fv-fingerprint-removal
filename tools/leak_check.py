"""How much of the chain's low APCER is the matching leak?

apply_removal.py specmatch mode=1 draws, for each input image (list_images order, no limit),
rng=default_rng(0).integers(len(ref profiles)) with ref profiles = list_images(ref, 500) = all 144
real PAI of the 8 users, sorted. That reconstructs which real PAI image each chain output was
matched to ("twin"). Here: distance of every real-PAI test image to its twin chain outputs vs to
the nearest non-twin chain outputs vs to bona fide, and the k-NN replica with the twins of the
test image removed from the training set (the leak-free version of the single split).

Run from this folder: python -I leak_check.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, r'D:\Studium\Master\MultimediaSecurity\PS\tools')
from common import list_images, load_gray   # noqa: E402
from pad_knn import f_fourier               # noqa: E402

S = r'D:\Studium\Master\MultimediaSecurity\PS\smoke_subset'
REF = os.path.join(S, 'PLUS', 'spoof')
BONA = os.path.join(S, 'PLUS', 'real')
K, LIMIT = 15, 120


def feats(files):
    return np.array([f_fourier(load_gray(p)) for p in files])


def knn(Xb, Xv, Xt, k=K, exclude=None):
    """exclude[i] = set of variant row indices not allowed as neighbours of test row i."""
    Xtr = np.vstack([Xb, Xv])
    ytr = np.r_[np.zeros(len(Xb)), np.ones(len(Xv))]
    D = np.sqrt(((Xt[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1))
    if exclude is not None:
        for i, ex in enumerate(exclude):
            for j in ex:
                D[i, len(Xb) + j] = np.inf
    nn = np.argsort(D, axis=1)[:, :k]
    yhat = (ytr[nn].mean(1) > 0.5).astype(int)
    return 1.0 - yhat.mean()


def main():
    ref_files = list_images(REF, 500)                       # 144 sorted, as make_specmatch saw them
    test_files = list_images(REF, LIMIT)                     # the 120-subset used in band_analysis
    test_idx = [ref_files.index(p) for p in test_files]
    Xb = feats(list_images(BONA, LIMIT))
    Xt = feats(test_files)
    print(f'ref images {len(ref_files)}, test subset {len(test_files)}, bona {len(Xb)}')
    for chain_in, chain_out in [('pk1.0_s7', 'pk1.0_s7_sm'), ('PLUS_spoof', 'sm'), ('pk2.0_s7', 'pk2.0_s7_sm')]:
        in_files = list_images(os.path.join(S, 'AE_Res', chain_in))      # order used by apply_removal
        rng = np.random.default_rng(0)
        twin = np.array([rng.integers(len(ref_files)) for _ in in_files])   # ref index per output
        out_files = list_images(os.path.join(S, 'AE_Res', chain_out))
        assert [os.path.splitext(os.path.basename(p))[0] for p in in_files] == \
               [os.path.splitext(os.path.basename(p))[0] for p in out_files]
        Xv = feats(out_files)
        D = np.sqrt(((Xt[:, None, :] - Xv[None, :, :]) ** 2).sum(-1))
        Db = np.sqrt(((Xt[:, None, :] - Xb[None, :, :]) ** 2).sum(-1))
        is_twin = np.array([[twin[j] == test_idx[i] for j in range(len(Xv))] for i in range(len(Xt))])
        n_twins = is_twin.sum(1)
        d_twin = np.array([D[i, is_twin[i]].mean() if n_twins[i] else np.nan for i in range(len(Xt))])
        d_non = np.array([np.sort(D[i, ~is_twin[i]])[:K].mean() for i in range(len(Xt))])
        d_bona = np.sort(Db, axis=1)[:, :K].mean(1)
        ap_all = knn(Xb, Xv, Xt)
        ap_noleak = knn(Xb, Xv, Xt, exclude=[np.where(is_twin[i])[0] for i in range(len(Xt))])
        print(f'\n{chain_out}: test images with >=1 twin in the 120 outputs: {(n_twins > 0).sum()}/{len(Xt)} '
              f'(mean twins per test image {n_twins.mean():.2f})')
        print(f'  mean distance test -> its twin outputs        {np.nanmean(d_twin):.3f}')
        print(f'  mean distance test -> 15 nearest NON-twin outputs {d_non.mean():.3f}')
        print(f'  mean distance test -> 15 nearest bona fide    {d_bona.mean():.3f}')
        print(f'  k-NN replica APCER with twins allowed {100*ap_all:.1f} %, twins excluded {100*ap_noleak:.1f} %')


if __name__ == '__main__':
    main()
