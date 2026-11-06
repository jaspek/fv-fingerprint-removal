# Result tables

All `results_pad_train_*.csv` files are tables of the harness pad_train (columns dataset, extractor,
protocol, train_attack, APCER, BPCER, ACER as fractions, `*_per_fold.csv` adds a fold column). Which
harness produced a table matters: the fold partition of GroupKFold changed between scikit-learn 1.7.2
and 1.9.1 (subject groups of equal size are assigned to the folds in a different order), so the
baselines differ by 1 to 3 points between the two, and the `fourier` default changed from the lower
thirds of the bands on IDIAP and SCUT to all 30 bands. Numbers from different harness versions must
not be mixed in one table.

| harness | files |
|---|---|
| pad-train 2.1.0 from PyPI, Python 3.12, scikit-learn 1.9.1. Defaults: `fourier` all 30 bands, Haralick scaled per fold, LBP raw | every `results_pad_train_*v210*.csv`, the tables of the top-level README |
| pad_train 1.1.0 with the LBP and scaler branches, Python 3.12, scikit-learn 1.9.1 | `results_pad_train_PLUS_gans.csv`, `results_pad_train_PLUS_gans_scaled.csv`: GAN variant 001 for every GAN, superseded by `results_pad_train_gans_v210.csv` |
| pad_train 1.0.2 with the LBP extractor (and the scaler option for `*scaled*`), Python 3.10, scikit-learn 1.7.2. `fourier` lower thirds on IDIAP (bands 0 to 19) and SCUT (0 to 9), Haralick unscaled unless `scaled` | all other `results_pad_train_*.csv` |

## pad-train 2.1.0

| file | content |
|---|---|
| `results_pad_train_gans_v210.csv` | the course table: PLUS, IDIAP, SCUT, the six sources (AE_Res, VAE, CycleGAN, DistanceGAN, DRIT, StarGANv2 with the prescribed variants), three extractors |
| `results_pad_train_v210_recheck_PLUS.csv` | cleaned AE_Res (notch + matching, rmin 0.05) and cleaned fixed-layer CycleGAN (rmin 0.02), twins from all real PAI |
| `results_pad_train_v210_recheck_IDIAP.csv` | white noise sigma 1.0 and 1.5 |
| `results_pad_train_v210_recheck_SCUT.csv` | white noise sigma 0.5 and 1.0 |
| `results_pad_train_v210_leakfree_PLUS.csv` (+ `_per_fold`) | notch + matching, rmin 0.05, twins from the training subjects of each fold |
| `results_pad_train_v210_leakfree_IDIAP.csv`, `..._SCUT.csv` (+ `_per_fold`) | noise-level matching per fold (equal-neighbour share, twins from the training subjects), mean sigma 1.4 on IDIAP and 0.3 on SCUT |
| `results_pad_train_v210_leakfree_IDIAP_match.csv` (+ `_per_fold`) | IDIAP: spectrum matching only (rmin 0.05), twins from the training subjects of each fold |
| `results_pad_train_v210_leakfree_SCUT_match.csv`, `..._fill.csv`, `..._chain.csv` (+ `_per_fold`) | SCUT: spectrum matching only (rmin 0.05), spectrum filling (rmin 0.45), notch then matching; twins from the training subjects of each fold |
| `results_pad_train_v210_leakfree_PLUS_drit.csv`, `..._cyclegan.csv`, `..._distancegan.csv`, `..._stargan-v2.csv` (+ `_per_fold`) | the chain (notch, then leak-free matching) on the course GAN sets of PLUS, variants 003, 003, 001, 001 |
| `results_pad_train_v210_PLUS_development.csv` | ablation on PLUS, twins from all real PAI: notch only, notch + ring matching, ring x sector matching at rmin 0 and strengths 0.5 and 0.75, rmin 0.03, the retrained CycleGAN sets untouched |
| `results_pad_train_v210_IDIAP_notch.csv` | notch only on IDIAP (harmful on all three extractors) |
| `results_pad_train_v210_SCUT_dither.csv` | dither probe on SCUT, LBP only: flat 3x3 neighbourhoods flipped by one grey level with probability 0.4 and 1.0 |
| `results_probe_PLUS_v210.csv` | radial-spectrum SVM probe: real attacks against untouched AE_Res, applied to the cleaned fold-0 set and the fixed-layer CycleGAN set before and after cleaning |

## pad_train 1.0.2, PLUS

| file | content |
|---|---|
| `results_pad_train_PLUS.csv` | untouched AE_Res and VAE sets, three extractors |
| `results_pad_train_PLUS_clean.csv` | notch only, notch + ring energy matching |
| `results_pad_train_PLUS_clean2.csv` | notch + ring x sector matching, rmin 0 |
| `results_pad_train_PLUS_clean2_strength.csv` | the same at strength 0.5 and 0.75 |
| `results_pad_train_PLUS_clean3.csv` | rmin 0.03 and 0.05 (0.05 is the operating point) |
| `results_pad_train_PLUS_leakfree*.csv` | operating point with twins from the training subjects of each fold |
| `results_pad_train_PLUS_scaled.csv` | scaled k-NN for AE_Res, cleaned AE_Res, CycleGAN-fsc and its cleaned version |
| `results_pad_train_PLUS_cyclegan.csv` | the retrained CycleGAN, stock and fixed smooth layer, untouched |
| `results_pad_train_PLUS_cyclegan_clean*.csv` | the chain on the fixed-layer set: rmin 0.05, matching only, rmin 0, rmin 0.02 |
| `results_pad_train_PLUS_CycleGAN-*_perfold*.csv` | per-fold protocol, synthetic samples only from training subjects |

## pad_train 1.0.2, IDIAP

| file | content |
|---|---|
| `results_pad_train_IDIAP.csv` | untouched AE_Res and VAE |
| `results_pad_train_IDIAP_leakfree*.csv` | notch + matching (harmful: the notch) |
| `results_pad_train_IDIAP_leakfree_nonotch*.csv`, `*_rmax*`, `*_r0*`, `*_r03*` | matching only, with window variants |
| `results_pad_train_IDIAP_noise.csv`, `*_noise2.csv`, `*_notch_noise.csv` | fixed white noise sigma 0.5 to 2.0, with and without notch |
| `results_pad_train_IDIAP_noisematch_std*.csv`, `*_ties*.csv` | noise level chosen per image from a twin: residual std, equal-neighbour share (the recipe) |
| `results_pad_train_IDIAP_haralick_scaled_noise.csv`, `*_nm_ties_haralick_scaled*.csv`, `results_pad_train_IDIAP_SCUT_haralick_scaled.csv` | the scaled Haralick against untouched, fixed noise and the tie-matching noise |

## pad_train 1.0.2, SCUT

| file | content |
|---|---|
| `results_pad_train_SCUT.csv` | untouched AE_Res and VAE |
| `results_pad_train_SCUT_leakfree*.csv`, `*_rmax*` | the chain, with and without the finest rings |
| `results_pad_train_SCUT_noise.csv`, `*_noisematch_ties*.csv` | fixed noise and tie-matching noise |
| `results_pad_train_SCUT_haralick_scaled_noise*.csv`, `*_nm_ties_haralick_scaled*.csv` | the scaled Haralick against noise sigma 0.25 to 1.0 and the tie-matching noise |

## Quality

`quality_*.csv` hold PSNR and SSIM per image of a cleaned PLUS set against the untouched one:
`clean2` is the chain with rmin 0, `clean2_s0.5` and `s0.75` the partial strengths, `clean3_r0.03`
and `r0.05` the window variants, `cyclegan_fsc_clean*` the chain on the fixed-layer CycleGAN set
(rmin 0.05, 0 and 0.02). `quality_PLUS_leakfree_v210_fold0.csv`, `quality_IDIAP_nmties_v210_fold0.csv`,
`quality_SCUT_nmties_v210_fold0.csv` and `quality_SCUT_noise0.5.csv` are the 2.1.0 recipes on fold 0 (PSNR is
inf for images the noise matching left unchanged: 10 on IDIAP, 274 on SCUT).

## Smoke test

`smoke_subset_sweep/` is the first sweep on eight PLUS users and 120 synthetic images with the
repository's own k-NN (`tools/pad_knn.py`), one `pad_<extractor>_<setting>.csv` and `q_<setting>.csv`
per setting and the summary `sweep_summary.csv`. These are pipeline checks, not results.
