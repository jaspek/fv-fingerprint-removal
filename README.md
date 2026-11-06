# fv-fingerprint-removal

Removing the model fingerprint from synthetic finger-vein presentation attack samples, and
measuring whether a presentation attack detector (PAD) trained on the cleaned samples still
detects real attacks.

Proseminar Media Security, University of Salzburg, winter term 2026/27. The evaluation harness is
the course package [pad-train](https://pypi.org/project/pad-train/)
([source](https://github.com/Kneidl18/PAD_train)). Its LBP extractor and its per-fold feature
scaling come from this project (pull requests 9 and 10).

## The problem

Synthetic attack samples from GANs and diffusion models carry a model fingerprint: too little
energy at high spatial frequencies (no sensor noise floor) and periodic peaks from the
upsampling layers. A k-NN PAD trained on bona fide versus synthetic attacks learns "camera
versus generator" instead of "finger versus attack" and then misses almost every real attack:
on PLUS the Fourier-band and LBP detectors reach an APCER of 98 % while a PAD trained on real
attacks reaches 5 % and 0 %.

## What we did

Two frequency-domain operators, both leaving the phase untouched, plus one pixel-domain operator:

- **Adaptive notch** (`peaks`): bins of the log spectrum that stick out above a local median
  window by more than a threshold are pressed down.
- **Spectrum matching** (`specmatch`): the energy of each ring and orientation sector of the
  spectrum (64 rings times 8 sectors) is rescaled to that of a randomly drawn real attack
  image, the twin. Rings below `rmin` cycles per pixel are left alone.
- **Noise-level matching** (`noisematch`): white noise is added per image until the share of
  equal-valued neighbour pixels matches that of a real attack twin, so the noise level is chosen
  by the data.
- **Spectrum filling** (`specfill`): random-phase noise whose energy per ring and sector is the
  deficit of the image against its twin, so nothing is taken away and the phase stays untouched.
- `dither` (probe only): flips flat 3x3 neighbourhoods by one grey level to change the share of
  equal-valued neighbours without touching the spectrum.

## Results

pad-train 2.1.0 (Python 3.12, scikit-learn 1.9.1) with its defaults: `fourier` with all 30 bands,
Haralick scaled per fold, LBP raw. All cleaned rows are leak-free: the twins, or the noise level,
of each cross-validation fold come only from that fold's training subjects. APCER in percent of a
PAD trained on the synthetic set and tested on real attacks, five subject-disjoint folds,
baseline = trained on real attacks.

| dataset | recipe | Fourier | LBP | Haralick |
|---|---|---|---|---|
| PLUS untouched | | 98.0 | 98.4 | 0.0 |
| PLUS cleaned | notch + spectrum matching, rmin 0.05 | 11.6 | 0.7 | 0.0 |
| PLUS baseline | | 4.8 | 0.0 | 0.0 |
| IDIAP untouched | | 79.6 | 74.6 | 51.3 |
| IDIAP cleaned | noise-level matching, sigma 1.4 on average | 0.0 | 5.8 | 2.1 |
| IDIAP cleaned | spectrum matching only, rmin 0.05 | 0.0 | 20.4 | 1.7 |
| IDIAP baseline | | 0.0 | 2.9 | 1.3 |
| SCUT untouched | | 94.1 | 100.0 | 59.6 |
| SCUT cleaned | spectrum matching only, rmin 0.05 | 11.6 | 99.8 | 29.3 |
| SCUT cleaned | spectrum filling, rmin 0.45, sigma 0.65 on average | 10.8 | 100.0 | 24.5 |
| SCUT cleaned | notch + spectrum matching, rmin 0.05 | 12.1 | 99.7 | 32.5 |
| SCUT cleaned | noise-level matching, sigma 0.3 on average | 23.0 | 99.8 | 30.4 |
| SCUT cleaned | white noise, sigma 0.5 (sweep, not a recipe) | 11.8 | 100.0 | 30.1 |
| SCUT baseline | | 4.6 | 1.4 | 0.1 |

PLUS and IDIAP are closed on all three extractors, IDIAP by the added noise: spectrum matching
alone closes its Fourier and Haralick detectors but leaves LBP at 20 % (one fold at 50 %), because
rescaling nearly empty spectrum cells cannot supply the missing noise floor that the LBP codes see.
On SCUT it is the other way round: spectrum matching brings the Fourier detector from 94 to 12 %
without any added noise, and spectrum filling (random-phase noise with the
per-cell energy deficit of the twin, rings above 0.45 cycles per pixel only) gives the same within
the fold spread. The notch adds nothing on SCUT. Two signatures stay: the Haralick detector keeps a
floor near 25 to 30 %, and LBP does not move under any spectral operator or noise level, nor under
the dither probe: flipping 40 % or all of the flat 3x3 neighbourhoods by one grey level brings the
share of equal-valued neighbours from 0.41 to 0.36 and 0.33 (the tie-matched set sits at 0.35) and
leaves LBP at 100 % (`results_pad_train_v210_SCUT_dither.csv`), so the LBP signature is not the
tie count but the pattern codes themselves. Added
noise alone does not solve it either: the tie statistic chooses too little for the full-band Fourier
detector (sigma 0.3 against the 0.5 of the sweep), and more noise is a fingerprint of its own
(sigma 1.0 gives Fourier 58 % and Haralick 84 %).

The chain of PLUS (notch, then leak-free spectrum matching) on the four GAN sets of the course,
360 images each, APCER in percent untouched / cleaned:

| source | Fourier | LBP | Haralick |
|---|---|---|---|
| DRIT | 100.0 / 37.4 | 98.9 / 0.7 | 98.1 / 0.3 |
| CycleGAN | 70.7 / 52.4 | 0.4 / 0.7 | 0.0 / 0.0 |
| DistanceGAN | 100.0 / 21.5 | 96.5 / 1.0 | 0.0 / 0.0 |
| StarGAN v2 | 100.0 / 18.9 | 99.2 / 0.7 | 0.0 / 0.0 |

LBP and Haralick are closed for every GAN without per-GAN tuning; the Fourier detector keeps 19 to
52 % with a fold spread of about 20 points, and the CycleGAN set, the least detectable one
untouched, ends up worst.

Ablation on PLUS (`results_pad_train_v210_PLUS_development.csv`, twins drawn from all real attack
images, so not leak-free), Fourier / LBP APCER: notch alone 98.0 / 95.1, notch + ring-energy
matching without sectors 84.7 / 0.7, ring times sector matching at full strength 6.4 / 0.7 (rmin 0),
7.8 / 0.9 (rmin 0.03) and 11.1 / 1.0 (rmin 0.05, the operating point, chosen for image quality),
partial strengths 0.5 and 0.75 leave 94.3 / 82.5 and 34.2 / 21.4. Haralick is 0 in every row. On
IDIAP the notch alone is harmful: Fourier 79.6 to 100, LBP 74.6 to 95.8, Haralick 51.3 to 98.3, the
pressed-down bins are a fingerprint of their own (`results_pad_train_v210_IDIAP_notch.csv`).

Generator side: a CycleGAN retrained on PLUS with the fixed smooth convolution layer of
Kinoshita and Kiya (Osakabe et al., IWAIT 2021) loses the checkerboard peaks but keeps the
high-frequency deficit. The Fourier detector still misses 99 % of real attacks, the cleaning chain
brings it to 8 to 10 %. At random initialisation the period-2 checkerboard energy on the Nyquist
lines drops from 49.5 to 0.25 times the spectrum mean.

Detector probe (`results_probe_PLUS_v210.csv`): a radial-spectrum SVM trained to separate real
attacks from untouched AE_Res samples (accuracy 0.999) calls 12 % of the cleaned fold-0 samples
and 0 % of the cleaned fixed-layer CycleGAN samples synthetic, with the ranking largely intact
(AUC 0.985 and 0.894). The cleaning moves the samples across the threshold of a detector trained
on untouched fakes; the harness's synthetic protocol retrains the detector on the cleaned set.

Earlier tables from pad_train 1.0.2 (lower-thirds Fourier bands on IDIAP and SCUT, unscaled
Haralick, scikit-learn 1.7.2) are kept in `results/` and must not be mixed with the 2.1.0 tables,
because the fold partition depends on the scikit-learn version, see
[results/README.md](results/README.md).

## Layout

| folder | content |
|---|---|
| `tools/` | removal operators (`apply_removal.py`), quality metrics, sweep scripts, runners for the harness, the batch driver `run_all_experiments.ps1` |
| `retrain/` | CycleGAN data preparation and export, the fixed-smooth-convolution networks, the patch for the official CycleGAN code |
| `results/` | every result table as CSV, with an index |

## Data

Not included. The bona fide and attack samples of PLUS, IDIAP and SCUT, the diffusion sets
(`AE_Res-diff`, `VAE-diff`) and the GAN sets (`Data_prepared`) are provided for the course.
`pad-train download` fetches them into a data root, the scripts here take that root as
`--data-root`.

## Setup

The tools run on Python 3.12 with numpy, scipy, scikit-image, scikit-learn, pillow, pandas and
matplotlib. The harness needs Python 3.12 as well:

```
pip install --upgrade pad-train
```

## Reproducing the results

Notch, then the leak-free chain on PLUS (the runner draws the twins per fold from the training
subjects, cleans the notched set per fold and evaluates each fold with the harness):

```
python tools/apply_removal.py --in <root>/AE_Res-diff/PLUS_matched/spoof/samples --out <root>/AE_Res-notch/PLUS_matched/spoof/samples --op peaks:t=1.0,size=7,rmin=0.1
python tools/run_pad_train_leakfree.py --data-root <root> --dataset PLUS --notched AE_Res-notch --match "strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05" --out-prefix AE_Res-leakfree --out results/results_pad_train_v210_leakfree_PLUS.csv
```

Noise-level matching on IDIAP and SCUT, and the fixed-sigma sweep on SCUT:

```
python tools/run_pad_train_leakfree.py --data-root <root> --dataset IDIAP --notched AE_Res-diff --op-name noisematch --match "strength=1.0,mode=1,stat=1" --out-prefix AE_Res-nmties --out results/results_pad_train_v210_leakfree_IDIAP.csv
python tools/run_pad_train_leakfree.py --data-root <root> --dataset SCUT --notched AE_Res-diff --op-name noisematch --match "strength=1.0,mode=1,stat=1" --out-prefix AE_Res-nmties --out results/results_pad_train_v210_leakfree_SCUT.csv
python tools/apply_removal.py --in <root>/AE_Res-diff/SCUT/spoof/samples --out <root>/AE_Res-noise0.5/SCUT/spoof/samples --op noise:sigma=0.5
python tools/run_pad_train_cleaned.py --data-root <root> --source AE_Res-noise0.5 --datasets SCUT --out results/results_pad_train_v210_recheck_SCUT.csv
```

Spectrum matching only on IDIAP and SCUT, spectrum filling on SCUT:

```
python tools/run_pad_train_leakfree.py --data-root <root> --dataset IDIAP --notched AE_Res-diff --op-name specmatch --match "strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05" --out-prefix AE_Res-match --out results/results_pad_train_v210_leakfree_IDIAP_match.csv
python tools/run_pad_train_leakfree.py --data-root <root> --dataset SCUT --notched AE_Res-diff --op-name specmatch --match "strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05" --out-prefix AE_Res-match --out results/results_pad_train_v210_leakfree_SCUT_match.csv
python tools/run_pad_train_leakfree.py --data-root <root> --dataset SCUT --notched AE_Res-diff --op-name specfill --match "strength=1.0,mode=1,sectors=8,rmin=0.45" --out-prefix AE_Res-fill --out results/results_pad_train_v210_leakfree_SCUT_fill.csv
```

Each leak-free run cleans the five folds one after the other; the runner extracts the features of
the real images once and reuses them for every fold, a SCUT run takes about 70 minutes on six
cores. `tools/run_all_experiments.ps1` runs the whole batch of 6 November (quality metrics, the
detector probe, the three SCUT variants, the four GAN sets, the PLUS ablation, the dither probe)
in order with checkpoints: it reads the data root from `PAD_DATA_ROOT`, expects `python` to be
the Python 3.12 with pad-train, and writes tables and logs below `results/`.

## Retraining the generator

`retrain/cyclegan_fsc/setup_cyclegan.ps1` clones the official CycleGAN code at the tested commit,
applies the patch (two new network names, `--netG resnet_9blocks_fsc` and `--netD basic_fsc`: a
hold kernel after every transposed convolution and before every stride-2 convolution) and fetches
the fixed-smooth-convolution layer from its original repository, which has no licence file and is
therefore not redistributed here. Then, from `retrain/`:

```
python prepare_cyclegan_data.py --data-root <root> --dataset PLUS --out datasets/plus_fv --limit-test 2048
cd pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ../datasets/plus_fv --name plus_fsc --model cycle_gan --netG resnet_9blocks_fsc --netD basic_fsc --input_nc 1 --output_nc 1 --preprocess crop --crop_size 192 --no_flip --batch_size 2 --n_epochs 30 --n_epochs_decay 30 --save_epoch_freq 5
python test.py --dataroot ../datasets/plus_fv --name plus_fsc --model test --model_suffix _A --netG resnet_9blocks_fsc --input_nc 1 --output_nc 1 --preprocess none --no_dropout --num_test 2048 --no_flip
python ../export_synthetic.py --results results/plus_fsc/test_latest/images --data-root <root> --name CycleGAN-fsc --dataset PLUS
```

The stock run is the same with `--netG resnet_9blocks --netD basic`. The exported folder is
evaluated like a cleaned set with `tools/run_pad_train_cleaned.py --source CycleGAN-fsc`.
`FSC_ORDER=1` selects the 3x3 triangle kernel instead of the 2x2 zero-order hold. Image sides must
be multiples of 4.

## References

- Vorderleitner, Uhl: synthetic finger-vein PAI data and the SAC 2025 PAD study.
- Osakabe, Tanaka, Kinoshita, Kiya: CycleGAN without checkerboard artifacts (IWAIT 2021), based
  on Kinoshita and Kiya, fixed smooth convolutional layer (ICASSP 2020).
- Wesselkamp et al.: frequency-domain attacks on GAN fingerprints (2022).
