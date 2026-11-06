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
| IDIAP baseline | | 0.0 | 2.9 | 1.3 |
| SCUT untouched | | 94.1 | 100.0 | 59.6 |
| SCUT cleaned | noise-level matching, sigma 0.3 on average | 23.0 | 99.8 | 30.4 |
| SCUT cleaned | white noise, sigma 0.5 | 11.8 | 100.0 | 30.1 |
| SCUT baseline | | 4.6 | 1.4 | 0.1 |

PLUS and IDIAP are closed on all three extractors. SCUT is the open case: LBP does not move, the
Haralick detector keeps a floor near 30 %, and the tie statistic chooses too little noise for the
full-band Fourier detector (sigma 0.3 against the 0.5 of the sweep). More noise is a fingerprint of
its own: sigma 1.0 gives Fourier 58 % and Haralick 84 % again, so the level must be matched, not
maximised. The sigma 0.5 row was selected on the evaluation itself and is reported as a sweep, not
as a recipe.

Generator side: a CycleGAN retrained on PLUS with the fixed smooth convolution layer of
Kinoshita and Kiya (Osakabe et al., IWAIT 2021) loses the checkerboard peaks but keeps the
high-frequency deficit. The Fourier detector still misses 99 % of real attacks, the cleaning chain
brings it to 8 to 10 %. At random initialisation the period-2 checkerboard energy on the Nyquist
lines drops from 49.5 to 0.25 times the spectrum mean.

Earlier tables from pad_train 1.0.2 (lower-thirds Fourier bands on IDIAP and SCUT, unscaled
Haralick, scikit-learn 1.7.2) are kept in `results/` and must not be mixed with the 2.1.0 tables,
because the fold partition depends on the scikit-learn version, see
[results/README.md](results/README.md).

## Layout

| folder | content |
|---|---|
| `tools/` | removal operators (`apply_removal.py`), quality metrics, sweep scripts, runners for the harness |
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

Each leak-free run cleans the five folds one after the other and takes hours on a CPU.

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
