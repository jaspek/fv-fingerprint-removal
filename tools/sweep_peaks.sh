#!/usr/bin/env bash
# Notch-threshold sweep: adaptive peak removal (peaks:t) BEFORE per-image radial spectrum
# matching to real PAI (specmatch mode 1).
# Subset rule (the same rule the 7 Oct smoke test used): PLUS users with the 8 smallest user ids
# (001..008: 240 bona fide, 144 real PAI), the 120 lexicographically first AE_Res PLUS_matched
# spoof samples, matching reference = the real PAI images of those 8 users.
# Note: the README anchors "t=1.0 too strong (20 dB), t>=1.5 no change" were measured with an
# earlier notch version without the rmin=0.1 lower limit, so the grid starts at 1.0 again and
# only rows of this table should be compared with each other.
#
# Usage (Git Bash, from anywhere):
#   bash tools/sweep_peaks.sh
# Options via environment variables:
#   T_VALUES="1.0 1.1 1.2 1.3 1.4 1.5 2.0"   notch thresholds (log(1+|F|) units, t=1 is about a factor e)
#   SIZE=7                       median window of the notch (try 5 or 9 as well)
#   STRENGTH=1.0                 matching strength (results go to a folder per strength)
#   ENERGY=0                     0 = match the ring mean log amplitude (tag _sm), 1 = match ring
#                                energies |F|^2 (tag _em); the plateau analysis recommends 1
#   NOISE=0                      sigma of white noise added between notch and matching (0 = none);
#                                adds _n<sigma> to the tag (plateau analysis: sigma about 1)
#   WORK=.../PS/smoke_subset     where the subset, cleaned sets and results go
#   ZIPS=.../Datasets            folder with Data_FV_Spoofing_WS2025_26.zip and AE_Res-diff.zip
# Output: $WORK/results_st<STRENGTH>/sweep_summary.csv plus a table on screen (one row per set).
# Caveat for reading the *_sm rows: the reference PAI images are the test images of the PAD
# folds, so the matching rows carry a leak; the ranking across t is still fair.
set -euo pipefail
export PYTHONIOENCODING=utf-8

ZIPS="${ZIPS:?set ZIPS to the folder with the dataset archives}"
WORK="${WORK:-./smoke_subset}"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T_VALUES="${T_VALUES:-1.0 1.1 1.2 1.3 1.4 1.5 2.0}"
SIZE="${SIZE:-7}"
STRENGTH="${STRENGTH:-1.0}"
ENERGY="${ENERGY:-0}"
NOISE="${NOISE:-0}"
case "$ENERGY" in 0|1) ;; *) echo "ENERGY must be 0 or 1 (got '$ENERGY')" >&2; exit 1;; esac
if awk "BEGIN{exit !($NOISE > 0)}"; then NTAG="_n${NOISE}"; else NTAG=""; NOISE=0; fi
MATCH_OP="specmatch:strength=$STRENGTH,mode=1,energy=$ENERGY"
if [ "$ENERGY" = "1" ]; then MTAG="em"; else MTAG="sm"; fi
R="$WORK/results_st${STRENGTH}"
echo "== settings: T_VALUES=\"$T_VALUES\" SIZE=$SIZE STRENGTH=$STRENGTH ENERGY=$ENERGY NOISE=$NOISE -> results in $R"
mkdir -p "$R"
cd "$TOOLS"

# ---- 1. subset from the zips (skipped when the 120 synthetic samples are already there) ----
if [ ! -d "$WORK/PLUS/spoof" ] || [ "$(ls "$WORK/AE_Res/PLUS_spoof" 2>/dev/null | wc -l)" -ne 120 ]; then
  echo "== extracting the 8-user PLUS subset and 120 AE_Res samples to $WORK"
  python - "$ZIPS" "$WORK" <<'PYEOF'
import sys, zipfile, os
zips, work = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(os.path.join(zips, "Data_FV_Spoofing_WS2025_26.zip"))
pre = "Data_FV_Spoofing_WS2025_26/PLUS/"
spoof = [n for n in z.namelist() if n.startswith(pre + "spoof/") and n.endswith(".png")]
users = sorted({os.path.basename(n).split("_")[2] for n in spoof})[:8]
sel = [n for n in spoof if os.path.basename(n).split("_")[2] in users]
sel += [n for n in z.namelist() if n.startswith(pre + "real/") and n.endswith(".png")
        and os.path.basename(n).split("_")[2] in users]
for n in sel:
    cls = "spoof" if "/spoof/" in n else "real"
    dst = os.path.join(work, "PLUS", cls, os.path.basename(n))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(z.read(n))
print("PLUS subset:", len(sel), "files, users", users)
z2 = zipfile.ZipFile(os.path.join(zips, "AE_Res-diff.zip"))
syn = [n for n in z2.namelist() if n.startswith("PLUS_matched/spoof/samples/") and n.endswith(".png")][:120]
for n in syn:
    dst = os.path.join(work, "AE_Res", "PLUS_spoof", os.path.basename(n))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(z2.read(n))
print("AE_Res subset:", len(syn), "files")
PYEOF
fi

SYN="$WORK/AE_Res/PLUS_spoof"
REF="$WORK/PLUS/spoof"
N_SYN="$(ls "$SYN" | wc -l)"
complete () {   # $1 = folder: true when it holds one output per synthetic image
  [ "$(ls "$1" 2>/dev/null | wc -l)" -eq "$N_SYN" ]
}

# ---- 2. folds by user --------------------------------------------------------------------
if [ ! -f "$R/folds_plus.csv" ]; then
  python make_folds.py --dataset plus --real "$WORK/PLUS/real" --spoof "$REF" --out "$R/folds_plus.csv"
fi

run_pad () {   # $1 = synthetic (or cleaned) folder, $2 = tag for the result files
  for f in fourier hist var entropy; do
    python pad_knn.py --folds "$R/folds_plus.csv" --feature "$f" --synthetic "$1" --out "$R/pad_${f}_$2.csv" | tail -1
  done
  python pad_knn.py --folds "$R/folds_plus.csv" --feature lbp --metric wasserstein --synthetic "$1" --out "$R/pad_lbp_$2.csv" | tail -1
}

# ---- 3. reference rows: untouched synthetic, matching only (skipped when already there) ---
if [ ! -f "$R/pad_lbp_untouched.csv" ]; then
  echo "== untouched synthetic"
  run_pad "$SYN" untouched
fi
if [ ! -f "$R/pad_lbp_${MTAG}.csv" ] || ! complete "$WORK/AE_Res/$MTAG"; then
  echo "== matching only ($MATCH_OP, ref = real PAI)"
  python apply_removal.py --in "$SYN" --out "$WORK/AE_Res/$MTAG" --op "$MATCH_OP" --ref "$REF"
  python quality_metrics.py --a "$SYN" --b "$WORK/AE_Res/$MTAG" --out "$R/q_${MTAG}.csv"
  run_pad "$WORK/AE_Res/$MTAG" "$MTAG"
fi

# ---- 4. the sweep: notch alone, then notch (+ noise) + matching ---------------------------
# (the notch output folder is reused; delete AE_Res/pk* and the pk*.csv files if op_peaks changes)
for t in $T_VALUES; do
  tag="pk${t}_s${SIZE}"
  echo "== peaks t=$t size=$SIZE"
  if [ ! -f "$R/pad_fourier_$tag.csv" ] || ! complete "$WORK/AE_Res/$tag"; then
    python apply_removal.py --in "$SYN" --out "$WORK/AE_Res/$tag" --op "peaks:t=$t,size=$SIZE,rmin=0.1"
    python quality_metrics.py --a "$SYN" --b "$WORK/AE_Res/$tag" --out "$R/q_$tag.csv"
    python pad_knn.py --folds "$R/folds_plus.csv" --feature fourier --synthetic "$WORK/AE_Res/$tag" --out "$R/pad_fourier_$tag.csv" | tail -1
  fi
  src="$WORK/AE_Res/$tag"
  if [ "$NOISE" != "0" ]; then
    echo "-- then noise sigma=$NOISE"
    python apply_removal.py --in "$src" --out "$WORK/AE_Res/${tag}${NTAG}" --op "noise:sigma=$NOISE"
    src="$WORK/AE_Res/${tag}${NTAG}"
  fi
  ctag="${tag}${NTAG}_${MTAG}"
  echo "-- then $MATCH_OP"
  python apply_removal.py --in "$src" --out "$WORK/AE_Res/$ctag" --op "$MATCH_OP" --ref "$REF"
  python quality_metrics.py --a "$SYN" --b "$WORK/AE_Res/$ctag" --out "$R/q_$ctag.csv"
  run_pad "$WORK/AE_Res/$ctag" "$ctag"
done

# ---- 5. one table ------------------------------------------------------------------------
python summarize_sweep.py --results "$R" --out "$R/sweep_summary.csv"
