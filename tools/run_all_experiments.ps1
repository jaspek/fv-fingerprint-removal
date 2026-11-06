# Batch driver of the 6 November 2026 experiments (quality, detector probe, SCUT variants, the four GANs, the PLUS ablation, the dither probe, IDIAP matching only, the two-stage chains).
# Runs every prepared experiment in order, with checkpoints:
#   - a step whose result files already exist is skipped, so the same command can be re-run after an interruption,
#   - every step's console output is saved to results\logs\<step>.log, start, end, exit code and duration go to results\logs\master.log,
#   - the leak-free runs resume per fold (--resume), the cleaning steps leave a .done marker next to their folder,
#   - a failing step is logged and the next step starts.
#
#   cd tools        (PAD_DATA_ROOT = the pad-train data root, python = Python 3.12 with pad-train 2.1.0)
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1              # everything, in order
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1 -List        # show the steps and which are done
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1 -Only E2a_SCUT_match,E6_GAN_drit
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1 -DryRun      # log the commands without running them
param([string[]]$Only = @(), [switch]$List, [switch]$DryRun)

$ErrorActionPreference = "Continue"
$Only = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ })     # "-File" passes a,b,c as one string
$D = ($env:PAD_DATA_ROOT -replace "\\", "/")
$PY = "python"                                  # not $P: variable names are case-insensitive, a loop variable $p would shadow it
$O = Join-Path $PSScriptRoot "..\results"      # result tables, logs/ and figures/
$T = $PSScriptRoot
$L = "$O\logs"
$REAL = "$D/Data_FV_Spoofing_WS2025_26"
New-Item -ItemType Directory -Force $L | Out-Null
New-Item -ItemType Directory -Force "$O\figures" | Out-Null
$master = "$L\master.log"
Set-Location $T

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $master -Value $line
}

# Runs one python command, output to the step log and the console, returns the exit code.
function Py($log, [string[]]$argv) {
    Log ("    python " + ($argv -join " "))
    if ($DryRun) { return 0 }
    # Add-Content instead of Tee-Object: Windows PowerShell's Tee-Object writes UTF-16, this writes UTF-8 (the output is small).
    # try/catch: a command that cannot start throws instead of setting $LASTEXITCODE (6 Nov: the dither step's loop variable $p
    # shadowed the interpreter path $P, the step was reported as done after 0 seconds), and the step must count as failed.
    $global:LASTEXITCODE = 0
    try {
        & $PY -u -X utf8 @argv 2>&1 | ForEach-Object { Add-Content -Path $log -Value "$_" -Encoding UTF8; $_ } | Out-Host
    } catch {
        Log ("    ERROR " + $_.Exception.Message)
        return 1
    }
    return $LASTEXITCODE
}

# Cleaning step with a .done marker next to the output folder (apply_removal writes image by image).
function Clean($log, $inDir, $outDir, $op, [string[]]$extra = @()) {
    $marker = "$outDir.done"
    if (Test-Path $marker) { Log "    cleaning exists: $outDir"; return 0 }
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }          # partial folder from an interrupted run
    $code = Py $log (@("apply_removal.py", "--in", $inDir, "--out", $outDir, "--op", $op) + $extra)
    if ($code -eq 0 -and -not $DryRun) { New-Item -ItemType File -Force $marker | Out-Null }
    return $code
}

$script:steps = @()
function Step($name, [string[]]$done, [scriptblock]$run) {
    $script:steps += [pscustomobject]@{ Name = $name; Done = $done; Run = $run }
}

$LEAKFREE_SPEC = "strength=1.0,mode=1,energy=1,sectors=8,rmin=0.05"

# ------------------------------------------------------------------ E1: quality and panels of the recipes (minutes)
Step "E1_quality" @("$O\quality_PLUS_leakfree_v210_fold0.csv", "$O\quality_IDIAP_nmties_v210_fold0.csv", "$O\quality_SCUT_nmties_v210_fold0.csv",
                    "$O\quality_SCUT_noise0.5.csv", "$O\figures\visual_panel_IDIAP_nmties_v210_fold0.png", "$O\figures\visual_panel_SCUT_noise0.5.png") {
    param($log)
    $pairs = @(
        @("$D/AE_Res-diff/PLUS_matched/spoof/samples", "$D/AE_Res-leakfree-v210-fold0/PLUS_matched/spoof/samples", "$O\quality_PLUS_leakfree_v210_fold0.csv"),
        @("$D/AE_Res-diff/IDIAP/spoof/samples", "$D/AE_Res-nmties-v210-fold0/IDIAP/spoof/samples", "$O\quality_IDIAP_nmties_v210_fold0.csv"),
        @("$D/AE_Res-diff/SCUT/spoof/samples", "$D/AE_Res-nmties-v210-fold0/SCUT/spoof/samples", "$O\quality_SCUT_nmties_v210_fold0.csv"),
        @("$D/AE_Res-diff/SCUT/spoof/samples", "$D/AE_Res-noise0.5/SCUT/spoof/samples", "$O\quality_SCUT_noise0.5.csv"))
    foreach ($q in $pairs) {
        if (Test-Path $q[2]) { continue }
        $c = Py $log @("quality_metrics.py", "--a", $q[0], "--b", $q[1], "--out", $q[2]); if ($c -ne 0) { return $c }
    }
    $c = Py $log @("visual_panel.py", "--a", "$D/AE_Res-diff/IDIAP/spoof/samples", "--b", "$D/AE_Res-nmties-v210-fold0/IDIAP/spoof/samples", "--n", "3", "--out", "$O\figures\visual_panel_IDIAP_nmties_v210_fold0.png"); if ($c -ne 0) { return $c }
    return (Py $log @("visual_panel.py", "--a", "$D/AE_Res-diff/SCUT/spoof/samples", "--b", "$D/AE_Res-noise0.5/SCUT/spoof/samples", "--n", "3", "--out", "$O\figures\visual_panel_SCUT_noise0.5.png"))
}

# ------------------------------------------------------------------ E3: detector probe on PLUS (under an hour)
Step "E3_probe_PLUS" @("$O\results_probe_PLUS_v210.csv") {
    param($log)
    return (Py $log @("detector_probe.py", "--real", "$REAL/PLUS/spoof", "--fake", "$D/AE_Res-diff/PLUS_matched/spoof/samples",
        "--eval", "cleaned=$D/AE_Res-leakfree-v210-fold0/PLUS_matched/spoof/samples",
        "--eval", "cyclegan_fsc=$D/CycleGAN-fsc/PLUS_matched/spoof/samples",
        "--eval", "cyclegan_fsc_clean=$D/CycleGAN-fsc-clean-r002/PLUS_matched/spoof/samples",
        "--tile", "192", "--limit", "400", "--out", "$O\results_probe_PLUS_v210.csv"))
}

# ------------------------------------------------------------------ E2: SCUT spectrum under 2.1.0, leak-free (hours each)
Step "E2a_SCUT_match" @("$O\results_pad_train_v210_leakfree_SCUT_match.csv") {
    param($log)
    return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", "SCUT", "--notched", "AE_Res-diff", "--op-name", "specmatch",
        "--match", $LEAKFREE_SPEC, "--out-prefix", "AE_Res-match-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_SCUT_match.csv"))
}

# ------------------------------------------------------------------ E6: the four course GANs on PLUS, the chain, leak-free (one or two nights)
$GANS = [ordered]@{ drit = "003"; cyclegan = "003"; distancegan = "001"; "stargan-v2" = "001" }
foreach ($g in $GANS.Keys) {
    $v = $GANS[$g]
    Step "E6_GAN_$g" @("$O\results_pad_train_v210_leakfree_PLUS_$g.csv") {
        param($log)
        $c = Clean $log "$D/Data_prepared/PLUS/spoofed_synthethic_$g/$v/all_rs/reference" "$D/GAN-$g-notch/PLUS_matched/spoof/samples" "peaks:t=1.0,size=7,rmin=0.1"
        if ($c -ne 0) { return $c }
        return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", "PLUS", "--notched", "GAN-$g-notch",
            "--match", $LEAKFREE_SPEC, "--out-prefix", "GAN-$g-leakfree-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_PLUS_$g.csv"))
    }.GetNewClosure()
}

Step "E2b_SCUT_fill" @("$O\results_pad_train_v210_leakfree_SCUT_fill.csv") {
    param($log)
    return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", "SCUT", "--notched", "AE_Res-diff", "--op-name", "specfill",
        "--match", "strength=1.0,mode=1,sectors=8,rmin=0.45", "--out-prefix", "AE_Res-fill-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_SCUT_fill.csv"))
}

Step "E2c_SCUT_chain" @("$O\results_pad_train_v210_leakfree_SCUT_chain.csv") {
    param($log)
    return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", "SCUT", "--notched", "AE_Res-notch", "--op-name", "specmatch",
        "--match", $LEAKFREE_SPEC, "--out-prefix", "AE_Res-chain-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_SCUT_chain.csv"))
}

# ------------------------------------------------------------------ E4: the PLUS development rows and the IDIAP notch row under 2.1.0 (one night), one call per source
$DEV = @("AE_Res-notch", "AE_Res-clean", "AE_Res-clean2", "AE_Res-clean2-s0.5", "AE_Res-clean2-s0.75", "AE_Res-clean3-r0.03", "CycleGAN-stock", "CycleGAN-fsc")
foreach ($src in $DEV) {
    Step "E4_dev_$src" @("$O\results_pad_train_v210_PLUS_development_$src.csv") {
        param($log)
        return (Py $log @("run_pad_train_cleaned.py", "--data-root", $D, "--source", $src, "--datasets", "PLUS", "--out", "$O\results_pad_train_v210_PLUS_development_$src.csv"))
    }.GetNewClosure()
}
Step "E4_IDIAP_notch" @("$O\results_pad_train_v210_IDIAP_notch.csv") {
    param($log)
    return (Py $log @("run_pad_train_cleaned.py", "--data-root", $D, "--source", "AE_Res-notch", "--datasets", "IDIAP", "--out", "$O\results_pad_train_v210_IDIAP_notch.csv"))
}
Step "E4_merge" @("$O\results_pad_train_v210_PLUS_development.csv") {
    param($log)
    return (Py $log @("merge_csv.py", "--pattern", "$O\results_pad_train_v210_PLUS_development_*.csv", "--out", "$O\results_pad_train_v210_PLUS_development.csv"))
}

# ------------------------------------------------------------------ E5: SCUT LBP dither probe, LBP only (about an hour)
Step "E5_SCUT_dither" @("$O\results_pad_train_v210_SCUT_dither.csv") {
    param($log)
    foreach ($share in @("0.4", "1.0")) {
        $c = Clean $log "$D/AE_Res-diff/SCUT/spoof/samples" "$D/AE_Res-dither-p$share/SCUT/spoof/samples" "dither:p=$share,amp=1"
        if ($c -ne 0) { return $c }
    }
    return (Py $log @("run_pad_train_cleaned.py", "--data-root", $D, "--source", "AE_Res-dither-p0.4", "--source", "AE_Res-dither-p1.0", "--datasets", "SCUT",
        "--extractors", "lbp", "--out", "$O\results_pad_train_v210_SCUT_dither.csv"))
}

# ------------------------------------------------------------------ E7: IDIAP spectrum matching only, leak-free (about 40 min with the cached runner)
Step "E7_IDIAP_match" @("$O\results_pad_train_v210_leakfree_IDIAP_match.csv") {
    param($log)
    return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", "IDIAP", "--notched", "AE_Res-diff", "--op-name", "specmatch",
        "--match", $LEAKFREE_SPEC, "--out-prefix", "AE_Res-match-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_IDIAP_match.csv"))
}

# ------------------------------------------------------------------ E8: two-stage leak-free combinations on SCUT and IDIAP (about 1 h 10 and 45 min per step)
#   a: matching, then tie-matched noise (reads the AE_Res-match-v210 fold folders); b: tie-matched noise, then matching (reads AE_Res-nmties-v210)
foreach ($ds in @("SCUT", "IDIAP")) {
    Step "E8a_${ds}_match_noise" @("$O\results_pad_train_v210_leakfree_${ds}_match_noise.csv") {
        param($log)
        return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", $ds, "--notched", "AE_Res-match-v210", "--per-fold-input", "--op-name", "noisematch",
            "--match", "strength=1.0,mode=1,stat=1", "--out-prefix", "AE_Res-match-nm-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_${ds}_match_noise.csv"))
    }.GetNewClosure()
    Step "E8b_${ds}_noise_match" @("$O\results_pad_train_v210_leakfree_${ds}_noise_match.csv") {
        param($log)
        return (Py $log @("run_pad_train_leakfree.py", "--data-root", $D, "--dataset", $ds, "--notched", "AE_Res-nmties-v210", "--per-fold-input", "--op-name", "specmatch",
            "--match", $LEAKFREE_SPEC, "--out-prefix", "AE_Res-nm-match-v210", "--resume", "--out", "$O\results_pad_train_v210_leakfree_${ds}_noise_match.csv"))
    }.GetNewClosure()
}

# ------------------------------------------------------------------ driver
function IsDone($s) { return (@($s.Done | Where-Object { -not (Test-Path $_) }).Count -eq 0) }

if ($List) {
    foreach ($s in $script:steps) { "{0,-6} {1}" -f $(if (IsDone $s) { "done" } else { "open" }), $s.Name }
    exit 0
}

Log "=== run_all_experiments start (Only: $($Only -join ','); DryRun: $DryRun)"
if (-not $DryRun) {
    $envinfo = & $PY -c "import sys, pad_train, sklearn, numpy; print('python', sys.version.split()[0], 'pad-train', pad_train.__version__, 'scikit-learn', sklearn.__version__, 'numpy', numpy.__version__)" 2>&1
    Log "    $envinfo"
}
$failed = @()
foreach ($s in $script:steps) {
    if ($Only.Count -gt 0 -and ($Only -notcontains $s.Name)) { continue }
    if (IsDone $s) { Log "SKIP  $($s.Name): result exists"; continue }
    $log = "$L\$($s.Name).log"
    Log "START $($s.Name) -> $log"
    $t0 = Get-Date
    $code = & $s.Run $log
    $dt = (Get-Date) - $t0
    if ($code -eq 0) { Log ("DONE  {0} in {1}" -f $s.Name, $dt.ToString("hh\:mm\:ss")) }
    else { $failed += $s.Name; Log ("FAIL  {0} exit code {1} after {2}, continuing with the next step" -f $s.Name, $code, $dt.ToString("hh\:mm\:ss")) }
}
Log ("=== run_all_experiments end, failed: " + $(if ($failed.Count) { $failed -join ", " } else { "none" }))
