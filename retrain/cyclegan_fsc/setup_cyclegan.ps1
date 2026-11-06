# Sets up the official CycleGAN code with the fixed smooth convolution variant, next to this folder.
#
#   cd retrain
#   powershell -ExecutionPolicy Bypass -File cyclegan_fsc\setup_cyclegan.ps1
#
# Afterwards the retraining commands in the top-level README apply. The fixed-smooth-convolution layer itself is
# fetched from its original repository (popura/fixed-smooth-conv, no licence file, hence not
# redistributed here); the CycleGAN code is BSD licensed and only patched.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$retrain = Split-Path -Parent $here
Set-Location $retrain

if (-not (Test-Path "pytorch-CycleGAN-and-pix2pix")) {
    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
}
Set-Location "pytorch-CycleGAN-and-pix2pix"
git checkout -q 2a7afba2895d52556dd5dfe07e8555ef657ced6f
git apply --check (Join-Path $here "pytorch_cyclegan.patch")
git apply (Join-Path $here "pytorch_cyclegan.patch")
Copy-Item (Join-Path $here "fsc_networks.py") "models\fsc_networks.py" -Force
Invoke-WebRequest "https://raw.githubusercontent.com/popura/fixed-smooth-conv/f914a0ccc86946ca064f5376556bb3b88bb81caa/src/layer.py" -OutFile "models\fsc_layer.py"
pip install dominate
Write-Host "CycleGAN with --netG resnet_9blocks_fsc and --netD basic_fsc is ready in $(Get-Location)"
