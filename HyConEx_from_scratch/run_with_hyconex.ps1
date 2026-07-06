# Lance un script Python dans l'environnement conda hyconex (CUDA).
# Usage : .\run_with_hyconex.ps1 train_hyconex_hyperlogic_dlbac.py --dataset amazon1

$Python = "C:\anaconda\envs\hyconex\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Environnement hyconex introuvable : $Python"
    exit 1
}
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot
& $Python @args
