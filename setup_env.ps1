$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install MuseTalk dependencies."
}

Write-Host "MuseTalk environment is ready. Activate it with .\.venv\Scripts\Activate.ps1"
