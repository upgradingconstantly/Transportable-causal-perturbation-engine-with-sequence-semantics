param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  & $PythonExe -m venv .venv
}

$VenvPython = Join-Path ".venv" "Scripts\\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements-dev.txt
& $VenvPython -m pip install -e .

Write-Host "Bootstrap complete. Activate with .\\.venv\\Scripts\\Activate.ps1"
