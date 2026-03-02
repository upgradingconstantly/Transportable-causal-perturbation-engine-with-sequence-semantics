$ErrorActionPreference = "Stop"

$VenvPython = Join-Path ".venv" "Scripts\\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtual environment not found. Run scripts/bootstrap.ps1 first."
}

& $VenvPython -m ruff check .
& $VenvPython -m mypy src
& $VenvPython -m pytest
