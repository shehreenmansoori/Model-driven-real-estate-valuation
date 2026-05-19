param(
    [switch]$RecreateVenv,
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

if ($RecreateVenv -and (Test-Path ".venv")) {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with Python $PythonVersion..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            py -$PythonVersion -m venv .venv
        } catch {
            throw "Python $PythonVersion was not found via 'py'. Install Python 3.11 or 3.12 and try again."
        }
    } else {
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            throw "Python is not installed or not on PATH. Install Python 3.11 or 3.12 and try again."
        }
        python -m venv .venv
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment creation failed. '.venv\\Scripts\\python.exe' was not created."
}

. .\.venv\Scripts\Activate.ps1

python -c "import sys; v=sys.version_info[:2]; assert (3,11) <= v < (3,13), f'Unsupported Python {sys.version.split()[0]}. Use Python 3.11 or 3.12.'"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add your OPENROUTER_API_KEY before using AI features."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run app with: .\scripts\run_app.ps1"
