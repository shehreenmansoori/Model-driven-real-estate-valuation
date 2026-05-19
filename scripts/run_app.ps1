param(
    [switch]$SkipTraining,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host ".venv not found. Running setup first..."
    & "$PSScriptRoot\setup_windows.ps1"
}

. .\.venv\Scripts\Activate.ps1

python -c "import sys; v=sys.version_info[:2]; assert (3,11) <= v < (3,13), f'Unsupported Python {sys.version.split()[0]}. Use Python 3.11 or 3.12.'"

if (-not $SkipTraining) {
    if (
        -not (Test-Path "model\saved_model.pkl") -or
        -not (Test-Path "model\metrics.pkl") -or
        -not (Test-Path "model\features.pkl")
    ) {
        Write-Host "Model artifacts missing. Training model..."
        python model\train_model.py
    }
}

function Get-FirstFreePort {
    param([int]$StartPort, [int]$MaxSteps = 20)

    for ($offset = 0; $offset -le $MaxSteps; $offset++) {
        $candidate = $StartPort + $offset
        $inUse = netstat -ano | Select-String ":$candidate\s+.*LISTENING"
        if (-not $inUse) {
            return $candidate
        }
    }

    throw "No free port found in range $StartPort-$($StartPort + $MaxSteps)."
}

$resolvedPort = Get-FirstFreePort -StartPort $Port
if ($resolvedPort -ne $Port) {
    Write-Host "Port $Port is in use. Starting Streamlit on port $resolvedPort instead."
}

python -m streamlit run app\app.py --server.port $resolvedPort
