$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements-app.txt"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $pythonArgs = @()

    if ($pythonCommand) {
        & $pythonCommand.Source --version
        if ($LASTEXITCODE -ne 0) {
            $pythonCommand = $null
        }
    }

    if (-not $pythonCommand) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            & $launcher.Source -3.11 --version
            if ($LASTEXITCODE -eq 0) {
                $pythonCommand = $launcher
                $pythonArgs = @("-3.11")
            }
        }
    }

    if (-not $pythonCommand) {
        Write-Host "Python 3.11 or newer was not found." -ForegroundColor Red
        Write-Host "Install Python from https://www.python.org/downloads/ and enable 'Add python.exe to PATH'." -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Creating local virtual environment..." -ForegroundColor Cyan
    & $pythonCommand.Source @pythonArgs -m venv (Join-Path $projectRoot ".venv")
}

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r $requirements

Write-Host "Starting Skin Lesion Analyzer at http://localhost:8501" -ForegroundColor Green
& $venvPython -m streamlit run (Join-Path $projectRoot "app.py")
