param(
    [string]$Matrix = "configs/experiment_matrix.example.json",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    throw "Python 3.11 project environment not found: $ProjectPython"
}

& $ProjectPython (Join-Path $PSScriptRoot "run_experiments.py") --matrix $Matrix @RemainingArgs
