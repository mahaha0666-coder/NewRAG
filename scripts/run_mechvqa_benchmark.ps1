param(
    [Parameter(Mandatory = $true)]
    [string]$BenchmarkRoot,
    [string]$OutputPath,
    [int]$Limit = 0,
    [int]$TargetWorkers = 4,
    [int]$JudgeWorkers = 8,
    [switch]$RetryErrors
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repositoryRoot 'data\mechvqa_eval'
}
$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$inputFile = Join-Path $BenchmarkRoot 'benchmark_data\vqa_benchmark\mechvqa_benchmark.jsonl'
$imageRoot = Join-Path $BenchmarkRoot 'benchmark_data'
$officialEvaluation = Join-Path $BenchmarkRoot 'evaluation'

foreach ($required in @($python, $inputFile, $officialEvaluation)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path not found: $required"
    }
}

Write-Host 'Enter the DeepSeek API key. Input is hidden, used only by this process, and is not saved.'
$secureKey = Read-Host -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$plainKey = $null
$oldPythonPath = $env:PYTHONPATH
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw 'No DeepSeek API key was entered.'
    }
    $env:DEEPSEEK_API_KEY = $plainKey
    $arguments = @(
        '-m', 'mechrag.mechvqa_benchmark',
        '--input-file', $inputFile,
        '--image-root', $imageRoot,
        '--output-dir', $OutputPath,
        '--workers', $TargetWorkers,
        '--judge-workers', $JudgeWorkers
    )
    if ($Limit -gt 0) {
        $arguments += @('--limit', $Limit)
    }
    if ($RetryErrors) {
        $arguments += '--retry-errors'
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MechRAG target inference failed with exit code $LASTEXITCODE"
    }

    $config = Join-Path $OutputPath 'official_eval_config.json'
    $env:PYTHONPATH = $officialEvaluation
    Push-Location $officialEvaluation
    try {
        & $python -m mechvqa_eval.evaluate_vqa --config $config --phase phase2
        if ($LASTEXITCODE -ne 0) {
            throw "MechVQA official judging failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }

    $env:PYTHONPATH = $repositoryRoot
    & $python -m mechrag.mechvqa_report `
        --evaluated-file (Join-Path $OutputPath 'evaluated_mechrag_baseline.jsonl') `
        --phase1-file (Join-Path $OutputPath 'responses_mechrag_baseline.jsonl') `
        --output-dir $OutputPath
    if ($LASTEXITCODE -ne 0) {
        throw "MechVQA report generation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($null -eq $oldPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $oldPythonPath
    }
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}
