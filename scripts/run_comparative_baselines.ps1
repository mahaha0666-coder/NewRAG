param(
    [string]$BenchmarkRoot,
    [string]$OutputPath,
    [ValidateSet('both', 'mechvqa', 'self')]
    [string]$Dataset = 'both',
    [int]$Limit = 0,
    [int]$Workers = 8,
    [int]$JudgeWorkers = 12,
    [switch]$SkipJudge
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BenchmarkRoot)) {
    $workspaceRoot = Split-Path -Parent $repositoryRoot
    $BenchmarkRoot = Join-Path $workspaceRoot 'MechVQA'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repositoryRoot 'data\comparative_baselines'
}
$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$inputFile = Join-Path $BenchmarkRoot 'benchmark_data\vqa_benchmark\mechvqa_benchmark.jsonl'
$imageRoot = Join-Path $BenchmarkRoot 'benchmark_data'
$officialEvaluation = Join-Path $BenchmarkRoot 'evaluation'
$selfManifest = Join-Path $repositoryRoot 'data\self_dataset_eval\manifest.json'
$groundTruth = Join-Path $repositoryRoot 'data\self_dataset_eval\ground_truth_objective_fields.json'

foreach ($required in @($python, $inputFile, $officialEvaluation, $selfManifest, $groundTruth)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path not found: $required"
    }
}

Write-Host 'Enter the DeepSeek credential. Input is hidden, process-only, and is not saved.'
$secureKey = Read-Host -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$plainKey = $null
$oldPythonPath = $env:PYTHONPATH
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw 'No DeepSeek credential was entered.'
    }
    $env:DEEPSEEK_API_KEY = $plainKey
    $env:PYTHONPATH = $repositoryRoot
    $arguments = @(
        '-m', 'mechrag.comparative_baselines',
        '--mechvqa-input', $inputFile,
        '--image-root', $imageRoot,
        '--self-manifest', $selfManifest,
        '--ground-truth', $groundTruth,
        '--output-dir', $OutputPath,
        '--dataset', $Dataset,
        '--workers', $Workers,
        '--judge-workers', $JudgeWorkers,
        '--limit', $Limit
    )
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Comparative target inference failed with exit code $LASTEXITCODE"
    }

    if (-not $SkipJudge -and $Dataset -ne 'self') {
        $config = Join-Path $OutputPath 'official_eval_config.json'
        $env:PYTHONPATH = $officialEvaluation
        Push-Location $officialEvaluation
        try {
            & $python -m mechvqa_eval.evaluate_vqa --config $config --phase phase2
            if ($LASTEXITCODE -ne 0) {
                throw "MechVQA judging failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }

    if (-not $SkipJudge -and $Dataset -eq 'both') {
        $env:PYTHONPATH = $repositoryRoot
        $summaryScript = Join-Path $repositoryRoot 'scripts\summarize_comparative_baselines.py'
        $evaluated = Join-Path $OutputPath 'evaluated_baselines.jsonl'
        $selfSummary = Join-Path $OutputPath 'self_dataset_baseline_summary.json'
        $oursSummary = Join-Path $repositoryRoot 'data\mechvqa_eval\summary.json'
        & $python $summaryScript `
            --evaluated $evaluated `
            --self-summary $selfSummary `
            --ours-summary $oursSummary `
            --output-dir $OutputPath
        if ($LASTEXITCODE -ne 0) {
            throw "Comparative result summarization failed with exit code $LASTEXITCODE"
        }
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
