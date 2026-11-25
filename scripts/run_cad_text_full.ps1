param(
    [string[]]$Dataset = @('FCM', 'DeepCAD', 'TriView2CAD'),
    [int]$Port = 18081
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experiment = Join-Path $project 'data\four_benchmark_eval'
$logs = Join-Path $experiment 'runtime'
$python = Join-Path $project '.venv\Scripts\python.exe'
$env:NO_PROXY = 'localhost,127.0.0.1'
$env:PYTHONPATH = $project
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$endpoint = "http://127.0.0.1:$Port"
$server = $null

try {
    try {
        $ready = (Invoke-RestMethod "$endpoint/health" -TimeoutSec 2).status -eq 'ok'
    } catch {
        $ready = $false
    }
    if (!$ready) {
        $launcher = Join-Path $PSScriptRoot 'start_cad_local_reader.ps1'
        $startParams = @{
            FilePath = (Get-Command pwsh).Source
            WindowStyle = 'Hidden'
            PassThru = $true
            ArgumentList = @('-NoProfile', '-File', ('"'+$launcher+'"'), '-Port', $Port)
            RedirectStandardError = Join-Path $logs "text-full-server-$Port.log"
            RedirectStandardOutput = Join-Path $logs "text-full-server-$Port.out"
        }
        $server = Start-Process @startParams
        for ($attempt=0; $attempt -lt 30; $attempt++) {
            try {
                $ready = (Invoke-RestMethod "$endpoint/health" -TimeoutSec 2).status -eq 'ok'
                if ($ready) {break}
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    }
    if (!$ready) {throw 'Local visual reader did not start.'}
    Set-Location -LiteralPath $project
    $argsToPython = @('-u', 'scripts/evaluate_cad_text_full.py', '--reader-url', "$endpoint/v1",
                      '--reader-model', 'qwen3-vl-2b-q4', '--dataset') + $Dataset
    & $python @argsToPython
    if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
} finally {
    if ($server -and !$server.HasExited) {
        $server.Kill($true)
    }
}
