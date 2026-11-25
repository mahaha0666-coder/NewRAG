param([switch]$Check)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$project = $PSScriptRoot
$experiment = Join-Path $project 'data\four_benchmark_eval'
$pythonRoot = Join-Path $project 'python'
$python = Join-Path $pythonRoot 'python.exe'
$runtime = Join-Path $experiment 'runtime\llama-vulkan'
$serverPath = Join-Path $runtime 'llama-server.exe'
$models = Join-Path $experiment 'models\Qwen3-VL-2B-Instruct-GGUF'
$weights = Join-Path $models 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'
$projector = Join-Path $models 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
$logs = Join-Path $experiment 'runtime'
$port = 18082
$endpoint = "http://127.0.0.1:$port"
$env:PATH = "$pythonRoot;$env:PATH"
$env:PYTHONPATH = $project
$env:PYTHONNOUSERSITE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:NO_PROXY = 'localhost,127.0.0.1'

$devices = & $serverPath --list-devices
if ($LASTEXITCODE -ne 0) { throw 'The bundled inference program could not list GPU devices.' }
$nvidia = $devices | Select-String '^\s*(Vulkan\d+):\s+NVIDIA' | Select-Object -First 1
if (!$nvidia) { throw 'No NVIDIA Vulkan device was found. Check the NVIDIA driver.' }
$device = $nvidia.Matches[0].Groups[1].Value
Write-Host $nvidia.Line.Trim()

$runLock = $null
$server = $null
try {
    $runLock = [System.IO.File]::Open((Join-Path $logs 'desktop-evaluation.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $serverArgs = @(
        '-m', ('"' + $weights + '"'), '--mmproj', ('"' + $projector + '"'),
        '--alias', 'qwen3-vl-2b-q4', '--host', '127.0.0.1', '--port', $port,
        '--device', $device, '--n-gpu-layers', 99, '--ctx-size', 16384,
        '--parallel', 4, '--cont-batching'
    )
    $server = Start-Process -FilePath $serverPath -WorkingDirectory $runtime -ArgumentList $serverArgs `
        -WindowStyle Hidden -PassThru -RedirectStandardError (Join-Path $logs 'desktop-server.err') `
        -RedirectStandardOutput (Join-Path $logs 'desktop-server.out')
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($server.HasExited) { break }
        try { $ready = (Invoke-RestMethod "$endpoint/health" -TimeoutSec 2).status -eq 'ok' } catch {}
        if ($ready) { break }
        Start-Sleep -Seconds 1
    }
    if (!$ready -or $server.HasExited) { throw 'The local reader did not start. See desktop-server.err.' }
    $readerArgs = @('--reader-url', "$endpoint/v1", '--reader-model', 'qwen3-vl-2b-q4',
                    '--dataset', 'TriView2CAD', '--triview-shard', '2')
    if ($Check) { $readerArgs += '--check-only' }
    & $python -u 'scripts/evaluate_cad_text_full.py' @readerArgs |
        Tee-Object -FilePath (Join-Path $logs 'desktop-progress.log') -Append
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE." }
} finally {
    if ($server -and !$server.HasExited) { Stop-Process -InputObject $server -ErrorAction SilentlyContinue }
    if ($runLock) { $runLock.Dispose() }
}
