param(
    [string]$ExperimentRoot = (Join-Path $PSScriptRoot '..\data\four_benchmark_eval'),
    [string]$ServerPath,
    [ValidateRange(1,65535)][int]$Port = 18081,
    [ValidateRange(1,16)][int]$Parallel = 4
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ExperimentRoot).Path
if (-not $ServerPath) {
    $runtime = Join-Path $root 'runtime\llama-vulkan'
    $ServerPath = Get-ChildItem -LiteralPath $runtime -Filter 'llama-server.exe' -File -Recurse |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ServerPath) {
    throw 'Set -ServerPath to a llama-server.exe build with multimodal support.'
}
$models = Join-Path $root 'models\Qwen3-VL-2B-Instruct-GGUF'
$weights = (Resolve-Path -LiteralPath (Join-Path $models 'Qwen3VL-2B-Instruct-Q4_K_M.gguf')).Path
$projector = (Resolve-Path -LiteralPath (Join-Path $models 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf')).Path
$serverArgs = @(
    '-m', $weights, '--mmproj', $projector, '--alias', 'qwen3-vl-2b-q4',
    '--host', '127.0.0.1', '--port', $Port, '--n-gpu-layers', 99,
    '--ctx-size', (4096 * $Parallel), '--parallel', $Parallel, '--cont-batching'
)
& $ServerPath @serverArgs
exit $LASTEXITCODE
