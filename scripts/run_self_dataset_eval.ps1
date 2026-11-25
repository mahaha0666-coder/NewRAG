param(
    [string]$DatasetPath,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
    throw 'DatasetPath is required.'
}
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $repositoryRoot 'data\self_dataset_eval'
}

$python = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}

Write-Host 'Enter a NEW or rotated DeepSeek API key. Input is hidden and is not saved.'
$secureKey = Read-Host -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
$plainKey = $null
$exitCode = 1
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw 'No DeepSeek API key was entered.'
    }
    $env:DEEPSEEK_API_KEY = $plainKey
    & $python (Join-Path $PSScriptRoot 'evaluate_drawing_pdfs.py') `
        $DatasetPath $OutputPath --dpi 240
    $exitCode = $LASTEXITCODE
}
finally {
    Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
    $plainKey = $null
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

exit $exitCode
