$ErrorActionPreference = 'Stop'

$key = $env:DEEPSEEK_API_KEY
if ([string]::IsNullOrWhiteSpace($key)) {
    throw 'DEEPSEEK_API_KEY is not set in this PowerShell session.'
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$secretDirectory = Join-Path $repositoryRoot 'data\secrets'
$secretPath = Join-Path $secretDirectory 'deepseek-api-key.dpapi'
New-Item -ItemType Directory -Path $secretDirectory -Force | Out-Null

$secureKey = ConvertTo-SecureString $key -AsPlainText -Force
$encryptedKey = ConvertFrom-SecureString $secureKey
Set-Content -LiteralPath $secretPath -Value $encryptedKey -Encoding utf8 -NoNewline

Write-Output "Encrypted DeepSeek credential saved to $secretPath"
