param(
    [string]$Python = '',
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$NoWarmup
)

$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $Python) { $Python = Join-Path $taskRepo 'backend\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Set -Python to your installed backend Python, or create backend/.venv first.' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRepo 'frontend\node_modules\vite\bin\vite.js'))) { throw 'Run npm.cmd ci in frontend first.' }
foreach ($taskPort in @($BackendPort, $FrontendPort)) {
    if ($taskPort -lt 1024 -or $taskPort -gt 65535) { throw 'Choose ports between 1024 and 65535.' }
    if (Get-NetTCPConnection -LocalPort $taskPort -State Listen -ErrorAction SilentlyContinue) { throw "Port $taskPort is in use. Choose another port; no existing process was stopped." }
}
$taskNode = (Get-Command node.exe -ErrorAction Stop).Source
$taskLogs = Join-Path $taskRepo '.runs'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskSavedEnv = @{}
foreach ($taskKey in @('DATABASE_URL', 'FRONTEND_URL', 'GOOGLE_REDIRECT_URI', 'CORS_ORIGINS', 'STEADY_API_TARGET', 'WARMUP_MODELS', 'PYTHONDONTWRITEBYTECODE')) {
    $taskSavedEnv[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
}
try {
    $env:DATABASE_URL = 'sqlite:///' + (Join-Path $taskRepo 'data\app.db').Replace('\', '/')
    $env:FRONTEND_URL = "http://localhost:$FrontendPort"
    $env:GOOGLE_REDIRECT_URI = "http://127.0.0.1:$BackendPort/api/google-health/callback"
    $env:CORS_ORIGINS = '["http://localhost:' + $FrontendPort + '","http://127.0.0.1:' + $FrontendPort + '"]'
    $env:STEADY_API_TARGET = "http://127.0.0.1:$BackendPort"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    if ($NoWarmup) { $env:WARMUP_MODELS = 'false' }
    $taskBackend = Start-Process -FilePath $Python -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', $BackendPort) -WorkingDirectory (Join-Path $taskRepo 'backend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs 'backend.log') -RedirectStandardError (Join-Path $taskLogs 'backend.err.log') -PassThru
    try {
        $taskFrontend = Start-Process -FilePath $taskNode -ArgumentList @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', $FrontendPort, '--strictPort') -WorkingDirectory (Join-Path $taskRepo 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs 'frontend.log') -RedirectStandardError (Join-Path $taskLogs 'frontend.err.log') -PassThru
    } catch {
        Stop-Process -Id $taskBackend.Id -ErrorAction SilentlyContinue
        throw
    }
    $taskServices = @(
        @{ name = 'backend'; process_id = $taskBackend.Id; started = $taskBackend.StartTime.ToUniversalTime().ToString('O') },
        @{ name = 'frontend'; process_id = $taskFrontend.Id; started = $taskFrontend.StartTime.ToUniversalTime().ToString('O') }
    )
    @{ root = $taskRepo; services = $taskServices; frontend = "http://localhost:$FrontendPort"; backend = "http://127.0.0.1:$BackendPort" } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskLogs 'processes.json') -Encoding UTF8
    Write-Output "Steady starting at http://localhost:$FrontendPort"
    Write-Output "API: http://127.0.0.1:$BackendPort/docs"
    Write-Output "Logs: $taskLogs"
    Write-Output 'Use scripts/stop_local.ps1 to stop these two processes.'
} finally {
    foreach ($taskKey in $taskSavedEnv.Keys) { [Environment]::SetEnvironmentVariable($taskKey, $taskSavedEnv[$taskKey], 'Process') }
}
