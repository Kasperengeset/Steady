$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskFile = Join-Path $taskRepo '.runs\processes.json'
if (-not (Test-Path -LiteralPath $taskFile)) { Write-Output 'No recorded local app processes.'; exit 0 }
$taskState = Get-Content -LiteralPath $taskFile -Raw | ConvertFrom-Json
if ($taskState.root -ne $taskRepo) { throw 'Process state belongs to another checkout.' }
foreach ($taskService in $taskState.services) {
    $taskProcess = Get-Process -Id $taskService.process_id -ErrorAction SilentlyContinue
    if ($taskProcess -and $taskProcess.StartTime.ToUniversalTime().ToString('O') -eq $taskService.started) {
        Stop-Process -Id $taskProcess.Id
        Write-Output ("Stopped " + $taskService.name)
    }
}
