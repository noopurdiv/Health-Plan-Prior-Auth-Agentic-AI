# Start ClearAuth API server
# Excludes scripts/ and *.docx from --reload to avoid hang during report generation

Set-Location $PSScriptRoot

$port = 8000
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $port is in use (PID $($existing.OwningProcess)). Stopping..."
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "Starting ClearAuth at http://127.0.0.1:$port"
# Use --reload-dir (not --reload-exclude globs) — PowerShell expands * before uvicorn sees it
uvicorn src.api.main:app --host 127.0.0.1 --port $port --reload --reload-dir src --reload-dir frontend
