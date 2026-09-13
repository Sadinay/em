$ErrorActionPreference = 'Stop'
$auditDir = $PSScriptRoot
$plan = Get-Content -LiteralPath (Join-Path $auditDir 'plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (Test-Path -LiteralPath (Join-Path $auditDir 'execution.json')) { throw 'Already executed; preserve original audit.' }
$workspacePath = [IO.Path]::GetFullPath($plan.workspace).TrimEnd('\')
$allowedPaths = @($plan.allowed_roots | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') + '\' })
if ((Get-FileHash -LiteralPath $plan.backup_path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.backup_sha256) { throw 'Result backup changed.' }
if ((Get-FileHash -LiteralPath (Join-Path $auditDir 'saved_torques.csv') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.saved_torques_sha256) { throw 'Saved torques changed.' }
if (Get-Process -Name femm,fkn,fkern,triangle -ErrorAction SilentlyContinue) { throw 'FEMM processes are active; do not clean live runs.' }
$verified = @()
foreach ($entry in $plan.delete) {
    $targetPath = [IO.Path]::GetFullPath((Join-Path $workspacePath $entry.path))
    $inside = $false
    foreach ($allowedPath in $allowedPaths) {
        if ($targetPath.StartsWith($allowedPath, [StringComparison]::OrdinalIgnoreCase)) { $inside = $true }
    }
    if (-not $inside -or [IO.Path]::GetExtension($targetPath) -ine '.ans') { throw "Unsafe deletion target: $targetPath" }
    $item = Get-Item -LiteralPath $targetPath -Force
    if ($item.PSIsContainer -or $item.Length -ne $entry.bytes) { throw "Changed target: $targetPath" }
    $ancestor = $item
    while ($ancestor.FullName -ne $workspacePath) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point in target path: $targetPath" }
        $ancestor = Get-Item -LiteralPath (Split-Path -Parent $ancestor.FullName) -Force
    }
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "ANS changed: $targetPath" }
    foreach ($pair in @(@($entry.result_path, $entry.result_sha256), @($entry.model_path, $entry.model_sha256))) {
        if ((Get-FileHash -LiteralPath (Join-Path $workspacePath $pair[0]) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pair[1]) { throw "Retained result/input changed: $($pair[0])" }
    }
    $verified += [PSCustomObject]@{ target = $targetPath; record = $entry }
}
# All concrete absolute targets and retained evidence have been checked before deletion.
$deleted = @()
foreach ($item in $verified) {
    Remove-Item -LiteralPath $item.target -Force
    $deleted += $item.record.path
    $item.record.path | Add-Content -LiteralPath (Join-Path $auditDir 'deleted_paths.txt') -Encoding UTF8
}
$result = [PSCustomObject]@{ completed_utc = [DateTime]::UtcNow.ToString('o'); count = $deleted.Count; freed_bytes = $plan.total_delete_bytes; deleted = $deleted }
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $auditDir 'execution.json') -Encoding UTF8
Write-Output "Deleted $($deleted.Count) verified ANS files; freed $($plan.total_delete_bytes) bytes."
