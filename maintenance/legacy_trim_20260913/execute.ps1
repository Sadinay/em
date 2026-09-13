$ErrorActionPreference = 'Stop'
$auditDir = $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $auditDir 'execution.json')) { throw 'Already executed.' }
$plan = Get-Content -LiteralPath (Join-Path $auditDir 'plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$rootPath = [IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$expectedRoot = [IO.Path]::GetFullPath((Join-Path $auditDir '..\..\01_legacy_topopt_prototype')).TrimEnd('\')
if ($rootPath -ne $expectedRoot) { throw 'Unexpected project root.' }
foreach ($entry in @($plan.retained) + @($plan.archives)) {
    if ((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Preserved file changed: $($entry.path)" }
}
$targets = @()
foreach ($entry in $plan.delete) {
    $targetPath = [IO.Path]::GetFullPath($entry.path)
    if (-not $targetPath.StartsWith($rootPath + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Target outside 01: $targetPath" }
    $item = Get-Item -LiteralPath $targetPath -Force
    if ($item.PSIsContainer -or $item.Length -ne $entry.bytes) { throw "Target changed: $targetPath" }
    $ancestor = $item
    while ($ancestor.FullName -ne $rootPath) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point: $targetPath" }
        $ancestor = Get-Item -LiteralPath (Split-Path -Parent $ancestor.FullName) -Force
    }
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Target content changed: $targetPath" }
    $targets += $targetPath
}
foreach ($targetPath in $targets) {
    Remove-Item -LiteralPath $targetPath -Force
    $targetPath | Add-Content -LiteralPath (Join-Path $auditDir 'deleted_paths.txt') -Encoding UTF8
}
[PSCustomObject]@{ completed_utc = [DateTime]::UtcNow.ToString('o'); deleted_count = $targets.Count; deleted_paths = $targets } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $auditDir 'execution.json') -Encoding UTF8
Write-Output "Removed $($targets.Count) files from retired 01 project."
