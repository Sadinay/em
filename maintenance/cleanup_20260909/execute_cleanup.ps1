$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$workspacePrefix = $workspaceRoot.TrimEnd('\') + '\'
$legacyPrefix = (Join-Path $workspaceRoot '01_legacy_topopt_prototype') + '\'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'delete_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$checkedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$candidateDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)

# Validate every absolute target, file size and parent directory before any deletion.
foreach ($entry in $plan) {
    $targetPath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.path))
    if (-not $targetPath.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $targetPath" }
    if ($targetPath -match '[\\/]\.git[\\/]') { throw "Git metadata is not a deletion target: $targetPath" }
    if ($entry.reason -ne 'regenerable_python_cache' -and -not $targetPath.StartsWith($legacyPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside legacy project: $targetPath" }
    $file = Get-Item -LiteralPath $targetPath -Force
    if ($file.PSIsContainer -or $file.Length -ne $entry.bytes) { throw "File changed since inventory: $targetPath" }
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse file: $targetPath" }
    $directory = $file.Directory
    while ($directory -and $directory.FullName.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (-not $checkedDirectories.Add($directory.FullName)) { break }
        if (($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse directory: $($directory.FullName)" }
        $directory = $directory.Parent
    }
    $null = $candidateDirectories.Add($file.DirectoryName)
}

$logPath = Join-Path $PSScriptRoot 'deletion_log.jsonl'
$writer = [IO.StreamWriter]::new($logPath, $false, [Text.UTF8Encoding]::new($false))
$deletedCount = 0
$deletedBytes = [long]0
try {
    foreach ($entry in $plan) {
        $targetPath = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.path))
        Remove-Item -LiteralPath $targetPath -Force
        $writer.WriteLine(($entry | ConvertTo-Json -Compress))
        $deletedCount++
        $deletedBytes += $entry.bytes
        if ($deletedCount % 5000 -eq 0) { Write-Output "Deleted $deletedCount / $($plan.Count) files"; $writer.Flush() }
    }
} finally { $writer.Dispose() }

# Only remove directories made empty by this deletion; never recurse on an unchecked target.
foreach ($directoryPath in ($candidateDirectories | Sort-Object Length -Descending)) {
    $currentDirectory = $directoryPath
    while ($currentDirectory.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (-not (Test-Path -LiteralPath $currentDirectory)) { break }
        if (@(Get-ChildItem -LiteralPath $currentDirectory -Force).Count -ne 0) { break }
        Remove-Item -LiteralPath $currentDirectory -Force
        $currentDirectory = [IO.Path]::GetDirectoryName($currentDirectory)
    }
}
Write-Output "Deleted $deletedCount files, $deletedBytes bytes"
