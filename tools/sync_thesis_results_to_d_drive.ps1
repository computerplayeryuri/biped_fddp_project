[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$SourceRoot = "\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project",
    [string]$DestinationRoot = "D:\毕业论文\机器人控制流程图\FDDP仿真结果归档"
)

$ErrorActionPreference = "Stop"
$sourceSets = @(
    "results",
    "transfer\trajectories",
    "docs"
)

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source root does not exist: $SourceRoot"
}

$sourceFiles = foreach ($relativeRoot in $sourceSets) {
    $absoluteRoot = Join-Path $SourceRoot $relativeRoot
    if (-not (Test-Path -LiteralPath $absoluteRoot -PathType Container)) {
        throw "Required result directory does not exist: $absoluteRoot"
    }
    Get-ChildItem -LiteralPath $absoluteRoot -Recurse -File | ForEach-Object {
        [pscustomobject]@{
            Source = $_.FullName
            RelativePath = $_.FullName.Substring($SourceRoot.TrimEnd("\").Length + 1)
            Length = $_.Length
        }
    }
}

if ($WhatIfPreference) {
    $totalBytes = ($sourceFiles | Measure-Object Length -Sum).Sum
    Write-Output "Would archive $($sourceFiles.Count) files ($totalBytes bytes)."
    Write-Output "Destination: $DestinationRoot"
    return
}

New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null

foreach ($file in $sourceFiles) {
    $destination = Join-Path $DestinationRoot $file.RelativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    if ($PSCmdlet.ShouldProcess($destination, "Copy $($file.Source)")) {
        Copy-Item -LiteralPath $file.Source -Destination $destination -Force
    }
}

$readmePath = Join-Path $DestinationRoot "归档说明.md"
$syncTime = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
$readmeLines = @(
    "# FDDP 仿真结果归档",
    "",
    "- Source workspace: $SourceRoot",
    "- Last sync: $syncTime",
    "- Included: results/, transfer/trajectories/, docs/",
    "- Excluded: source code and real-hardware claims",
    "- Rule: every future experiment must run tools/sync_thesis_results_to_d_drive.ps1 after final verification.",
    "",
    "项目 workspace 保持 source of truth；本目录是论文材料的 D 盘镜像。tau_ref 仍仅用于 planning/analysis，不能解释为 hardware torque command。"
)
$readme = $readmeLines -join [Environment]::NewLine
Set-Content -LiteralPath $readmePath -Value $readme -Encoding UTF8

$verified = foreach ($file in $sourceFiles) {
    $destination = Join-Path $DestinationRoot $file.RelativePath
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        throw "Archive verification failed; missing: $destination"
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.Source).Hash
    $destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash
    if ($sourceHash -ne $destinationHash) {
        throw "Archive verification failed; hash mismatch: $($file.RelativePath)"
    }
    [pscustomobject]@{
        RelativePath = $file.RelativePath.Replace("\", "/")
        Bytes = $file.Length
        SHA256 = $destinationHash
    }
}

$manifestPath = Join-Path $DestinationRoot "SHA256_MANIFEST.csv"
$verified | Sort-Object RelativePath | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$totalBytes = ($verified | Measure-Object Bytes -Sum).Sum
Write-Output "Archived and verified $($verified.Count) files ($totalBytes bytes)."
Write-Output "Destination: $DestinationRoot"
Write-Output "Manifest: $manifestPath"

