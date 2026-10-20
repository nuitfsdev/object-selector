[CmdletBinding()]
param(
    [switch]$AcceptScanNetTos,
    [ValidateRange(1, 64)]
    [int]$Workers = 16,
    [string]$OutputDirectory = "",
    [string]$ArchivePath = "",
    [switch]$SkipArchive,
    [switch]$SkipHash
)

$ErrorActionPreference = "Stop"
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceDirectory = Split-Path -Parent $ScriptDirectory

if (-not $AcceptScanNetTos) {
    throw "Read and accept https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf, then rerun with -AcceptScanNetTos."
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $WorkspaceDirectory "data\scannet"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

$PythonExecutable = $null
$PythonPrefixArguments = @()
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExecutable = (Get-Command python).Source
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExecutable = (Get-Command py).Source
    $PythonPrefixArguments = @("-3")
}
else {
    throw "Python 3 was not found. Install Python 3 and enable 'Add Python to PATH'."
}

$TrainAnnotations = Join-Path $WorkspaceDirectory "data\scanqa\ScanQA_v1.0_train.json"
$ValAnnotations = Join-Path $WorkspaceDirectory "data\scanqa\ScanQA_v1.0_val.json"
foreach ($AnnotationPath in @($TrainAnnotations, $ValAnnotations)) {
    if (-not (Test-Path -LiteralPath $AnnotationPath -PathType Leaf)) {
        throw "Missing ScanQA annotation: $AnnotationPath"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
Write-Host "Downloading the ScanNet files needed by ScanQA train + val..."
& $PythonExecutable @PythonPrefixArguments `
    (Join-Path $ScriptDirectory "download_scanqa_scannet.py") `
    --split both `
    --workers $Workers `
    --accept-scannet-tos `
    --output $OutputDirectory `
    --verify
if ($LASTEXITCODE -ne 0) {
    throw "ScanNet downloader failed with exit code $LASTEXITCODE. Run this script again to resume completed files."
}

$Rows = @()
$Rows += Get-Content -LiteralPath $TrainAnnotations -Raw | ConvertFrom-Json
$Rows += Get-Content -LiteralPath $ValAnnotations -Raw | ConvertFrom-Json
$SceneIds = @($Rows | ForEach-Object { $_.scene_id } | Sort-Object -Unique)
$Suffixes = @(
    "_vh_clean_2.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json"
)
$MissingFiles = [System.Collections.Generic.List[string]]::new()
$TotalBytes = [int64]0
foreach ($SceneId in $SceneIds) {
    foreach ($Suffix in $Suffixes) {
        $FilePath = Join-Path $OutputDirectory "scans\$SceneId\$SceneId$Suffix"
        if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
            $MissingFiles.Add($FilePath)
            continue
        }
        $FileInfo = Get-Item -LiteralPath $FilePath
        if ($FileInfo.Length -le 0) {
            $MissingFiles.Add($FilePath)
            continue
        }
        $TotalBytes += $FileInfo.Length
    }
}
if ($MissingFiles.Count -gt 0) {
    $Preview = ($MissingFiles | Select-Object -First 10) -join "`n"
    throw "Download verification failed: $($MissingFiles.Count) files are missing or empty.`n$Preview"
}

$ExpectedFiles = $SceneIds.Count * $Suffixes.Count
$TotalGiB = [Math]::Round($TotalBytes / 1GB, 2)
Write-Host "READY: $($SceneIds.Count) scenes, $ExpectedFiles files, $TotalGiB GiB"

if ($SkipArchive) {
    Write-Host "Archive creation skipped. Data: $OutputDirectory"
    exit 0
}

if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    throw "tar.exe was not found. Install Windows tar or rerun with -SkipArchive."
}
if ([string]::IsNullOrWhiteSpace($ArchivePath)) {
    $TransferDirectory = Join-Path $WorkspaceDirectory "outputs\scannet_transfer"
    $ArchivePath = Join-Path $TransferDirectory "scannet_scanqa_$($SceneIds.Count)_scenes.tar"
}
$ArchivePath = [System.IO.Path]::GetFullPath($ArchivePath)
$ArchiveDirectory = Split-Path -Parent $ArchivePath
New-Item -ItemType Directory -Force -Path $ArchiveDirectory | Out-Null
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Write-Host "Creating an uncompressed transfer archive (requires roughly another $TotalGiB GiB free)..."
& tar.exe -cf $ArchivePath -C $OutputDirectory scans
if ($LASTEXITCODE -ne 0) {
    throw "tar.exe failed with exit code $LASTEXITCODE."
}

$ArchiveInfo = Get-Item -LiteralPath $ArchivePath
$Manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    scenes = $SceneIds.Count
    files = $ExpectedFiles
    source_bytes = $TotalBytes
    archive_path = $ArchiveInfo.FullName
    archive_bytes = $ArchiveInfo.Length
    sha256 = $null
}
if (-not $SkipHash) {
    Write-Host "Computing SHA256 for transfer verification..."
    $Manifest.sha256 = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash
}
$ManifestPath = "$ArchivePath.manifest.json"
$Manifest | ConvertTo-Json | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

Write-Host "UPLOAD THESE FILES TO GOOGLE DRIVE:"
Write-Host "  $ArchivePath"
Write-Host "  $ManifestPath"
Write-Host "On the VPS, extract the archive into the repository data directory, then run:"
Write-Host "  bash run_vps_all.sh --skip-download"
