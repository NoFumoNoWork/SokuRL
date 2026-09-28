[CmdletBinding()]
param(
    [string]$BuildDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LockFile = Join-Path $RepoRoot "config\dependencies.lock.json"
$Lock = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
$SokuModsDir = Join-Path $RepoRoot "third_party\SokuMods"
$SkipIntroDir = Join-Path $SokuModsDir "modules\SkipIntro\Soku-SkipIntro"

function Assert-Equal {
    param([string]$Label, [string]$Actual, [string]$Expected)

    if ($Actual -cne $Expected) {
        throw "$Label mismatch: expected $Expected, found $Actual"
    }
}

function Get-GitValue {
    param([string]$Directory, [string]$Expression)

    $value = (& git -C $Directory rev-parse $Expression).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "git rev-parse $Expression failed in $Directory"
    }
    return $value
}

function Get-Sha256 {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

Assert-Equal "SokuMods commit" `
    (Get-GitValue $SokuModsDir "HEAD") $Lock.sokumods.commit
Assert-Equal "SokuMods tree" `
    (Get-GitValue $SokuModsDir "HEAD^{tree}") $Lock.sokumods.tree_sha1
Assert-Equal "SkipIntro commit" `
    (Get-GitValue $SkipIntroDir "HEAD") $Lock.skipintro.upstream_commit
Assert-Equal "SkipIntro upstream tree" `
    (Get-GitValue $SkipIntroDir "HEAD^{tree}") $Lock.skipintro.upstream_tree_sha1

$PatchFile = Join-Path $RepoRoot $Lock.skipintro.patch
Assert-Equal "SkipIntro patch SHA-256" `
    (Get-Sha256 $PatchFile) $Lock.skipintro.patch_sha256

$expectedChangedFiles = @($Lock.skipintro.patched_files | ForEach-Object { $_.path } | Sort-Object)
$actualChangedFiles = @(& git -C $SkipIntroDir diff --name-only | Sort-Object)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the SkipIntro worktree."
}
$changedFileDiff = @(Compare-Object $expectedChangedFiles $actualChangedFiles)
if ($changedFileDiff.Count -ne 0) {
    throw "SkipIntro changed-file set does not match the dependency lock."
}

foreach ($file in $Lock.skipintro.patched_files) {
    $sourceFile = Join-Path $SkipIntroDir $file.path
    Assert-Equal "Patched SkipIntro file $($file.path) SHA-256" `
        (Get-Sha256 $sourceFile) $file.sha256
}

if ($BuildDirectory) {
    $resolvedBuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
    foreach ($output in $Lock.verified_build.outputs) {
        $outputFile = Join-Path $resolvedBuildDirectory $output.path
        Assert-Equal "Build output $($output.path) SHA-256" `
            (Get-Sha256 $outputFile) $output.sha256

        $bytes = [System.IO.File]::ReadAllBytes($outputFile)
        $peOffset = [BitConverter]::ToInt32($bytes, 0x3c)
        $machine = [BitConverter]::ToUInt16($bytes, $peOffset + 4)
        if ($machine -ne 0x014c) {
            throw "Build output $($output.path) is not x86 (machine=0x$($machine.ToString('X4')))."
        }
    }
}

Write-Output "SokuMods dependency lock verified."
if ($BuildDirectory) {
    Write-Output "Verified build outputs and x86 PE architecture: $resolvedBuildDirectory"
}
