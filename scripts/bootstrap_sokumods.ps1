[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LockFile = Join-Path $RepoRoot "config\dependencies.lock.json"
$Lock = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json
$SokuModsUrl = $Lock.sokumods.repository
$SokuModsCommit = $Lock.sokumods.commit
$SkipIntroCommit = $Lock.skipintro.upstream_commit
$SokuModsDir = Join-Path $RepoRoot "third_party\SokuMods"
$SkipIntroDir = Join-Path $SokuModsDir "modules\SkipIntro\Soku-SkipIntro"
$PatchFile = Join-Path $RepoRoot $Lock.skipintro.patch
$VerifyScript = Join-Path $PSScriptRoot "verify_sokumods.ps1"

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found in PATH."
}

if (-not (Test-Path -LiteralPath $SokuModsDir)) {
    Invoke-Git clone --no-checkout $SokuModsUrl $SokuModsDir
    Invoke-Git -C $SokuModsDir checkout --detach $SokuModsCommit
    Invoke-Git -C $SokuModsDir submodule update --init --recursive
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $SokuModsDir ".git"))) {
        throw "$SokuModsDir exists but is not a Git checkout."
    }

    $actualSokuModsCommit = (& git -C $SokuModsDir rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualSokuModsCommit -ne $SokuModsCommit) {
        throw "SokuMods must be at $SokuModsCommit; found $actualSokuModsCommit."
    }

    if (-not (Test-Path -LiteralPath (Join-Path $SkipIntroDir ".git"))) {
        Invoke-Git -C $SokuModsDir submodule update --init --recursive
    }
}

$actualSkipIntroCommit = (& git -C $SkipIntroDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualSkipIntroCommit -ne $SkipIntroCommit) {
    throw "SkipIntro must be at $SkipIntroCommit; found $actualSkipIntroCommit."
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& git -C $SkipIntroDir apply --check -- $PatchFile *> $null
$patchApplies = $LASTEXITCODE -eq 0
$ErrorActionPreference = $previousErrorActionPreference

if ($patchApplies) {
    Invoke-Git -C $SkipIntroDir apply -- $PatchFile
} else {
    $ErrorActionPreference = "SilentlyContinue"
    & git -C $SkipIntroDir apply --reverse --check -- $PatchFile *> $null
    $patchAlreadyApplied = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $previousErrorActionPreference
    if (-not $patchAlreadyApplied) {
        throw "The SokuRL SkipIntro patch is neither applicable nor already applied."
    }
}

& $VerifyScript

Write-Output "SokuMods:  $SokuModsCommit"
Write-Output "SkipIntro: $SkipIntroCommit + patches/skipintro-replaydnd-command-line.patch"
Write-Output "Ready:     $SokuModsDir"
