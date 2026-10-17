[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateRange(1,10)][int]$Games=10,
    [ValidateRange(1,18000)][int]$MaxFrames=18000,
    [switch]$Smoke,
    [string]$TrainingDirectory='',
    [string]$Checkpoint='final.zip',
    [string]$Device='cpu'
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'rl/runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be under rl/runs'}
if(Test-Path -LiteralPath $destination){throw 'Preserve existing output'}
& (Join-Path $root '.venv/python.exe') -B -c 'from lupa.lua51 import LuaRuntime; print(LuaRuntime().lua_implementation)'
if($LASTEXITCODE){throw 'Install the declared god optional dependencies before deploying'}
Import-Module (Join-Path $root 'platform/tools/BridgeDeployment.psm1') -Force
$oldTemp=$env:TEMP; $oldTmp=$env:TMP
try {
    New-Item -ItemType Directory -Path "$destination/temp" | Out-Null
    $env:TEMP="$destination/temp"; $env:TMP="$destination/temp"
    Invoke-BridgeSession -GameDirectory (Join-Path $root 'th123_jp') -Dll (Join-Path $root 'platform/build/bridge/Release/SokuRLBridge.dll') -EvidenceDirectory "$destination/session" -Action {
        $options=@()
        if($Smoke){$options+='--smoke'}
        if($TrainingDirectory){$options+=@('--training-directory',$TrainingDirectory,'--checkpoint',$Checkpoint,'--device',$Device)}
        & (Join-Path $root '.venv/python.exe') -B (Join-Path $root 'rl/run.py') native_cpu_benchmark --output "$destination/experiment" --games $Games --max-frames $MaxFrames @options
        if($LASTEXITCODE){throw 'Native CPU benchmark failed; evidence preserved'}
    }.GetNewClosure()
} finally {
    $env:TEMP=$oldTemp; $env:TMP=$oldTmp
}
