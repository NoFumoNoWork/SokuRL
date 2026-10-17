[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('validate','smoke','train','evaluate')][string]$Stage,
    [Parameter(Mandatory=$true)][ValidateSet('V2','V3C')][string]$Arm,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Frames=0,
    [string]$PairedInitial='',
    [string]$Model='',
    [switch]$RewardsOnly
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'rl/runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be under rl/runs'}
if(Test-Path -LiteralPath $destination){throw 'Preserve existing output'}
$python=Join-Path $root '.venv/python.exe'
$game=Join-Path $root 'th123_jp'
Import-Module (Join-Path $root 'platform/tools/BridgeDeployment.psm1') -Force
$oldTemp=$env:TEMP; $oldTmp=$env:TMP
try {
    New-Item -ItemType Directory -Path "$destination/temp" | Out-Null
    $env:TEMP="$destination/temp"; $env:TMP="$destination/temp"
    Invoke-BridgeSession -GameDirectory $game -Dll (Join-Path $root 'platform/build/bridge/Release/SokuRLBridge.dll') -EvidenceDirectory "$destination/session" -Action {
        $options=@()
        if($Frames){$options+=@('--frames',"$Frames")}
        if($PairedInitial){$options+=@('--paired-initial',[IO.Path]::GetFullPath((Join-Path $root $PairedInitial)))}
        if($Model){$options+=@('--model',[IO.Path]::GetFullPath((Join-Path $root $Model)))}
        if($RewardsOnly){$options+='--rewards-only'}
        & $python -B (Join-Path $root 'rl/run.py') compare_v3c --stage $Stage --arm $Arm --output "$destination/experiment" @options
        if($LASTEXITCODE){throw 'V3-C experiment failed; evidence preserved'}
    }.GetNewClosure()
} finally {
    $env:TEMP=$oldTemp; $env:TMP=$oldTmp
}
