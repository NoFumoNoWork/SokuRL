[CmdletBinding(DefaultParameterSetName='Record')]
param(
    [Parameter(Mandatory=$true,ParameterSetName='Record')][string]$TrainingDirectory,
    [Parameter(Mandatory=$true,ParameterSetName='Watch')][string]$Catalog,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(ParameterSetName='Record')][string]$Checkpoint='final.zip',
    [Parameter(ParameterSetName='Record')][string]$Opponent='pressure',
    [Parameter(ParameterSetName='Record')][uint32]$Seed=918042743,
    [Parameter(ParameterSetName='Record')][ValidateSet('left','right','both')][string]$Seat='left',
    [Parameter(ParameterSetName='Record')][string]$Device='cpu',
    [Parameter(ParameterSetName='Record')][switch]$Watch,
    [int]$Index=0,
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
function AbsolutePath([string]$Path) {
    if([IO.Path]::IsPathRooted($Path)){return [IO.Path]::GetFullPath($Path)}
    return [IO.Path]::GetFullPath((Join-Path $root $Path))
}
$destination=AbsolutePath $Output
$allowed=[IO.Path]::GetFullPath((Join-Path $root 'rl/runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be under rl/runs'}
if(Test-Path -LiteralPath $destination){throw 'Preserve existing run: choose a new Output'}
$python=Join-Path $root '.venv/python.exe'
$launcher=Join-Path $root 'rl/run.py'
$dll=Join-Path $root 'platform/build/bridge/Release/SokuRLBridge.dll'
$game=if($env:SOKURL_GAME_DIR){$env:SOKURL_GAME_DIR}else{Join-Path $root 'th123_jp'}
if($PSCmdlet.ParameterSetName -eq 'Record'){
    $training=AbsolutePath $TrainingDirectory
    $recordArgs=@('--training-directory',$training,'--checkpoint',$Checkpoint,
                  '--output',"$destination/battles",'--opponents',$Opponent,
                  '--seeds',"$Seed",'--seat',$Seat,'--device',$Device)
    & $python -B $launcher record_battles @recordArgs --dry-run
    if($LASTEXITCODE){throw 'Recording preflight failed'}
    $catalogPath="$destination/battles/catalog.json"
}else{
    $catalogPath=AbsolutePath $Catalog
    & $python -B $launcher watch_battle --catalog $catalogPath --index $Index --dry-run
    if($LASTEXITCODE){throw 'Playback preflight failed'}
}
if($DryRun){return}
if(-not (Test-Path -LiteralPath $dll)){throw 'Build the compatible Platform Bridge first'}
if(Get-Process th123 -ErrorAction SilentlyContinue){throw 'Existing game process; deployment refused'}
Import-Module (Join-Path $root 'platform/tools/BridgeDeployment.psm1') -Force
$oldTemp=$env:TEMP; $oldTmp=$env:TMP
try{
    New-Item -ItemType Directory -Path "$destination/temp" | Out-Null
    $env:TEMP="$destination/temp"; $env:TMP="$destination/temp"
    $mode=$PSCmdlet.ParameterSetName
    Invoke-BridgeSession -GameDirectory $game -Dll $dll -EvidenceDirectory "$destination/session" -Action {
        if($mode -eq 'Record'){
            & $python -B $launcher record_battles @recordArgs
            if($LASTEXITCODE){throw 'Recording failed; preserve evidence'}
        }
        if($mode -eq 'Watch' -or $Watch){
            & $python -B $launcher watch_battle --catalog $catalogPath --index $Index
            if($LASTEXITCODE){throw 'Playback failed; preserve evidence'}
        }
    }.GetNewClosure()
    Write-Output "Replay catalog: $catalogPath"
}finally{
    $env:TEMP=$oldTemp; $env:TMP=$oldTmp
}
