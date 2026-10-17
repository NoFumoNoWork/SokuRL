param(
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Frames=7200,
    [int]$Limit=20,
    [switch]$Resume,
    [switch]$PrefixOnly,
    [ValidateSet('train','validation')][string]$Partition='train',
    [string]$Plan=''
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[System.IO.Path]::GetFullPath((Join-Path $root $Output))
$allowed=Join-Path $root 'rl\runs\'
if(-not $destination.StartsWith($allowed,[System.StringComparison]::OrdinalIgnoreCase)){
    throw 'Output must be inside project rl/runs'
}
$session="$destination-deployment-$(Get-Date -Format yyyyMMdd-HHmmss)"
$temp="$session/temp"
New-Item -ItemType Directory -Path $temp -Force | Out-Null
Import-Module "$root/platform/tools/BridgeDeployment.psm1" -Force
$oldTemp=$env:TEMP;$oldTmp=$env:TMP
try{
    $env:TEMP=$temp;$env:TMP=$temp
    $arguments=@('-B',"$root/rl/run.py",'export_human_replays',
        '--audit',"$root/rl/runs/human-bc-20261016/audit-v3",
        '--source',"$root/rl/runs/combat-v2-adaptive-20261015/campaign/train-4",
        '--output',$destination,'--frames',"$Frames",'--limit',"$Limit",'--partition',$Partition)
    if($Resume){$arguments+='--resume'}
    if($PrefixOnly){$arguments+='--prefix-only'}
    if($Plan){$arguments+=@('--plan',[IO.Path]::GetFullPath((Join-Path $root $Plan)))}
    Invoke-BridgeSession -GameDirectory "$root/th123_jp" `
        -Dll "$root/platform/build/bridge/Release/SokuRLBridge.dll" `
        -EvidenceDirectory "$session/session" -Action {
            & "$root/.venv/python.exe" @arguments
            if($LASTEXITCODE){throw 'Human export failed; preserved evidence,do not train'}
        }.GetNewClosure()
}finally{
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp
}
