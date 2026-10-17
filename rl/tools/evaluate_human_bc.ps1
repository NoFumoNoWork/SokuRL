param(
    [Parameter(Mandatory=$true)][string]$Candidate,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
$allowed=Join-Path $root 'rl\runs\'
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or
   (Test-Path -LiteralPath $destination)){throw 'Use a new project rl/runs output'}
$candidatePath=[IO.Path]::GetFullPath((Join-Path $root $Candidate))
if(-not $candidatePath.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Candidate must be a project-local accepted training artifact'
}
$python=Join-Path $root '.venv/python.exe'
$source=Join-Path $root 'rl/runs/combat-v2-adaptive-20261015/campaign/train-4'
$seeds=@('974210001','974210127','974210239','974210363',
         '974210489','974210593','974210707','974210831')
New-Item -ItemType Directory -Path "$destination/temp","$destination/cuda-cache" | Out-Null
$oldTemp=$env:TEMP;$oldTmp=$env:TMP;$oldCache=$env:CUDA_CACHE_PATH
$oldCapture=$env:SOKURL_REPLAY_INPUT_CAPTURE
Import-Module "$root/platform/tools/BridgeDeployment.psm1" -Force
try{
    $env:TEMP="$destination/temp";$env:TMP="$destination/temp"
    $env:CUDA_CACHE_PATH="$destination/cuda-cache"
    $env:SOKURL_REPLAY_INPUT_CAPTURE='0'
    Invoke-BridgeSession -GameDirectory "$root/th123_jp" `
        -Dll "$root/platform/build/bridge/Release/SokuRLBridge.dll" `
        -EvidenceDirectory "$destination/session" -Action {
            # One two-seat batch per fresh Worker avoids the known post-KO
            # scene-reset failure of the recording configuration.
            foreach($entry in @(@('baseline',$source,'final.zip'),@('human-bc',$candidatePath,'best.zip'))){
                foreach($seed in $seeds){
                    & $python -B "$root/rl/run.py" record_battles --training-directory $entry[1] `
                        --checkpoint $entry[2] --output "$destination/$($entry[0])/seed-$seed" `
                        --opponents pressure --num-envs 2 --device cpu --seat both --seeds $seed
                    if($LASTEXITCODE){throw "Paired human BC recording failed:$($entry[0])/$seed;preserve evidence"}
                }
            }
        }.GetNewClosure()
}finally{
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp;$env:CUDA_CACHE_PATH=$oldCache
    $env:SOKURL_REPLAY_INPUT_CAPTURE=$oldCapture
}
