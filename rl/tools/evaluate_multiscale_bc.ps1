param(
    [Parameter(Mandatory=$true)][string]$TrainingRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('A','B','C','D')][string[]]$Groups=@('A','B','C')
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
$training=[IO.Path]::GetFullPath((Join-Path $root $TrainingRoot))
$allowed=Join-Path $root 'rl\runs\'
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or
   -not $training.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or
   (Test-Path -LiteralPath $destination)){throw 'Use project-local training and a NEW runs output'}
$python=Join-Path $root '.venv/python.exe'
# Four held-out paired worlds x both seats;identical policy seeds via benchmark.
$seeds=@('974220019','974220131','974220257','974220379')
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
            foreach($group in $Groups){
                foreach($seed in $seeds){
                    & $python -B "$root/rl/run.py" record_battles --training-directory "$training/$group" `
                        --checkpoint best-actor.zip --bc-context "$training/$group/best.json" `
                        --output "$destination/$group/seed-$seed" --opponents pressure `
                        --num-envs 2 --device cpu --seat both --seeds $seed
                    if($LASTEXITCODE){throw "BC paired recording failed:$group/$seed;preserve evidence"}
                }
            }
        }.GetNewClosure()
}finally{
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp;$env:CUDA_CACHE_PATH=$oldCache
    $env:SOKURL_REPLAY_INPUT_CAPTURE=$oldCapture
}
