param([Parameter(Mandatory=$true)][string]$Output,[string]$Checkpoint='')
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
if(-not $destination.StartsWith((Join-Path $root 'rl\runs\'),[StringComparison]::OrdinalIgnoreCase) -or
   (Test-Path -LiteralPath $destination)){throw 'Use a NEW project-local runs output'}
$python=Join-Path $root '.venv/python.exe'
New-Item -ItemType Directory -Path "$destination/temp","$destination/cuda-cache" | Out-Null
$oldTemp=$env:TEMP;$oldTmp=$env:TMP;$oldCache=$env:CUDA_CACHE_PATH
$oldCapture=$env:SOKURL_REPLAY_INPUT_CAPTURE
Import-Module "$root/platform/tools/BridgeDeployment.psm1" -Force
try{
    $env:TEMP="$destination/temp";$env:TMP="$destination/temp";$env:CUDA_CACHE_PATH="$destination/cuda-cache"
    $env:SOKURL_REPLAY_INPUT_CAPTURE='0'
    & $python -B -c "import runpy,json;from pathlib import Path;runpy.run_path('rl/run.py')['configure']();from game_runtime.identity import fingerprints;Path(r'$destination/before.json').write_text(json.dumps(fingerprints(),indent=2),encoding='utf-8')"
    if($LASTEXITCODE){throw 'Could not preserve before identity'}
    Invoke-BridgeSession -GameDirectory "$root/th123_jp" `
        -Dll "$root/platform/build/bridge/Release/SokuRLBridge.dll" `
        -EvidenceDirectory "$destination/session" -Action {
            $arguments=@('-B',"$root/rl/run.py",'smoke_hierarchical_native','--output',"$destination/run")
            if($Checkpoint){$arguments+=@('--checkpoint',$Checkpoint)}
            & $python @arguments
            if($LASTEXITCODE){throw 'Hierarchical native smoke failed;preserve evidence'}
        }.GetNewClosure()
}finally{
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp;$env:CUDA_CACHE_PATH=$oldCache
    $env:SOKURL_REPLAY_INPUT_CAPTURE=$oldCapture
}
