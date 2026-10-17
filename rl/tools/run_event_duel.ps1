param(
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$PpoDirectory='rl/runs/combat-v2-adaptive-20261015/campaign/train-4'
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$destination=[IO.Path]::GetFullPath((Join-Path $root $Output))
if(-not $destination.StartsWith((Join-Path $root 'rl\runs\'),[StringComparison]::OrdinalIgnoreCase) -or
    (Test-Path -LiteralPath $destination)){throw 'Use NEW project-local output'}
if(Get-Process th123 -ErrorAction SilentlyContinue){throw 'Game deployment is occupied'}
New-Item -ItemType Directory -Path "$destination/temp","$destination/cuda-cache" | Out-Null
$python=Join-Path $root '.venv/python.exe'
$oldTemp=$env:TEMP;$oldTmp=$env:TMP;$oldCache=$env:CUDA_CACHE_PATH;$oldCapture=$env:SOKURL_REPLAY_INPUT_CAPTURE
Import-Module "$root/platform/tools/BridgeDeployment.psm1" -Force
try{
    $env:TEMP="$destination/temp";$env:TMP=$env:TEMP;$env:CUDA_CACHE_PATH="$destination/cuda-cache"
    $env:SOKURL_REPLAY_INPUT_CAPTURE='0'
    & $python -B -c "import runpy,json;from pathlib import Path;runpy.run_path('rl/run.py')['configure']();from game_runtime.identity import fingerprints;Path(r'$destination/before.json').write_text(json.dumps(fingerprints(),indent=2))"
    if($LASTEXITCODE){throw 'Before identity failed'}
    Invoke-BridgeSession -GameDirectory "$root/th123_jp" `
        -Dll "$root/platform/build/bridge/Release/SokuRLBridge.dll" `
        -EvidenceDirectory "$destination/session" -Action {
            & $python -B "$root/rl/run.py" evaluate_event_duel --output "$destination/evaluation" `
                --bc ([IO.Path]::GetFullPath((Join-Path $root $Checkpoint))) `
                --ppo-directory ([IO.Path]::GetFullPath((Join-Path $root $PpoDirectory)))
            if($LASTEXITCODE){throw 'Native duel failed;preserve evidence'}
        }.GetNewClosure()
}finally{
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp;$env:CUDA_CACHE_PATH=$oldCache;$env:SOKURL_REPLAY_INPUT_CAPTURE=$oldCapture
    if(Test-Path -LiteralPath "$destination/before.json"){
        & $python -B -c "import runpy,json;from pathlib import Path;runpy.run_path('rl/run.py')['configure']();from game_runtime.identity import fingerprints;p=Path(r'$destination');a=json.loads((p/'before.json').read_text());b=fingerprints();(p/'after.json').write_text(json.dumps(b,indent=2));assert a['artifact_hashes']==b['artifact_hashes'],'Game restore differs';assert a['source_hashes']==b['source_hashes'],'Source identity changed'"
        if($LASTEXITCODE){throw 'After identity differs;stop further deployment'}
    }
}
