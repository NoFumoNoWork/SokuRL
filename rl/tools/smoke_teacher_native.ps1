param([Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('marisa','reimu')][string]$Character='marisa',
    [ValidateRange(1,1024)][int]$Frames=128,
    [ValidateRange(1,128)][int]$ResetFrames=64)
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
$failure=$null
try{
    $env:TEMP="$destination/temp";$env:TMP="$destination/temp";$env:CUDA_CACHE_PATH="$destination/cuda-cache"
    $env:SOKURL_REPLAY_INPUT_CAPTURE='0'
    & $python -B -c "import runpy,json;from pathlib import Path;runpy.run_path('rl/run.py')['configure']();from game_runtime.identity import fingerprints;Path(r'$destination/before.json').write_text(json.dumps(fingerprints(),indent=2),encoding='utf-8')"
    if($LASTEXITCODE){throw 'Could not preserve baseline identity'}
    Invoke-BridgeSession -GameDirectory "$root/th123_jp" `
        -Dll "$root/platform/build/bridge/Release/SokuRLBridge.dll" `
        -EvidenceDirectory "$destination/session" -Action {
            & $python -B "$root/rl/run.py" smoke_teacher_native --output "$destination/run" `
                --character $Character --frames $Frames --reset-frames $ResetFrames
            if(-not $?){throw 'Native teacher smoke failed;preserve evidence'}
            $nativeReport=Get-Content -LiteralPath "$destination/run/report.json" -Raw|ConvertFrom-Json
            if($nativeReport.status -ne 'PASS_BOUNDED_NATIVE_SIDECAR'){throw 'Native guidance acceptance not PASS'}
        }.GetNewClosure()
}catch{
    $failure=$_
}finally{
    & $python -B -c "import runpy,json;from pathlib import Path;runpy.run_path('rl/run.py')['configure']();from game_runtime.identity import fingerprints;now=fingerprints();Path(r'$destination/after.json').write_text(json.dumps(now,indent=2),encoding='utf-8');assert now==json.loads(Path(r'$destination/before.json').read_text()),'outer deployment restoration mismatch'"
    $restored=$?
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp;$env:CUDA_CACHE_PATH=$oldCache
    $env:SOKURL_REPLAY_INPUT_CAPTURE=$oldCapture
    if(-not $restored){throw 'Restoration verification failed'}
}
if($failure){throw $failure}
