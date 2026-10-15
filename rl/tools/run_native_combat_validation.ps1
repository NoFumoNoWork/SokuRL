param(
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Frames=2400,
    [ValidateSet('state','animation')][string]$Mode='animation',
    [string]$Replay='',
    [switch]$RewardDamageCH,
    [switch]$Commands
)
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$root='D:\github\SokuRL'
$python=Join-Path $root '.venv\python.exe'
$cmake='T:\VisualStudio\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$destination=[IO.Path]::GetFullPath((Join-Path $repo $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $repo 'rl\runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Validation outputs must remain inside rl/runs'
}
if(Test-Path -LiteralPath $destination){throw 'Output exists; preserve previous evidence'}
if(Get-Process th123 -ErrorAction SilentlyContinue){throw 'Existing th123 process; no DLL deployment'}
New-Item -ItemType Directory -Path $destination|Out-Null
$target=Join-Path $root 'th123_jp\modules\SokuRLBridge\SokuRLBridge.dll'
$backup=Join-Path $destination 'original.dll'
Copy-Item -LiteralPath $target -Destination $backup
$originalHash=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
if($originalHash -ne (Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash){throw 'Backup verification failed'}
Push-Location $repo
try {
    $extra=@()
    if($Replay){$extra=@('--replay',$Replay)}
    if($RewardDamageCH){$extra+=@('--reward-damage-ch')}
    & $python -B rl/run.py validate_native_combat_capture --baseline-only --mode $Mode --frames $Frames --output "$destination\stock" @extra
    if($LASTEXITCODE){throw 'Original Bridge control failed'}
    & $cmake -S platform/native/SokuRLBridge -B platform/build/bridge -A Win32 "-DSOKUMODS_ROOT=$root/third_party/SokuMods"
    if($LASTEXITCODE){throw 'Native configure failed'}
    & $cmake --build platform/build/bridge --config Release --parallel 4
    if($LASTEXITCODE){throw 'Native build failed'}
    $sources=Join-Path $destination 'sources'
    New-Item -ItemType Directory -Path $sources|Out-Null
    foreach($file in 'CombatCapture.cpp','CombatCapture.hpp','InputCounters.hpp','SokuRLBridge.cpp','NativeTrace.cpp','CMakeLists.txt'){
        Copy-Item -LiteralPath (Join-Path $repo "platform/native/SokuRLBridge/$file") -Destination $sources
    }
    foreach($file in 'RL/tools/game_runtime/combat_capture.py','RL/src/soku_rl/env/input_lineage.py',
                    'RL/src/soku_rl/env/reward.py','RL/tools/game_runtime/observation.py',
                    'RL/tools/validate_native_combat_capture.py','RL/tools/validate_command_player.py',
                    'RL/src/soku_rl/env/commands.py','RL/tools/game_runtime/action_state.py',
                    'platform/src/sokurl_platform/command_input.py','platform/src/sokurl_platform/native_action_state.py',
                    'platform/src/sokurl_platform/combat_capture.py','platform/src/sokurl_platform/run_command_input.py'){
        Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $sources
    }
    Get-ChildItem -LiteralPath $sources -File | Get-FileHash -Algorithm SHA256 |
        Select-Object Path,Hash | ConvertTo-Json | Set-Content -LiteralPath "$destination/source_hashes.json"
    Copy-Item -LiteralPath (Join-Path $repo 'platform/build/bridge/Release/SokuRLBridge.dll') -Destination "$destination\observer.dll"
    try {
        Copy-Item -LiteralPath "$destination\observer.dll" -Destination $target -Force
        & $python -B rl/run.py validate_native_combat_capture --mode $Mode --frames $Frames --output "$destination\observer" @extra
        if($LASTEXITCODE){throw 'Observer/native parity or coverage failed'}
        if($Commands){
            & $python -B rl/run.py validate_command_player --output "$destination\commands.json"
            if($LASTEXITCODE){throw 'Generic command player validation failed'}
        }
    } finally {
        Copy-Item -LiteralPath $backup -Destination $target -Force
        if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $originalHash){throw 'Original DLL restoration failed'}
    }
    # The original DLL and rebuilt default-OFF DLL must also agree, not only OFF/ON.
    $stock=Get-Content "$destination\stock\off.json" -Raw|ConvertFrom-Json
    $off=Get-Content "$destination\observer\off.json" -Raw|ConvertFrom-Json
    if($stock.states.Count -ne $off.states.Count){throw 'Stock/rebuilt frame lengths differ'}
    for($i=0;$i -lt $stock.states.Count;$i++){
        if($stock.states[$i].hash -ne $off.states[$i].hash){throw "Stock/rebuilt hash differs at frame $($i+1)"}
    }
    Write-Output 'PASS: stock/rebuilt OFF/ON native parity, contact accounting; original DLL restored'
} finally { Pop-Location }
