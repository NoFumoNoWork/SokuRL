param(
    [string[]]$Replays=@(),
    [string[]]$Cases=@(),
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Frames=2500,
    [string]$ReuseCapture='',
    [switch]$CompiledReader,
    [switch]$LiveVS,
    [switch]$SyncInitialRng
)
$ErrorActionPreference='Stop'
if(($Replays.Count -gt 0) -eq ($Cases.Count -gt 0)){throw 'Provide either Replays or Cases, not both'}
$repo=Split-Path $PSScriptRoot -Parent
$original='D:\github\SokuRL'
$python=Join-Path $original '.venv\python.exe'
$target=Join-Path $original 'th123_jp\modules\SokuRLBridge\SokuRLBridge.dll'
$built=Join-Path $repo 'build\bridge\Release\SokuRLBridge.dll'
$destination=[IO.Path]::GetFullPath((Join-Path $repo $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $repo 'docs\reports\replay_commands'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Output must be under docs/reports/replay_commands'
}
if(Test-Path -LiteralPath $destination){throw 'Preserve previous evidence; choose a new output'}
if(Get-Process th123 -ErrorAction SilentlyContinue){throw 'Existing game process; deployment refused'}
if(-not (Test-Path -LiteralPath $built)){throw 'Build Platform Bridge first'}
New-Item -ItemType Directory -Path $destination|Out-Null
$backup=Join-Path $destination 'original.dll'
Copy-Item -LiteralPath $target -Destination $backup
$hash=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
if((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $hash){throw 'Backup hash mismatch'}
Push-Location $repo
try {
    Copy-Item -LiteralPath $built -Destination $target -Force
    $extra=@()
    if($ReuseCapture){$extra=@('--reuse-capture',$ReuseCapture)}
    if($CompiledReader){$extra+=@('--compiled-reader')}
    if($LiveVS){$extra+=@('--live-vs')}
    if($SyncInitialRng){$extra+=@('--sync-initial-rng')}
    if($Cases.Count -gt 0){
        $index=0
        foreach($case in $Cases){
            $index++
            & $python -B tools/run_replay_command_case.py --case $case --output "$destination/case_$index.json"
            if($LASTEXITCODE){throw 'Independent fixture validation failed; preserve evidence'}
        }
    } else {
        & $python -B tools/probe_replay_commands.py @Replays --frames $Frames --output "$destination/run" @extra
        if($LASTEXITCODE){throw 'Replay experiment failed; preserve partial evidence'}
    }
} finally {
    Copy-Item -LiteralPath $backup -Destination $target -Force
    if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash){throw 'DLL restore hash mismatch'}
    Pop-Location
}
