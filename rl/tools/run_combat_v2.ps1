param(
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Timesteps=0,
    [string]$ConfigName='train_combat_v2',
    [switch]$IdlePenaltyTest,
    [switch]$PlatformIntegrationTest,
    [switch]$LearningCampaign,
    [string]$ResumeCampaign='',
    [string]$ContinueFrom='',
    [string]$EvaluateBaseline='',
    [string]$EvaluateCandidate='',
    [string[]]$TrainOverrides=@()
)
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$game=if($env:SOKURL_GAME_DIR){$env:SOKURL_GAME_DIR}else{Join-Path $repo 'th123_jp'}
$python=Join-Path $repo '.venv\python.exe'
$destination=[IO.Path]::GetFullPath((Join-Path $repo $Output))
$allowed=[IO.Path]::GetFullPath((Join-Path $repo 'rl\runs'))+[IO.Path]::DirectorySeparatorChar
if(-not $destination.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output must be in rl/runs'}
if(Test-Path -LiteralPath $destination){throw 'Preserve prior evidence'}
if([bool]$EvaluateBaseline -ne [bool]$EvaluateCandidate){throw 'Paired evaluation requires both training directories'}
$built=Join-Path $repo 'platform\build\bridge\Release\SokuRLBridge.dll'
if(-not (Test-Path -LiteralPath $built)){throw 'Previously verified main Bridge build unavailable'}
Import-Module (Join-Path $repo 'platform/tools/BridgeDeployment.psm1') -Force
Push-Location $repo
try {
    Invoke-BridgeSession -GameDirectory $game -Dll $built -EvidenceDirectory $destination -Action {
    & $python -B rl/run.py validate_command_player --output "$destination/command_reward.json" --full-reward
    if($LASTEXITCODE){throw 'Native command/reward validation failed; training not started'}
    if($IdlePenaltyTest){
        & $python -B rl/run.py validate_idle_penalty --output "$destination/idle_penalty.json"
        if($LASTEXITCODE){throw 'Native idle penalty validation failed'}
    }
    if($PlatformIntegrationTest){
        & $python -B rl/run.py validate_platform_integration --workers 4 --frames 160 --output "$destination/platform_integration.json"
        if($LASTEXITCODE){throw 'Four-instance integration/reset validation failed'}
    }
    if($LearningCampaign -or $ResumeCampaign){
        if(-not $ContinueFrom){throw 'Campaign requires source training directory'}
        $campaignOutput=if($ResumeCampaign){$ResumeCampaign}else{"$destination/campaign"}
        $resumeCampaignArgs=@()
        if($ResumeCampaign){$resumeCampaignArgs+='--resume'}
        & $python -B rl/run.py run_v2_learning_campaign --source $ContinueFrom --output $campaignOutput @resumeCampaignArgs
        if($LASTEXITCODE){throw 'V2 adaptive learning campaign failed; preserve all evidence'}
    }
    if($Timesteps -gt 0){
        $resume=@()
        if($ContinueFrom){
            $source=[IO.Path]::GetFullPath((Join-Path $repo $ContinueFrom)).Replace('\','/')
            $resume=@('algorithm.initial_policy.kind=checkpoint',"+algorithm.initial_policy.path=$source/final.zip",
                      "+algorithm.initial_policy.training_config=$source/config.yaml")
        }
        & $python -B rl/run.py train --config-name $ConfigName "algorithm.timesteps=$Timesteps" "output=$($destination.Replace('\','/'))/training" @resume @TrainOverrides
        if($LASTEXITCODE){throw 'V2 training failed; preserve logs/checkpoints'}
    }
    if($EvaluateBaseline){
        foreach($entry in @(@('baseline',$EvaluateBaseline,'initial.zip'),@('candidate',$EvaluateCandidate,'final.zip'))){
            $source=[IO.Path]::GetFullPath((Join-Path $repo $entry[1])).Replace('\','/')
            & $python -B rl/run.py benchmark_br --config-name benchmark_combat_v2 "training_directory=$source" "checkpoint=$($entry[2])" "output=$($destination.Replace('\','/'))/evaluation-$($entry[0])"
            if($LASTEXITCODE){throw "V2 paired evaluation failed: $($entry[0])"}
        }
    }
    }.GetNewClosure()
} finally {
    Pop-Location
}
