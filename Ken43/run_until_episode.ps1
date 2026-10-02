param(
    [int]$TargetEpisode = 1500,
    [int]$StartEpisode = 0,
    [string]$RunDirectory = "Ken43\checkpoints\corner_drive_v2\long_run_seed7",
    [string]$Bot = "mixed",
    [int]$Seed = 7
)

$ErrorActionPreference = "Stop"

if ($TargetEpisode -le $StartEpisode) {
    throw "TargetEpisode must be greater than StartEpisode."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\python.exe"
$trainer = Join-Path $PSScriptRoot "train_fixed_bot.py"
$runPath = Join-Path $repoRoot $RunDirectory
$resume = Join-Path $runPath "ppo_${Bot}_episode_${StartEpisode}.zip"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if ($StartEpisode -gt 0 -and -not (Test-Path -LiteralPath $resume)) {
    throw "Resume checkpoint not found: $resume"
}

$remainingEpisodes = $TargetEpisode - $StartEpisode
$timestepCeiling = $remainingEpisodes * 600

$arguments = @(
    $trainer,
    "--bot", $Bot,
    "--episodes", $TargetEpisode,
    "--episode-offset", $StartEpisode,
    "--timesteps", $timestepCeiling,
    "--seed", $Seed,
    "--timeout-penalty", "0.5",
    "--device", "cuda",
    "--num-envs", "4",
    "--checkpoint-every-episodes", "500",
    "--output", $runPath
)
if ($StartEpisode -gt 0) {
    $arguments += @("--resume", $resume)
}

& $python @arguments

if ($LASTEXITCODE -ne 0) {
    throw "Training exited with code $LASTEXITCODE."
}

$targetCheckpoint = Join-Path $runPath "ppo_${Bot}_episode_${TargetEpisode}.zip"
if (-not (Test-Path -LiteralPath $targetCheckpoint)) {
    throw "Training ended without the target checkpoint: $targetCheckpoint"
}

Write-Output "Completed episode ${TargetEpisode}: $targetCheckpoint"
