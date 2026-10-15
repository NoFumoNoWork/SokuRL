function Invoke-BridgeSession {
    param(
        [Parameter(Mandatory=$true)][string]$GameDirectory,
        [Parameter(Mandatory=$true)][string]$Dll,
        [Parameter(Mandatory=$true)][string]$EvidenceDirectory,
        [Parameter(Mandatory=$true)][scriptblock]$Action
    )
    $ErrorActionPreference = 'Stop'
    if (Get-Process th123 -ErrorAction SilentlyContinue) { throw 'Existing game process; no deployment' }
    if (-not (Test-Path -LiteralPath $Dll)) { throw "Bridge build missing: $Dll" }
    if (Test-Path -LiteralPath $EvidenceDirectory) { throw 'Preserve existing evidence directory' }
    $target = Join-Path $GameDirectory 'modules\SokuRLBridge\SokuRLBridge.dll'
    $before = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
    New-Item -ItemType Directory -Path $EvidenceDirectory | Out-Null
    $backup = Join-Path $EvidenceDirectory 'original.dll'
    Copy-Item -LiteralPath $target -Destination $backup
    if ((Get-FileHash -LiteralPath $backup -Algorithm SHA256).Hash -ne $before) { throw 'Backup hash differs' }
    $record = [ordered]@{target=$target; build=[IO.Path]::GetFullPath($Dll);
        before_sha256=$before; deployed_sha256=(Get-FileHash -LiteralPath $Dll -Algorithm SHA256).Hash;
        restored=$false}
    try {
        Copy-Item -LiteralPath $Dll -Destination $target -Force
        & $Action
    } finally {
        Copy-Item -LiteralPath $backup -Destination $target -Force
        $record.restored = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $before
        $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'deployment.json') -Encoding utf8
        if (-not $record.restored) { throw 'Original DLL restore hash differs' }
    }
}
Export-ModuleMember -Function Invoke-BridgeSession
