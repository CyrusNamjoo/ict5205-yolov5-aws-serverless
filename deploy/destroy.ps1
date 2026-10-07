# ICT5205 Assessment 2 - delete a stack created by deploy.ps1 (empties the buckets first).
# Usage:  .\destroy.ps1 -Suffix 240344-cfn
param(
    [Parameter(Mandatory = $true)][string]$Suffix,
    [string]$Region = "ap-southeast-2"
)
$ErrorActionPreference = "Continue"   # AWS CLI writes warnings to stderr; don't abort on them
$stack = "yolo-stack-$Suffix"

foreach ($bucket in "yolo-input-$Suffix", "yolo-output-$Suffix") {
    Write-Host ">> Emptying $bucket (all object versions) ..." -ForegroundColor Cyan
    $raw = aws s3api list-object-versions --region $Region --bucket $bucket --output json
    if ($LASTEXITCODE -ne 0 -or -not $raw) { Write-Host "   (bucket not found or already empty)"; continue }
    $list = $raw | ConvertFrom-Json
    foreach ($v in @($list.Versions) + @($list.DeleteMarkers)) {
        if ($null -ne $v) {
            aws s3api delete-object --region $Region --bucket $bucket --key $v.Key --version-id $v.VersionId | Out-Null
        }
    }
}

Write-Host ">> Deleting stack $stack ..." -ForegroundColor Cyan
aws cloudformation delete-stack --region $Region --stack-name $stack
aws cloudformation wait stack-delete-complete --region $Region --stack-name $stack
Write-Host "Stack $stack deleted." -ForegroundColor Green
