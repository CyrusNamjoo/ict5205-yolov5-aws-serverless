# ICT5205 Assessment 2 - deploy the whole YOLOv5 detection stack with CloudFormation.
# Author: Sirous (Cyrus) Namjoo - student 240344
#
# Requirements: AWS CLI v2 configured with an admin user, Python 3 (for the test),
#               the container image already pushed to ECR (build_and_push.sh).
#
# Usage (PowerShell, from the deploy folder):
#   .\deploy.ps1 -Suffix 240344-cfn -ImageUri 743204764250.dkr.ecr.ap-southeast-2.amazonaws.com/yolo-lambda:v2
#
# What it does:
#   1. creates/updates the CloudFormation stack (S3 x2, DynamoDB + auto scaling,
#      IAM role, Lambda, API Gateway with API key + usage plan)
#   2. uploads 10 COCO128 sample images to the input bucket
#   3. warms the Lambda up with one direct invoke, so the first API call does not
#      hit API Gateway's 29 s timeout while Lambda caches the new image
#   4. prints the API URL + key and runs the functional test

param(
    [Parameter(Mandatory = $true)][string]$Suffix,
    [Parameter(Mandatory = $true)][string]$ImageUri,
    [string]$Region = "ap-southeast-2",
    [string]$ImagesDir = "..\coco128\coco128\images\train2017"
)
$ErrorActionPreference = "Continue"   # rely on $LASTEXITCODE for AWS CLI errors
$stack = "yolo-stack-$Suffix"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ">> 1/4 Deploying CloudFormation stack $stack ..." -ForegroundColor Cyan
aws cloudformation deploy `
    --region $Region `
    --stack-name $stack `
    --template-file "$here\template.yaml" `
    --capabilities CAPABILITY_NAMED_IAM `
    --parameter-overrides "Suffix=$Suffix" "ImageUri=$ImageUri"
if ($LASTEXITCODE -ne 0) { throw "CloudFormation deploy failed - see the stack Events tab in the console" }

$outputs = aws cloudformation describe-stacks --region $Region --stack-name $stack `
    --query "Stacks[0].Outputs" --output json | ConvertFrom-Json
function Get-Out($name) { ($outputs | Where-Object { $_.OutputKey -eq $name }).OutputValue }
$apiUrl   = Get-Out "ApiUrl"
$keyId    = Get-Out "ApiKeyId"
$inBucket = Get-Out "InputBucketName"
$function = Get-Out "FunctionName"

Write-Host ">> 2/4 Uploading sample images to s3://$inBucket/images/ ..." -ForegroundColor Cyan
$samples = "000000000025", "000000000034", "000000000072", "000000000081", "000000000110",
           "000000000113", "000000000165", "000000000247", "000000000471", "000000000532"
$imgPath = Join-Path $here $ImagesDir
if (Test-Path $imgPath) {
    foreach ($s in $samples) {
        aws s3 cp (Join-Path $imgPath "$s.jpg") "s3://$inBucket/images/$s.jpg" --region $Region --only-show-errors
    }
} else {
    Write-Warning "COCO128 not found at $imgPath. Download https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip, unzip it, and upload images to s3://$inBucket/images/"
}

Write-Host ">> 3/4 Warming up $function (first start after a deploy can take ~50 s) ..." -ForegroundColor Cyan
$payload = Join-Path $env:TEMP "yolo-warmup.json"
'{"key": "images/000000000081.jpg"}' | Out-File -Encoding ascii $payload
$response = Join-Path $env:TEMP "yolo-warmup-response.json"
aws lambda invoke --region $Region --function-name $function `
    --cli-binary-format raw-in-base64-out --payload "fileb://$payload" `
    --cli-read-timeout 120 $response | Out-Null
Get-Content $response | Select-Object -First 1

Write-Host ">> 4/4 Testing the API ..." -ForegroundColor Cyan
$env:API_URL = $apiUrl
$env:API_KEY = aws apigateway get-api-key --region $Region --api-key $keyId --include-value --query value --output text
Start-Sleep -Seconds 20   # new API keys / usage plans take a few seconds to become active
python "$here\..\code\tests\api_test.py" functional

Write-Host ""
Write-Host "API URL : $apiUrl" -ForegroundColor Green
Write-Host "API key : (stored in `$env:API_KEY for this window)" -ForegroundColor Green
Write-Host "Remove everything later with:  .\destroy.ps1 -Suffix $Suffix" -ForegroundColor Yellow
