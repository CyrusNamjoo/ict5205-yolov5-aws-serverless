#!/usr/bin/env bash
# ICT5205 Assessment 2 - build the YOLOv5 Lambda container image and push it to ECR.
# Author: Sirous (Cyrus) Namjoo - student 240344
#
# Run on a Linux x86_64 machine with Docker and the AWS CLI (we used the EC2
# instance yolo-ec2, Ubuntu, m7i-flex.large). The ECR repository must already
# exist (created once by an admin user):
#   aws ecr create-repository --repository-name yolo-lambda --image-scanning-configuration scanOnPush=true
#
# Usage:  ./build_and_push.sh <aws-account-id> [tag] [region]
#   e.g.  ./build_and_push.sh 743204764250 v2 ap-southeast-2
set -euo pipefail

ACCOUNT="${1:?usage: $0 <aws-account-id> [tag] [region]}"
TAG="${2:-v2}"
REGION="${3:-ap-southeast-2}"
REPO="yolo-lambda"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$(mktemp -d)"

echo ">> Preparing build context in ${BUILD}"
cp "${HERE}/../code/lambda/app.py" "${HERE}/../code/lambda/Dockerfile" "${HERE}/../code/lambda/.dockerignore" "${BUILD}/"
git clone --depth 1 https://github.com/ultralytics/yolov5.git "${BUILD}/yolov5"
curl -L -o "${BUILD}/yolov5s.pt" https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

echo ">> Building image ${REPO}:${TAG} (x86_64)"
# --provenance=false: Lambda needs a plain Docker v2 manifest, not an OCI index
docker build --provenance=false --platform linux/amd64 -t "${REPO}:${TAG}" "${BUILD}"

echo ">> Pushing to ${REGISTRY}/${REPO}:${TAG}"
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${REGISTRY}"
docker tag "${REPO}:${TAG}" "${REGISTRY}/${REPO}:${TAG}"
docker push "${REGISTRY}/${REPO}:${TAG}"

echo ">> Done. ImageUri for deploy.ps1 / template.yaml:"
echo "   ${REGISTRY}/${REPO}:${TAG}"
rm -rf "${BUILD}"
