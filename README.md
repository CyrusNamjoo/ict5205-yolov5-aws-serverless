# Serverless YOLOv5 Object Detection on AWS

**ICT5205 Cloud Computing – Assessment 2** · Sirous Namjoo · Student 240344

An image object identification service built from AWS managed services. A client sends the S3 location of an image to a REST API; a Lambda function running **YOLOv5s** detects the objects, stores an annotated copy and a JSON result in S3, writes a summary record to DynamoDB, and returns the detections.

```
Client ──HTTPS + x-api-key──▶ API Gateway (REST, /prod/detect)
                               │  API key + usage plan · body validation · throttling
                               ▼
                         AWS Lambda  (container image from ECR, YOLOv5s + PyTorch CPU, 3008 MB)
                          │        │                          │
           GetObject ◀────┘        └──▶ PutObject             └──▶ PutItem
     S3 yolo-input-240344        S3 yolo-output-240344         DynamoDB yolo-detections
       (raw images)          (annotated/*.jpg, results/*.json)   (auto scaling 1–10 RCU/WCU)

Build path:  EC2 (Ubuntu, YOLOv5 install + test, docker build) ──push──▶ Amazon ECR
Monitoring:  CloudWatch Logs (structured JSON) + metrics
```

## Live endpoint

`POST https://dqepoan9na.execute-api.ap-southeast-2.amazonaws.com/prod/detect`

```json
{ "key": "images/000000000081.jpg" }
```
Header `x-api-key` is required (the key is supplied in the report, not in this repository).

Response (shortened):
```json
{
  "object_count": 1,
  "label_counts": { "airplane": 1 },
  "detections": [ { "label": "airplane", "confidence": 0.8187, "bbox_xyxy": [24.0, 30.3, 623.5, 363.5] } ],
  "annotated_image": "s3://yolo-output-240344/annotated/000000000081/<run>.jpg",
  "result_json": "s3://yolo-output-240344/results/000000000081/<run>.json",
  "metrics": { "cold_start": false, "inference_ms": 185.0, "total_ms": 301.0 }
}
```

## Repository layout

| Path | Contents |
|---|---|
| `code/lambda/app.py` | Lambda handler (final version v2): validation, S3 read, YOLOv5 inference, box drawing, S3 + DynamoDB writes with retry, structured logging |
| `code/lambda/app_v1.py` | First version, kept for comparison |
| `code/lambda/Dockerfile` | Container image: Lambda Python 3.12 base, CPU-only PyTorch, headless OpenCV, YOLOv5 + weights baked in, pre-built font cache |
| `code/tests/api_test.py` | Test client (standard library only): `functional`, `load`, `burst` (with `--retries`), `coldstart`, `memory` |
| `code/tests/results_*.csv` | Raw results of every test run |
| `deploy/template.yaml` | CloudFormation template for the complete stack |
| `deploy/build_and_push.sh` | Build the image and push it to ECR (run on Linux/EC2 with Docker) |
| `deploy/deploy.ps1` / `destroy.ps1` | Deploy or remove the stack, upload sample images, warm up, run tests |
| `images/` | Screenshots used in the report |
| `readme.txt` | Name, student number, short description, links |

## Deploy your own copy

```bash
# 1. once, as an admin user
aws ecr create-repository --repository-name yolo-lambda --image-scanning-configuration scanOnPush=true
# 2. on a Linux x86_64 machine with Docker
./deploy/build_and_push.sh <account-id> v2 ap-southeast-2
```
```powershell
# 3. on Windows with the AWS CLI
cd deploy
.\deploy.ps1 -Suffix <unique-suffix> -ImageUri <account-id>.dkr.ecr.ap-southeast-2.amazonaws.com/yolo-lambda:v2
# 4. remove everything
.\destroy.ps1 -Suffix <unique-suffix>
```

## Key results (measured)

| Measurement | Result |
|---|---|
| YOLOv5s inference, EC2 m7i-flex.large (CPU) | ~115 ms / image |
| Lambda 3008 MB, warm request (end-to-end via API) | ~0.4 s (inference ~185 ms) |
| Lambda cold start, image cached | ~7 s |
| First call after a new image deploy | 504 at API Gateway's 29 s limit (Lambda ran 49.6 s) → warm-up step added |
| Memory 1024 / 2048 / 3008 MB (warm) | 891 / 504 / 398 ms · $0.0132 / 0.0139 / 0.0149 per 1000 requests |
| Burst 40 requests @ 20 concurrent | 10 OK + 30 throttled (account limit 10) → 40/40 OK with client retry + backoff |
| Functional/security tests | 8/8 pass (403 without key, 400 invalid body, 404 missing image, …) |

## Dataset & references

- COCO128 (first 128 images of COCO 2017 train): <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip>
- YOLOv5 by Ultralytics: <https://github.com/ultralytics/yolov5>
- AWS Lambda container images: <https://docs.aws.amazon.com/lambda/latest/dg/images-create.html>
- API Gateway usage plans and API keys: <https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html>
