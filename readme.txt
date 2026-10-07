ICT5205 Cloud Computing - Assessment 2
Implementing an Image Object Identification System on AWS using YOLOv5, Lambda, API Gateway and Amazon S3

Name:            Sirous (Cyrus) Namjoo
Student number:  240344
Institution:     Apex Australia Higher Education - Master of Information Systems (Data Analytics)

SHORT DESCRIPTION
-----------------
A serverless object detection service on AWS (region ap-southeast-2, Sydney).
A client calls a REST API (Amazon API Gateway, protected by an API key and a usage plan)
with the S3 location of an image. API Gateway invokes an AWS Lambda function packaged as a
container image (stored in Amazon ECR) that runs the YOLOv5s model with PyTorch on CPU.
The function reads the image from the input S3 bucket, detects objects (80 COCO classes),
draws bounding boxes, and stores the annotated image and a JSON result in the output S3
bucket and a summary record in Amazon DynamoDB (with auto scaling). An EC2 instance was used
to install and verify YOLOv5 and to build the container image.

LINKS
-----
Live API (POST, header x-api-key required):
  https://dqepoan9na.execute-api.ap-southeast-2.amazonaws.com/prod/detect
Source code repository:
  https://github.com/CyrusNamjoo/ict5205-yolov5-aws-serverless
Public dataset (COCO128, 128 images from COCO 2017, 80 classes):
  https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
YOLOv5 (model and code):
  https://github.com/ultralytics/yolov5   (weights: yolov5s.pt, release v7.0)

FOLDERS
-------
code/lambda   Lambda function (app.py = final v2, app_v1.py = first version), Dockerfile
code/tests    api_test.py - functional, load, burst, cold-start and memory tests; results_*.csv
deploy        template.yaml (CloudFormation for the whole stack), build_and_push.sh,
              deploy.ps1, destroy.ps1
images        screenshots used in the report

EXAMPLE REQUEST (PowerShell)
----------------------------
$env:API_URL = "https://dqepoan9na.execute-api.ap-southeast-2.amazonaws.com/prod/detect"
$env:API_KEY = "<API key supplied in the report>"
python code\tests\api_test.py functional
