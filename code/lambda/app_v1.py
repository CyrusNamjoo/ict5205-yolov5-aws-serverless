"""
ICT5205 Assessment 2 - YOLOv5 object identification on AWS Lambda
Author: Sirous Namjoo - student 240344

Flow:
  API Gateway (POST /detect, JSON body {"key": "images/x.jpg"})
    -> this Lambda (container image)
    -> read image from the INPUT S3 bucket
    -> run YOLOv5s (PyTorch, CPU)
    -> write annotated image + full JSON result to the OUTPUT S3 bucket
    -> write a summary item to DynamoDB
    -> return the detections to the caller

Best practices applied:
  * Stateless: nothing is kept between requests except the read-only model
    and AWS clients, which are created once per container (outside the
    handler) so warm invocations reuse them.
  * Input validation: only the configured input bucket and image file
    types are accepted; size is checked before download.
  * Error handling: clear HTTP status codes (400/404/413/500/502).
  * Retries: boto3 standard retry mode + an application-level retry with
    exponential backoff for every write; permanent errors fail fast.
  * Logging: one JSON log line per event, so CloudWatch Logs Insights can
    query fields such as cold_start, inference_ms and total_ms.
"""

import base64
import zlib
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from pathlib import PurePosixPath

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# --------------------------------------------------------------------------
# Configuration (Lambda environment variables)
# --------------------------------------------------------------------------
INPUT_BUCKET = os.environ["INPUT_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
TABLE_NAME = os.environ["TABLE_NAME"]
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MODEL_DIR = os.environ.get("MODEL_DIR", "/var/task/yolov5")
MODEL_PATH = os.environ.get("MODEL_PATH", "/var/task/yolov5s.pt")
MAX_WRITE_ATTEMPTS = int(os.environ.get("MAX_WRITE_ATTEMPTS", "3"))
MAX_DDB_DETECTIONS = 100  # keep DynamoDB items small; full list is in S3

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Errors that will never succeed on retry -> fail fast
PERMANENT_ERRORS = {
    "AccessDenied", "AccessDeniedException", "NoSuchBucket",
    "ResourceNotFoundException", "ValidationException", "InvalidBucketName",
}

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def log(level, message, **fields):
    """Structured (JSON) log line."""
    logger.log(level, json.dumps({"message": message, **fields}, default=str))


# --------------------------------------------------------------------------
# Initialisation - runs ONCE per container (cold start), reused when warm
# --------------------------------------------------------------------------
_init_start = time.time()

_boto_cfg = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=5,
    read_timeout=30,
)
s3 = boto3.client("s3", config=_boto_cfg)
table = boto3.resource("dynamodb", config=_boto_cfg).Table(TABLE_NAME)


def _load_model():
    import torch  # imported here so the (large) import is timed with the model

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    m = torch.hub.load(MODEL_DIR, "custom", path=MODEL_PATH, source="local", verbose=False)
    m.conf = CONF_THRESHOLD
    m.max_det = 300
    m.eval()
    return m, torch


MODEL, torch = _load_model()
INIT_MS = round((time.time() - _init_start) * 1000)
_is_cold = True
log(logging.INFO, "container initialised", init_ms=INIT_MS, cpu_count=os.cpu_count())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def respond(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=str),
    }


def parse_request(event):
    """Accept an API Gateway proxy event or a direct invocation payload."""
    event = event or {}
    if "body" in event or "httpMethod" in event or "requestContext" in event:
        raw = event.get("body") or ""
        if event.get("isBase64Encoded") and raw:
            raw = base64.b64decode(raw).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raise ValueError("Request body must be valid JSON")
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object")
        query = event.get("queryStringParameters") or {}
        data = {**query, **data}
    else:
        data = event

    # optional "s3_uri": "s3://bucket/key"
    if data.get("s3_uri"):
        uri = str(data["s3_uri"])
        if not uri.startswith("s3://") or "/" not in uri[5:]:
            raise ValueError("s3_uri must look like s3://bucket/path/image.jpg")
        data["bucket"], data["key"] = uri[5:].split("/", 1)

    bucket = str(data.get("bucket") or INPUT_BUCKET).strip()
    key = str(data.get("key") or "").strip()

    if not key:
        raise ValueError("Missing required field 'key', e.g. {\"key\": \"images/000000000081.jpg\"}")
    if bucket != INPUT_BUCKET:
        raise ValueError(f"Bucket '{bucket}' is not allowed; use '{INPUT_BUCKET}'")
    if ".." in key or key.startswith("/"):
        raise ValueError("Invalid object key")
    if PurePosixPath(key).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type; allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return bucket, key


class NotFound(Exception):
    pass


class TooLarge(Exception):
    pass


def load_image(bucket, key):
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            raise NotFound(f"s3://{bucket}/{key} does not exist")
        raise
    size = head["ContentLength"]
    if size > MAX_IMAGE_BYTES:
        raise TooLarge(f"Image is {size} bytes; limit is {MAX_IMAGE_BYTES}")
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    try:
        img = Image.open(BytesIO(data))
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("Object is not a readable image")
    return img, size


def to_detections(results):
    names = MODEL.names
    out = []
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].tolist():
        out.append({
            "label": names[int(cls)],
            "confidence": round(float(conf), 4),
            "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
        })
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out


_PALETTE = ["#FF3838", "#2C99A8", "#FF701F", "#6473FF", "#CFD231", "#48F90A",
            "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#FF9D97", "#00C2FF"]


def draw_boxes(img, detections):
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default(size=max(12, img.width // 45))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    width = max(2, img.width // 250)
    for d in detections:
        color = _PALETTE[zlib.crc32(d["label"].encode()) % len(_PALETTE)]
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f'{d["label"]} {d["confidence"]:.2f}'
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
        h = ty2 - ty1 + 4
        top = y1 - h if y1 - h > 0 else y1
        draw.rectangle([x1, top, x1 + (tx2 - tx1) + 6, top + h], fill=color)
        draw.text((x1 + 3, top + 1), text, fill="white", font=font)
    buf = BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_with_retry(description, fn):
    """Application-level retry with exponential backoff (on top of boto3's)."""
    for attempt in range(1, MAX_WRITE_ATTEMPTS + 1):
        try:
            fn()
            if attempt > 1:
                log(logging.INFO, "write succeeded after retry", target=description, attempt=attempt)
            return None
        except (ClientError, BotoCoreError) as e:
            code = e.response["Error"]["Code"] if isinstance(e, ClientError) else type(e).__name__
            permanent = code in PERMANENT_ERRORS
            log(logging.WARNING, "write failed", target=description, attempt=attempt,
                error_code=code, error=str(e), permanent=permanent)
            if permanent or attempt == MAX_WRITE_ATTEMPTS:
                log(logging.ERROR, "giving up on write", target=description, error_code=code)
                return f"{description}: {code}"
            time.sleep(0.2 * (2 ** attempt))  # 0.4s, 0.8s, ...
    return f"{description}: unknown error"


def to_dynamo(value):
    """DynamoDB needs Decimal instead of float."""
    return json.loads(json.dumps(value), parse_float=Decimal)


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------
def handler(event, context):
    global _is_cold
    cold_start, _is_cold = _is_cold, False
    started = time.time()
    request_id = getattr(context, "aws_request_id", "local")

    # 1. validate input
    try:
        bucket, key = parse_request(event)
    except ValueError as e:
        log(logging.WARNING, "bad request", request_id=request_id, error=str(e))
        return respond(400, {"error": str(e), "request_id": request_id})

    log(logging.INFO, "request received", request_id=request_id, bucket=bucket,
        key=key, cold_start=cold_start)

    # 2. read image from S3
    try:
        img, size_bytes = load_image(bucket, key)
    except NotFound as e:
        return respond(404, {"error": str(e), "request_id": request_id})
    except TooLarge as e:
        return respond(413, {"error": str(e), "request_id": request_id})
    except ValueError as e:
        return respond(400, {"error": str(e), "request_id": request_id})
    except (ClientError, BotoCoreError) as e:
        log(logging.ERROR, "S3 read failed", request_id=request_id, error=str(e))
        return respond(502, {"error": "Could not read image from S3", "request_id": request_id})

    # 3. inference
    try:
        t0 = time.time()
        with torch.inference_mode():
            results = MODEL(img, size=640)
        inference_ms = round((time.time() - t0) * 1000, 1)
        detections = to_detections(results)
        annotated_jpg = draw_boxes(img, detections)
    except Exception as e:  # model errors must not crash the container silently
        log(logging.ERROR, "inference failed", request_id=request_id, error=repr(e))
        return respond(500, {"error": "Inference failed", "request_id": request_id})

    # 4. build outputs
    now = datetime.now(timezone.utc)
    processed_at = now.isoformat(timespec="milliseconds")
    stem = PurePosixPath(key).stem
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{request_id[:8]}"
    json_key = f"results/{stem}/{run_id}.json"
    image_key = f"annotated/{stem}/{run_id}.jpg"
    label_counts = dict(Counter(d["label"] for d in detections))

    result = {
        "request_id": request_id,
        "source": f"s3://{bucket}/{key}",
        "image_size": {"width": img.width, "height": img.height, "bytes": size_bytes},
        "model": "yolov5s",
        "conf_threshold": CONF_THRESHOLD,
        "object_count": len(detections),
        "label_counts": label_counts,
        "detections": detections,
        "annotated_image": f"s3://{OUTPUT_BUCKET}/{image_key}",
        "result_json": f"s3://{OUTPUT_BUCKET}/{json_key}",
        "processed_at": processed_at,
        "metrics": {
            "cold_start": cold_start,
            "init_ms": INIT_MS if cold_start else 0,
            "inference_ms": inference_ms,
        },
    }

    # 5. persist (S3 + DynamoDB) with retries
    errors = []
    err = write_with_retry("s3-annotated-image", lambda: s3.put_object(
        Bucket=OUTPUT_BUCKET, Key=image_key, Body=annotated_jpg, ContentType="image/jpeg"))
    if err:
        errors.append(err)

    result["metrics"]["total_ms"] = round((time.time() - started) * 1000, 1)
    err = write_with_retry("s3-result-json", lambda: s3.put_object(
        Bucket=OUTPUT_BUCKET, Key=json_key, ContentType="application/json",
        Body=json.dumps(result, indent=2).encode("utf-8")))
    if err:
        errors.append(err)

    item = to_dynamo({
        "image_key": key,
        "processed_at": processed_at,
        "request_id": request_id,
        "source_bucket": bucket,
        "object_count": len(detections),
        "label_counts": label_counts,
        "detections": detections[:MAX_DDB_DETECTIONS],
        "annotated_image": result["annotated_image"],
        "result_json": result["result_json"],
        "inference_ms": inference_ms,
        "total_ms": result["metrics"]["total_ms"],
        "cold_start": cold_start,
        "model": "yolov5s",
    })
    err = write_with_retry("dynamodb-item", lambda: table.put_item(Item=item))
    if err:
        errors.append(err)

    result["metrics"]["total_ms"] = round((time.time() - started) * 1000, 1)
    log(logging.INFO, "request completed", request_id=request_id, key=key,
        object_count=len(detections), cold_start=cold_start, inference_ms=inference_ms,
        total_ms=result["metrics"]["total_ms"], storage_errors=errors)

    if errors:
        result["storage_errors"] = errors
        return respond(502, result)
    return respond(200, result)
