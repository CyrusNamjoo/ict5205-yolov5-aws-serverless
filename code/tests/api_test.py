"""
ICT5205 Assessment 2 - API Gateway test & load-test client
Author: Sirous (Cyrus) Namjoo - student 240344

Uses only the Python standard library (no pip install needed).

Usage (PowerShell):
    $env:API_URL = "https://<api-id>.execute-api.ap-southeast-2.amazonaws.com/prod/detect"
    $env:API_KEY = "<your api key>"

    python api_test.py functional                      # correctness + security tests
    python api_test.py load --requests 20 --concurrency 5
    python api_test.py burst --requests 40 --concurrency 20   # exceed concurrency limits
    python api_test.py burst --requests 40 --concurrency 20 --retries 5   # same, client retries with backoff
    python api_test.py coldstart --rounds 3 --label v2-lazy   # force + measure cold starts (needs AWS CLI)
    python api_test.py memory --sizes 1024 2048 3008          # memory vs speed vs cost (needs AWS CLI)

Every run also writes a CSV (results_<mode>_<time>.csv) for the report.
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

IMAGES = [
    "images/000000000025.jpg", "images/000000000034.jpg", "images/000000000072.jpg",
    "images/000000000081.jpg", "images/000000000110.jpg", "images/000000000113.jpg",
    "images/000000000165.jpg", "images/000000000247.jpg", "images/000000000471.jpg",
    "images/000000000532.jpg",
]


def call_api(url, api_key, body, send_key=True, timeout=60):
    """POST to the API and return (status, latency_ms, parsed_json)."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if send_key and api_key:
        req.add_header("x-api-key", api_key)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, text = resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # timeout, DNS, connection reset...
        status, text = 0, json.dumps({"client_error": repr(e)})
    latency = (time.perf_counter() - start) * 1000
    try:
        payload = json.loads(text) if text else {}
    except json.JSONDecodeError:
        payload = {"raw": text[:200]}
    return status, round(latency, 1), payload


RETRYABLE = {0, 429, 500, 502, 503, 504}


def call_with_retry(url, api_key, body, retries):
    """Retry 'busy' responses with exponential backoff + jitter (AWS best practice)."""
    total = 0.0
    for attempt in range(retries + 1):
        status, ms, payload = call_api(url, api_key, body)
        total += ms
        if status not in RETRYABLE or attempt == retries:
            return status, round(total, 1), payload, attempt
        wait = min(8.0, 0.5 * (2 ** attempt)) * random.uniform(0.5, 1.0)
        time.sleep(wait)
        total += wait * 1000
    return status, round(total, 1), payload, attempt


def short(payload):
    if not isinstance(payload, dict):
        return str(payload)[:80]
    for k in ("error", "message", "client_error", "raw"):
        if k in payload:
            return str(payload[k])[:80]
    if "label_counts" in payload:
        m = payload.get("metrics", {})
        return f'{payload["label_counts"]} cold={m.get("cold_start")} inf={m.get("inference_ms")}ms'
    return str(payload)[:80]


def save_csv(mode, rows):
    name = f"results_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {name}")


# ---------------------------------------------------------------- AWS CLI helpers
FUNCTION = os.environ.get("FUNCTION_NAME", "yolo-detector")
PRICE_GB_S = 0.0000166667      # Lambda x86 price per GB-second (on-demand)
PRICE_REQ = 0.20 / 1_000_000   # price per request


def aws(*args):
    out = subprocess.run(["aws", *args], capture_output=True, text=True, shell=(os.name == "nt"))
    if out.returncode != 0:
        sys.exit(f"AWS CLI failed: aws {' '.join(args)}\n{out.stderr}")
    return out.stdout.strip()


def force_cold_start(**config):
    """Any configuration change makes Lambda start fresh execution environments."""
    extra = []
    if "memory" in config:
        extra = ["--memory-size", str(config["memory"])]
    aws("lambda", "update-function-configuration", "--function-name", FUNCTION,
        "--description", f"perf-test {datetime.now().strftime('%H:%M:%S')}", *extra)
    aws("lambda", "wait", "function-updated", "--function-name", FUNCTION)


def metrics_of(payload):
    return payload.get("metrics", {}) if isinstance(payload, dict) else {}


# ---------------------------------------------------------------- cold starts
def coldstart(url, key, rounds, label):
    rows = []
    print(f"Measuring {rounds} forced cold starts ({label}) on {FUNCTION} ...\n")
    print(f"{'ROUND':5} {'TYPE':5} {'STATUS':>6} {'CLIENT_MS':>10} {'INIT_MS':>8} {'MODEL_LOAD_MS':>13} "
          f"{'INFER_MS':>9} {'LAMBDA_MS':>10} MODE")
    for r in range(1, rounds + 1):
        force_cold_start()
        for kind in ("cold", "warm"):
            status, ms, payload = call_api(url, key, {"key": "images/000000000081.jpg"})
            m = metrics_of(payload)
            row = {"label": label, "round": r, "type": kind, "status": status, "client_ms": ms,
                   "cold_start": m.get("cold_start"), "init_ms": m.get("init_ms"),
                   "model_load_ms": m.get("model_load_ms"), "inference_ms": m.get("inference_ms"),
                   "lambda_total_ms": m.get("total_ms"), "load_mode": m.get("load_mode", "eager(v1)"),
                   "version": m.get("version", "v1")}
            rows.append(row)
            print(f"{r:<5} {kind:5} {status:>6} {ms:>10.0f} {str(row['init_ms']):>8} "
                  f"{str(row['model_load_ms']):>13} {str(row['inference_ms']):>9} "
                  f"{str(row['lambda_total_ms']):>10} {row['load_mode']}")
    cold = [r["client_ms"] for r in rows if r["type"] == "cold" and r["status"] == 200]
    warm = [r["client_ms"] for r in rows if r["type"] == "warm" and r["status"] == 200]
    print("\n================ SUMMARY ================")
    if cold:
        print(f"Cold start, client latency : mean {statistics.mean(cold):.0f} ms  (min {min(cold):.0f}, max {max(cold):.0f})")
    if warm:
        print(f"Warm request, client latency: mean {statistics.mean(warm):.0f} ms")
    print("Tip: check CloudWatch for INIT_REPORT ... Status: timeout lines during this test.")
    save_csv(f"coldstart_{label}", rows)


# ---------------------------------------------------------------- memory tuning
def memory(url, key, sizes, warm_calls):
    rows, summary = [], []
    original = aws("lambda", "get-function-configuration", "--function-name", FUNCTION,
                   "--query", "MemorySize", "--output", "text")
    print(f"Memory tuning on {FUNCTION} (currently {original} MB): sizes {sizes}, {warm_calls} warm calls each\n")
    for size in sizes:
        force_cold_start(memory=size)
        status, cold_ms, payload = call_api(url, key, {"key": "images/000000000081.jpg"})
        warm = []
        for i in range(warm_calls):
            img = IMAGES[i % len(IMAGES)]
            st, ms, pl = call_api(url, key, {"key": img})
            m = metrics_of(pl)
            if st == 200:
                warm.append((ms, m.get("inference_ms"), m.get("total_ms")))
            rows.append({"memory_mb": size, "call": i + 1, "image": img, "status": st, "client_ms": ms,
                         "inference_ms": m.get("inference_ms"), "lambda_total_ms": m.get("total_ms")})
        if not warm:
            print(f"{size} MB: no successful warm calls")
            continue
        client = statistics.mean(w[0] for w in warm)
        infer = statistics.mean(w[1] for w in warm)
        handler = statistics.mean(w[2] for w in warm)
        cost_1k = 1000 * ((size / 1024) * (handler / 1000) * PRICE_GB_S + PRICE_REQ)
        summary.append((size, cold_ms, client, infer, handler, cost_1k))
        print(f"{size:>5} MB | cold {cold_ms:>7.0f} ms | warm client {client:>6.0f} ms | "
              f"inference {infer:>6.0f} ms | handler {handler:>6.0f} ms | ${cost_1k:.4f} per 1000 requests")
    print("\n================ SUMMARY ================")
    print(f"{'MEMORY':>7} {'COLD_MS':>8} {'WARM_MS':>8} {'INFER_MS':>9} {'HANDLER_MS':>11} {'USD/1000':>9}")
    for s_ in summary:
        print(f"{s_[0]:>5}MB {s_[1]:>8.0f} {s_[2]:>8.0f} {s_[3]:>9.0f} {s_[4]:>11.0f} {s_[5]:>9.4f}")
    print(f"\nRestoring memory to {original} MB ...")
    force_cold_start(memory=int(original))
    save_csv("memory", rows)


# ---------------------------------------------------------------- functional
def functional(url, key):
    tests = [
        # name, body, send_api_key, expected_status
        ("valid request",              {"key": "images/000000000110.jpg"}, True, 200),
        ("no API key",                 {"key": "images/000000000110.jpg"}, False, 403),
        ("wrong API key",              {"key": "images/000000000110.jpg"}, "WRONG", 403),
        ("missing 'key' field",        {"image": "x.jpg"}, True, 400),
        ("body is not JSON",           b"this is not json", True, 400),
        ("image does not exist",       {"key": "images/nope.jpg"}, True, 404),
        ("unsupported file type",      {"key": "images/notes.txt"}, True, 400),
        ("bucket not allowed",         {"bucket": "someone-else", "key": "images/000000000081.jpg"}, True, 400),
    ]
    rows, passed = [], 0
    print(f"{'TEST':28} {'EXPECT':>6} {'GOT':>4} {'RESULT':6} {'MS':>8}  DETAIL")
    print("-" * 110)
    for name, body, send, expected in tests:
        k = "WRONG-KEY-123" if send == "WRONG" else key
        status, ms, payload = call_api(url, k, body, send_key=bool(send))
        ok = status == expected
        passed += ok
        print(f"{name:28} {expected:>6} {status:>4} {'PASS' if ok else 'FAIL':6} {ms:>8}  {short(payload)}")
        rows.append({"test": name, "expected": expected, "status": status,
                     "result": "PASS" if ok else "FAIL", "latency_ms": ms, "detail": short(payload)})
    print(f"\n{passed}/{len(tests)} tests passed")
    save_csv("functional", rows)


# ---------------------------------------------------------------- load / burst
def load(url, key, n, concurrency, mode, retries=0):
    jobs = [IMAGES[i % len(IMAGES)] for i in range(n)]
    rows = []
    print(f"Sending {n} requests with concurrency {concurrency}, client retries {retries} ...\n")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_with_retry, url, key, {"key": img}, retries): (i, img)
                   for i, img in enumerate(jobs)}
        for fut in as_completed(futures):
            i, img = futures[fut]
            status, ms, payload, retried = fut.result()
            m = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            rows.append({
                "n": i + 1, "image": img, "status": status, "latency_ms": ms, "retries": retried,
                "cold_start": m.get("cold_start"), "init_ms": m.get("init_ms"),
                "inference_ms": m.get("inference_ms"), "lambda_total_ms": m.get("total_ms"),
                "objects": payload.get("object_count") if isinstance(payload, dict) else None,
                "detail": short(payload),
            })
            print(f"#{i+1:<3} {status:>3} {ms:>9.1f} ms  retries={retried}  {img:28} {short(payload)}")
    wall = time.perf_counter() - t0

    rows.sort(key=lambda r: r["n"])
    ok = [r for r in rows if r["status"] == 200]
    lat = sorted(r["latency_ms"] for r in ok)
    print("\n================ SUMMARY ================")
    print(f"Requests        : {n}  (concurrency {concurrency})")
    print(f"Status codes    : {dict(Counter(r['status'] for r in rows))}")
    print(f"Client retries  : {sum(r['retries'] for r in rows)} total "
          f"({sum(1 for r in rows if r['retries'])} requests needed at least one retry)")
    print(f"Wall time       : {wall:.1f} s   Throughput: {len(ok)/wall:.2f} successful req/s")
    if lat:
        def pct(p):
            return lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))]
        print(f"Latency (200s)  : min {lat[0]:.0f} | p50 {pct(50):.0f} | p90 {pct(90):.0f} | "
              f"p99 {pct(99):.0f} | max {lat[-1]:.0f} | mean {statistics.mean(lat):.0f} ms")
        cold = [r for r in ok if r["cold_start"]]
        warm = [r for r in ok if r["cold_start"] is False]
        print(f"Cold starts     : {len(cold)} of {len(ok)} successful requests")
        if cold:
            print(f"  cold latency  : mean {statistics.mean(r['latency_ms'] for r in cold):.0f} ms")
        if warm:
            print(f"  warm latency  : mean {statistics.mean(r['latency_ms'] for r in warm):.0f} ms, "
                  f"inference mean {statistics.mean(r['inference_ms'] for r in warm):.0f} ms")
    save_csv(mode, rows)


def main():
    p = argparse.ArgumentParser(description="Test the YOLOv5 detection API")
    p.add_argument("mode", choices=["functional", "load", "burst", "coldstart", "memory"])
    p.add_argument("--url", default=os.environ.get("API_URL"))
    p.add_argument("--key", default=os.environ.get("API_KEY"))
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--retries", type=int, default=0, help="client retries on 429/5xx with backoff")
    p.add_argument("--rounds", type=int, default=3, help="coldstart: number of forced cold starts")
    p.add_argument("--label", default="test", help="coldstart: label for the CSV, e.g. v2-lazy")
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 3008], help="memory: sizes in MB")
    p.add_argument("--warm-calls", type=int, default=10, help="memory: warm calls per size")
    a = p.parse_args()
    if not a.url or not a.key:
        sys.exit("Set API_URL and API_KEY (environment variables or --url/--key)")
    if a.mode == "functional":
        functional(a.url, a.key)
    elif a.mode == "coldstart":
        coldstart(a.url, a.key, a.rounds, a.label)
    elif a.mode == "memory":
        memory(a.url, a.key, a.sizes, a.warm_calls)
    else:
        load(a.url, a.key, a.requests, a.concurrency, a.mode, a.retries)


if __name__ == "__main__":
    main()
