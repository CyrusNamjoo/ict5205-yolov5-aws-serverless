# YOLOv5 Lambda container image (x86_64) - v2 (cold-start optimised)
# Build on the EC2 instance:  docker build --provenance=false -t yolo-lambda:v2 .
FROM public.ecr.aws/lambda/python:3.12

# Only /tmp is writable inside Lambda -> point every library cache there
ENV YOLOV5_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    GIT_PYTHON_REFRESH=quiet \
    PYTHONUNBUFFERED=1 \
    MPL_PREBUILT_CACHE=/opt/mplcache \
    MODEL_LOAD=lazy \
    APP_VERSION=v2

# 1) CPU-only PyTorch (no CUDA -> image is ~3x smaller, Lambda has no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) YOLOv5 dependencies; swap OpenCV for the headless build (no libGL on Lambda)
COPY yolov5/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# 3) YOLOv5 source + weights (baked in -> no downloads at runtime)
COPY yolov5/ ${LAMBDA_TASK_ROOT}/yolov5/
COPY yolov5s.pt ${LAMBDA_TASK_ROOT}/yolov5s.pt

# 4) Pre-compile Python files (the task root is read-only at runtime)
#    and check the model loads during the build, not in production
RUN python -m compileall -q ${LAMBDA_TASK_ROOT}/yolov5 && \
    python -c "import torch; m=torch.hub.load('${LAMBDA_TASK_ROOT}/yolov5','custom',path='${LAMBDA_TASK_ROOT}/yolov5s.pt',source='local'); print('MODEL OK', len(m.names), 'classes')"

# 5) v2: build the matplotlib font cache once at build time (v1 rebuilt it on
#    every cold start - "generated new fontManager" in the logs).
#    app.py copies it to /tmp at runtime because /opt is read-only in Lambda.
RUN MPLCONFIGDIR=/opt/mplcache python -c "import matplotlib.font_manager; print('font cache built')" && \
    ls /opt/mplcache

# 6) Handler
COPY app.py ${LAMBDA_TASK_ROOT}/
CMD ["app.handler"]
