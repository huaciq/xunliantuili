FROM train-platform/pytorch:development

USER root
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TIMEOUT=60
RUN --mount=type=cache,id=point-pip,target=/root/.cache/pip \
    python -m pip install --cache-dir /root/.cache/pip \
    --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}" \
    --no-deps --only-binary=:all: "ninja==1.11.1.2"
RUN --mount=type=cache,id=point-pip,target=/root/.cache/pip \
    python -m pip install --cache-dir /root/.cache/pip \
    --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}" \
    "numpy==1.26.4" "Pillow==11.0.0" "tqdm==4.66.5" \
    "scikit-learn==1.5.2" "opencv-python-headless==4.10.0.84" \
    "onnx==1.16.2" "onnxruntime==1.19.2"
RUN python -m pip --no-cache-dir check
RUN python -c "import torch, torchvision, cv2, onnx, onnxruntime, sklearn; assert torch.__version__.split('+')[0] == '2.5.1'; assert torchvision.__version__.split('+')[0] == '0.20.1'"

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER train
