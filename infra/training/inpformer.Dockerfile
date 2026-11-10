FROM train-platform/pytorch:development

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TIMEOUT=60
# Ninja 1.11.1.1 ships malformed wheel tags that fail pip check.
RUN --mount=type=cache,id=inpformer-pip,target=/root/.cache/pip \
    python -m pip install --cache-dir /root/.cache/pip \
    --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}" \
    --no-deps --only-binary=:all: "ninja==1.11.1.2"

RUN --mount=type=cache,id=inpformer-pip,target=/root/.cache/pip \
    python -m pip install --cache-dir /root/.cache/pip \
    --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}" \
    "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.3" \
    "scikit-learn==1.5.2" "scikit-image==0.24.0" "matplotlib==3.9.2" \
    "opencv-python==4.10.0.84" "tabulate==0.9.0" "tqdm==4.66.5" \
    "timm==0.9.12" "kornia==0.7.3" "adeval==1.1.0"

RUN python -m pip --no-cache-dir check
RUN python -c "import torch, torchvision, cv2, timm, kornia; from adeval import EvalAccumulatorCuda"

ENV MPLCONFIGDIR=/workspace/output/.cache/matplotlib \
    PYTHONDONTWRITEBYTECODE=1

USER train
