FROM train-platform/pytorch:development

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.3" \
    "scikit-learn==1.5.2" "scikit-image==0.24.0" "matplotlib==3.9.2" \
    "opencv-python==4.10.0.84" "tabulate==0.9.0" "tqdm==4.66.5" \
    "timm==0.9.12" "kornia==0.7.3" "adeval==1.1.0" \
    && python -m pip check \
    && python -c "import torch, torchvision, cv2, timm, kornia; from adeval import EvalAccumulatorCuda"

ENV MPLCONFIGDIR=/workspace/output/.cache/matplotlib \
    PYTHONDONTWRITEBYTECODE=1

USER train
