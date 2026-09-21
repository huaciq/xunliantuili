FROM train-platform/pytorch:development

ARG ULTRALYTICS_VERSION=8.4.155

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "ultralytics==${ULTRALYTICS_VERSION}" \
    && mkdir -p /opt/models

# Keep image builds independent from GitHub access. Download this file before
# building; see LINUX_DEPLOYMENT.md for the transfer and verification steps.
COPY infra/training/assets/yolo11n.pt /opt/models/yolo11n.pt
RUN chmod 0644 /opt/models/yolo11n.pt \
    && python -c "from ultralytics import YOLO; YOLO('/opt/models/yolo11n.pt'); print('yolo11n.pt loaded successfully')"
USER train

CMD ["yolo", "checks"]
