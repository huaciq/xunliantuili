FROM train-platform/pytorch:development

ARG ULTRALYTICS_VERSION=8.4.155
ARG YOLO11N_SHA256=0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "ultralytics==${ULTRALYTICS_VERSION}" \
    && mkdir -p /opt/models /opt/ultralytics/Ultralytics \
    && install -m 0666 /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /opt/ultralytics/Ultralytics/Arial.ttf \
    && install -m 0666 /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /opt/ultralytics/Ultralytics/Arial.Unicode.ttf \
    && chmod 0777 /opt/ultralytics /opt/ultralytics/Ultralytics

ENV YOLO_CONFIG_DIR=/opt/ultralytics

# Keep image builds independent from GitHub access. Download this file before
# building; see LINUX_DEPLOYMENT.md for the transfer and verification steps.
COPY infra/training/assets/yolo11n.pt /opt/models/yolo11n.pt
RUN echo "${YOLO11N_SHA256}  /opt/models/yolo11n.pt" | sha256sum --check --strict \
    && chmod 0644 /opt/models/yolo11n.pt \
    && python -c "from ultralytics import YOLO; YOLO('/opt/models/yolo11n.pt'); print('yolo11n.pt loaded successfully')" \
    && rm -rf /tmp/Ultralytics \
    && chmod -R a+rwX /opt/ultralytics
USER train

RUN python -c "from pathlib import Path; from ultralytics.utils import USER_CONFIG_DIR; p = Path(USER_CONFIG_DIR); print(f'Ultralytics config path: {p}'); probe = p / '.write-test'; probe.write_text('ok'); probe.unlink(); assert (p / 'Arial.ttf').is_file(); print('Ultralytics config and font are ready')"

CMD ["yolo", "checks"]
