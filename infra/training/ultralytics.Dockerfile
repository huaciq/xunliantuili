FROM train-platform/pytorch:development

ARG ULTRALYTICS_VERSION=8.4.155

USER root
RUN python -m pip install --no-cache-dir "ultralytics==${ULTRALYTICS_VERSION}" \
    && mkdir -p /opt/models \
    && cd /opt/models \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
USER train

CMD ["yolo", "checks"]
