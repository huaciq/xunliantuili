FROM train-platform/pytorch:development

ARG ULTRALYTICS_VERSION=8.4.155

USER root
RUN python -m pip install --no-cache-dir "ultralytics==${ULTRALYTICS_VERSION}"
USER train

CMD ["yolo", "checks"]
