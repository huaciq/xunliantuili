FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG TRAIN_UID=10001
ARG TRAIN_GID=10001

RUN groupadd --gid ${TRAIN_GID} train \
    && useradd --uid ${TRAIN_UID} --gid ${TRAIN_GID} --create-home train \
    && mkdir -p /workspace/code /workspace/dataset /workspace/output /workspace/config \
    && chown -R train:train /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/workspace/output \
    XDG_CACHE_HOME=/workspace/output/.cache

WORKDIR /workspace/code
USER train

CMD ["python", "--version"]
