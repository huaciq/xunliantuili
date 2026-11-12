from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from shutil import copyfile
from typing import Literal

from fastapi import HTTPException

from app.config import GLH_CONVERTER_COMMAND, settings
from app.models import Project

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PARTITIONS = ("train", "val")


def nb_conversion_status(project: Project) -> Literal["ready", "unconfigured"] | None:
    projects = {value.strip() for value in settings.nb_conversion_projects.split(",")}
    if project.id not in projects and project.name not in projects:
        return None
    command = tuple(settings.nb_converter_command)
    if not command:
        return "unconfigured"
    if command == GLH_CONVERTER_COMMAND:
        entrypoint = Path(settings.nb_converter_workdir) / "convert.sh"
        if not entrypoint.is_file():
            return "unconfigured"
    return "ready"


def conversion_task_root() -> Path:
    workdir = Path(settings.nb_converter_workdir).resolve()
    tasks = workdir / "converter" / "tasks"
    tasks.mkdir(exist_ok=True)
    return tasks.resolve()


def stage_calibration_dataset(source: Path, workspace: Path) -> Path:
    point_root = source / "dataset200_point"
    if point_root.is_dir() and not point_root.is_symlink():
        source = point_root

    dataset = workspace / "dataset"
    split: dict[str, list[str]] = {}
    for partition in PARTITIONS:
        image_dir = source / partition / "images"
        images = []
        if image_dir.is_dir() and image_dir.resolve().is_relative_to(source):
            images = sorted(
                path
                for path in image_dir.iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
                and not path.is_symlink()
                and path.is_file()
            )
        if not images:
            raise HTTPException(
                status_code=422,
                detail=f"校准数据集的 {partition}/images 中没有可用图片",
            )
        target = dataset / partition / "images"
        target.mkdir(parents=True)
        split[partition] = []
        for image in images:
            staged = target / image.name
            copyfile(image, staged)
            split[partition].append(str(staged))

    (workspace / "training_split.json").write_text(
        json.dumps(split, ensure_ascii=False), encoding="utf-8"
    )
    return dataset


def _stop_converter(process: subprocess.Popen) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def convert_to_nb(source: Path, dataset: Path, output_dir: Path) -> Path:
    output = output_dir / "result.nb"
    argv = [
        arg.replace("{input}", str(source))
        .replace("{dataset}", str(dataset))
        .replace("{output}", str(output))
        .replace("{output_dir}", str(output_dir))
        for arg in settings.nb_converter_command
    ]
    log_path = source.parent / "converter.log"
    try:
        with log_path.open("wb") as log, subprocess.Popen(
            argv,
            cwd=settings.nb_converter_workdir,
            env={**os.environ, "TMPDIR": str(source.parent)},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        ) as process:
            try:
                exit_code = process.wait(timeout=settings.nb_conversion_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _stop_converter(process)
                raise HTTPException(status_code=504, detail="模型转换超时，请重试") from exc
    except OSError as exc:
        logger.exception("Could not start the host NB converter")
        raise HTTPException(status_code=503, detail="无法启动 NB 转换器，请联系管理员") from exc

    if exit_code:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 4096))
            diagnostic = log.read(4096).decode("utf-8", errors="replace").strip()
        logger.warning("NB converter exited with code %d: %s", exit_code, diagnostic)
        raise HTTPException(status_code=422, detail="模型转换失败，请联系管理员检查转换日志")

    if output_dir.is_symlink() or output.is_symlink() or not output.is_file():
        raise HTTPException(status_code=422, detail="转换器未生成有效的 NB 文件")
    size = output.stat().st_size
    if not size:
        raise HTTPException(status_code=422, detail="转换器未生成有效的 NB 文件")
    if size > settings.nb_conversion_max_bytes:
        raise HTTPException(status_code=422, detail="转换后的 NB 文件超过大小限制")
    return output
