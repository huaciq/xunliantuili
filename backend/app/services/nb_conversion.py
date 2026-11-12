from __future__ import annotations

import logging
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from app.config import settings
from app.models import Project

logger = logging.getLogger(__name__)


def nb_conversion_status(project: Project) -> Literal["ready", "unconfigured"] | None:
    projects = {value.strip() for value in settings.nb_conversion_projects.split(",")}
    if project.id not in projects and project.name not in projects:
        return None
    return "ready" if settings.nb_converter_command else "unconfigured"


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


def convert_to_nb(source: Path, output_dir: Path) -> Path:
    output = output_dir / "model.nb"
    argv = [
        arg.replace("{input}", str(source))
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
