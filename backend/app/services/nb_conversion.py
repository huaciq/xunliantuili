from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException

from app.config import settings
from app.models import Project

logger = logging.getLogger(__name__)


def nb_conversion_status(project: Project) -> Literal["ready", "unconfigured"] | None:
    projects = {value.strip() for value in settings.nb_conversion_projects.split(",")}
    if project.id not in projects and project.name not in projects:
        return None
    return "ready" if settings.nb_converter_image.strip() else "unconfigured"


def _remove_container(name: str) -> None:
    try:
        subprocess.run(
            [settings.docker_binary, "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("Could not remove NB conversion container %s", name)


def convert_to_nb(source: Path, output_dir: Path) -> Path:
    name = f"train-platform-nb-{uuid4().hex}"
    argv = [
        settings.docker_binary, "run", "--rm", "--name", name, "--pull=never",
        "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
        "--init", "--log-driver=none", "--user", settings.docker_user,
        "--pids-limit", str(settings.docker_pids_limit),
        "--cpus", str(settings.docker_cpu_limit), "--memory", f"{settings.docker_memory_limit_gb}g",
        "--mount", f"type=bind,source={source},target=/input/model.onnx,readonly",
        "--mount", f"type=bind,source={output_dir},target=/output",
        settings.nb_converter_image.strip(), "/input/model.onnx", "/output/model.nb",
    ]
    if settings.docker_user.split(":", 1)[0] in {"0", "root", ""}:
        raise HTTPException(status_code=503, detail="转换容器需要配置非 root 用户")
    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=settings.nb_conversion_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _remove_container(name)
        raise HTTPException(status_code=504, detail="模型转换超时，请重试") from exc
    except OSError as exc:
        logger.exception("Could not start the NB conversion container")
        raise HTTPException(status_code=503, detail="无法启动 NB 转换器，请联系管理员") from exc

    if result.returncode:
        logger.warning("NB converter %s exited with code %d", name, result.returncode)
        if result.returncode in {125, 126, 127}:
            raise HTTPException(status_code=503, detail="无法启动 NB 转换器，请联系管理员")
        raise HTTPException(status_code=422, detail="转换失败，请检查 ONNX 与目标设备的兼容性")

    output = output_dir / "model.nb"
    size = output.stat().st_size if not output.is_symlink() and output.is_file() else 0
    if not size:
        raise HTTPException(status_code=422, detail="转换器未生成有效的 NB 文件")
    if size > settings.nb_conversion_max_bytes:
        raise HTTPException(status_code=422, detail="转换后的 NB 文件超过大小限制")
    return output
