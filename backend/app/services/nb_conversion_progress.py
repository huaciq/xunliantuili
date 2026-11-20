"""Observe existing converter logs without changing the conversion recipe."""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any
from uuid import UUID

from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)
PUBLIC_FIELDS = ("status", "percent", "stage", "message", "updated_at")
_file_lock = RLock()


def _root() -> Path:
    tasks = Path(settings.nb_converter_workdir).resolve() / "converter" / "tasks"
    tasks.mkdir(exist_ok=True)
    root = tasks / ".progress"
    root.mkdir(exist_ok=True)
    return root


def _clean_finished(root: Path) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for path in root.glob("*.json"):
        try:
            with _file_lock:
                data = json.loads(path.read_text(encoding="utf-8"))
                finished = data.get("status") in ("succeeded", "failed")
                if finished and datetime.fromisoformat(data["updated_at"]) < cutoff:
                    path.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, TypeError):
            continue


class ConversionProgress:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    def update(self, percent: int, stage: str, message: str, *, status: str = "running") -> None:
        if self.data["status"] in ("succeeded", "failed"):
            return
        limit = 100 if status == "succeeded" else 99
        self.data.update(
            status=status, percent=min(limit, max(self.data["percent"], percent)),
            stage=stage, message=message, updated_at=datetime.now(timezone.utc).isoformat(),
        )
        staged = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False,
            ) as file:
                staged = Path(file.name)
                json.dump(self.data, file, ensure_ascii=False)
            with _file_lock:
                staged.replace(self.path)
        except OSError:
            # Progress reporting must never turn a working conversion into a failure.
            logger.warning("Cannot save NB conversion progress", exc_info=True)
        finally:
            if staged is not None:
                with suppress(OSError):
                    staged.unlink(missing_ok=True)

    def succeed(self) -> None:
        self.update(100, "completed", "转换完成，可以下载 NB", status="succeeded")

    def fail(self, message: str) -> None:
        self.update(self.data["percent"], self.data["stage"], message, status="failed")


def start_progress(conversion_id: UUID, project_id: str, user_id: str) -> ConversionProgress | None:
    try:
        root = _root()
        _clean_finished(root)
    except OSError:
        logger.warning(
            "Cannot open NB conversion progress storage; continuing conversion", exc_info=True,
        )
        return None
    path = None
    created = False
    try:
        path = root / f"{conversion_id}.json"
        now = datetime.now(timezone.utc).isoformat()
        data = dict(project_id=project_id, user_id=user_id, status="running", percent=5,
                    stage="preparing_dataset", message="准备校准数据", updated_at=now)
        with _file_lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False)
        return ConversionProgress(path, data)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="此转换进度编号已使用，请重新发起转换") from exc
    except OSError:
        if created and path is not None:
            with suppress(OSError):
                path.unlink(missing_ok=True)
        logger.warning("Cannot create NB conversion progress; continuing conversion", exc_info=True)
        return None


def get_progress(conversion_id: UUID, project_id: str, user_id: str) -> dict[str, Any]:
    try:
        with _file_lock:
            data = json.loads((_root() / f"{conversion_id}.json").read_text(encoding="utf-8"))
        if data["project_id"] != project_id or data["user_id"] != user_id:
            raise ValueError("Progress belongs to another request")
        return {key: data[key] for key in PUBLIC_FIELDS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="找不到该转换进度",
                            headers={"Cache-Control": "no-store"}) from exc


class ConversionLogProgress:
    MARKER = re.compile(r"SDK(?: native)? stage:\s*(import|quantize|inference|export)\b")

    def __init__(self, path: Path, progress: ConversionProgress):
        self.path = path
        self.progress = progress
        self.offset = 0
        self.pending = ""
        self.inferences = 0

    def refresh(self) -> None:
        try:
            with self.path.open("rb") as file:
                file.seek(self.offset)
                content = file.read(256 * 1024)
                self.offset = file.tell()
        except OSError:
            return
        lines = (self.pending + content.decode("utf-8", errors="replace")).splitlines(keepends=True)
        self.pending = ""
        for line in lines:
            if not line.endswith(("\n", "\r")):
                self.pending = line[-4096:]
                continue
            for match in self.MARKER.finditer(line):
                stage = match[1]
                if stage == "import":
                    self.progress.update(25, "import", "导入 ONNX 模型")
                elif stage == "quantize":
                    self.progress.update(40, "quantize", "量化校准中")
                elif stage == "inference":
                    self.inferences += 1
                    if self.inferences == 1:
                        self.progress.update(75, "inference_float", "执行浮点试运行")
                    else:
                        self.progress.update(82, "inference_quantized", "执行量化试运行")
                elif stage == "export":
                    self.progress.update(90, "export", "生成 NB 文件")
