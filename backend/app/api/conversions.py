from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.api.projects import ensure_project_access, load_project
from app.config import settings
from app.deps import CurrentUser, DbSession
from app.models import ProjectRole, SystemRole
from app.services.archive_ingest import storage_root
from app.services.nb_conversion import convert_to_nb, nb_conversion_status

router = APIRouter(prefix="/projects", tags=["模型转换"])
conversion_slot = Lock()


@router.post(
    "/{project_id}/conversions/onnx-to-nb",
    response_class=FileResponse,
    responses={
        200: {
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
    summary="上传 ONNX 并转换为 NB",
)
def convert_onnx(
    project_id: str,
    db: DbSession,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="包含全部权重的单个 ONNX 文件")],
) -> FileResponse:
    project = load_project(db, project_id)
    role = ensure_project_access(db, project, user)
    if user.system_role != SystemRole.ADMIN and role not in (
        ProjectRole.OWNER, ProjectRole.RESEARCHER,
    ):
        raise HTTPException(status_code=403, detail="当前账号没有模型转换权限")

    converter_status = nb_conversion_status(project)
    if converter_status is None:
        raise HTTPException(status_code=404, detail="该项目未启用 NB 转换")
    if converter_status == "unconfigured":
        raise HTTPException(status_code=503, detail="服务器尚未配置 NB 转换器")

    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    if PurePosixPath(filename).suffix.lower() != ".onnx" or not file.size:
        raise HTTPException(status_code=422, detail="请选择非空的 .onnx 文件")
    if file.size > settings.nb_conversion_max_bytes:
        limit = settings.nb_conversion_max_bytes
        raise HTTPException(status_code=413, detail=f"ONNX 文件大小超过 {limit:,} 字节限制")

    db.close()
    if not conversion_slot.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有模型正在转换，请稍后重试")
    workspace = None
    try:
        workspace = TemporaryDirectory(prefix="nb-conversion-", dir=storage_root())
        root = Path(workspace.name)
        source, output_dir = root / "model.onnx", root / "output"
        output_dir.mkdir()
        with source.open("wb") as target:
            copyfileobj(file.file, target)
        output = convert_to_nb(source, output_dir)
        return FileResponse(
            output,
            media_type="application/octet-stream",
            filename=f"{PurePosixPath(filename).stem[:200]}.nb",
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(workspace.cleanup),
        )
    except Exception:
        if workspace is not None:
            workspace.cleanup()
        raise
    finally:
        conversion_slot.release()
