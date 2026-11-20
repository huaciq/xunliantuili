from hashlib import file_digest
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, BinaryIO
from uuid import UUID

from fastapi import APIRouter, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from starlette.background import BackgroundTask

from app.api.projects import ensure_project_access, load_project
from app.config import settings
from app.deps import CurrentUser, DbSession
from app.models import (
    Dataset,
    DatasetVersion,
    Project,
    ProjectMember,
    ProjectRole,
    ResourceVersionStatus,
    RunArtifact,
    SystemRole,
    TrainingRun,
    TrainingRunStatus,
    User,
)
from app.services.archive_ingest import storage_root
from app.services.nb_conversion import (
    conversion_task_root,
    convert_to_nb,
    nb_conversion_status,
    stage_calibration_dataset,
)
from app.services.nb_conversion_progress import get_progress, start_progress
from app.services.scheduler import _resolve_storage_path

router = APIRouter(tags=["模型转换"])
conversion_slot = Lock()


def _find_training_split(
    db: DbSession, user: User, dataset_version_id: str | None, model_sha256: str, storage: Path
) -> tuple[DatasetVersion, Path] | None:
    matching_runs = (
        select(TrainingRun.id)
        .join(RunArtifact, RunArtifact.run_id == TrainingRun.id)
        .where(
            TrainingRun.status == TrainingRunStatus.SUCCEEDED,
            RunArtifact.artifact_type == "export",
            RunArtifact.name.ilike("%.onnx"),
            RunArtifact.sha256 == model_sha256,
        )
    )
    if dataset_version_id is not None:
        matching_runs = matching_runs.where(TrainingRun.dataset_version_id == dataset_version_id)
    if user.system_role != SystemRole.ADMIN:
        memberships = select(ProjectMember.project_id).where(ProjectMember.user_id == user.id)
        matching_runs = matching_runs.where(TrainingRun.project_id.in_(memberships))

    reports = db.execute(
        select(RunArtifact, DatasetVersion)
        .join(TrainingRun, TrainingRun.id == RunArtifact.run_id)
        .join(DatasetVersion, DatasetVersion.id == TrainingRun.dataset_version_id)
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(
            RunArtifact.run_id.in_(matching_runs),
            RunArtifact.artifact_type == "report",
            RunArtifact.name == "training_split.json",
            Dataset.project_id == TrainingRun.project_id,
            DatasetVersion.status == ResourceVersionStatus.READY,
        )
    )
    splits = {}
    for report, version in reports:
        try:
            path = _resolve_storage_path(storage, report.uri)
        except ValueError:
            continue
        if path.is_file():
            key = (version.sha256 or version.id, report.sha256 or str(path))
            splits[key] = (version, path)
    if len(splits) > 1:
        raise HTTPException(status_code=422, detail="该 ONNX 匹配了多个不同的训练划分")
    return next(iter(splits.values()), None)


@router.post(
    "/projects/{project_id}/conversions/onnx-to-nb",
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
    dataset_version_id: Annotated[
        str | None, Form(description="校准数据集版本，省略时使用匹配 ONNX 的原训练数据集")
    ] = None,
    conversion_id: Annotated[UUID | None, Header(alias="X-NB-Conversion-ID")] = None,
) -> FileResponse:
    project = load_project(db, project_id)
    _ensure_conversion_access(db, project, user)

    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    _validate_onnx(filename, file.size)

    version = None
    if dataset_version_id is not None:
        version = db.scalar(
            select(DatasetVersion).join(Dataset).where(
                DatasetVersion.id == dataset_version_id, Dataset.project_id == project_id
            )
        )
        if version is None:
            raise HTTPException(status_code=404, detail="找不到该项目的数据集版本")
        if version.status != ResourceVersionStatus.READY:
            raise HTTPException(status_code=409, detail="数据集版本尚未就绪")

    model_sha256 = file_digest(file.file, "sha256").hexdigest()
    file.file.seek(0)
    storage = storage_root()
    match = _find_training_split(db, user, dataset_version_id, model_sha256, storage)
    training_split = None
    if match is not None:
        version, training_split = match
    if version is None:
        raise HTTPException(
            status_code=422,
            detail="找不到这份 ONNX 对应的训练划分，请选择已划分训练图和验证图的数据集版本",
        )
    try:
        cache = _resolve_storage_path(storage, version.cache_uri)
        dataset_root = (
            cache if training_split else _resolve_storage_path(cache, version.root_subpath)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="数据集目录不可用") from exc
    progress_owner = (str(project.id), str(user.id))
    db.close()
    return _convert_model(
        filename, file.file, dataset_root, training_split, conversion_id, progress_owner,
    )


@router.get("/projects/{project_id}/conversions/onnx-to-nb/progress/{conversion_id}")
def conversion_progress(
    project_id: str, conversion_id: UUID, response: Response, db: DbSession, user: CurrentUser,
):
    project = load_project(db, project_id)
    _ensure_conversion_access(db, project, user)
    response.headers["Cache-Control"] = "no-store"
    return get_progress(conversion_id, str(project.id), str(user.id))


@router.post(
    "/runs/{run_id}/artifacts/{artifact_id}/conversions/onnx-to-nb",
    response_class=FileResponse,
    summary="使用训练产物转换为 NB",
)
def convert_run_onnx(
    run_id: str, artifact_id: str, db: DbSession, user: CurrentUser,
    conversion_id: Annotated[UUID | None, Header(alias="X-NB-Conversion-ID")] = None,
) -> FileResponse:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="找不到训练任务")
    project = load_project(db, run.project_id)
    _ensure_conversion_access(db, project, user)
    if run.status != TrainingRunStatus.SUCCEEDED:
        raise HTTPException(status_code=409, detail="训练成功后才能转换模型")

    artifact = db.get(RunArtifact, artifact_id)
    if artifact is None or artifact.run_id != run.id:
        raise HTTPException(status_code=404, detail="找不到该任务的模型产物")
    if (
        artifact.artifact_type != "export"
        or PurePosixPath(artifact.name).suffix.lower() != ".onnx"
        or artifact.details.get("development_artifact") is not False
    ):
        raise HTTPException(status_code=422, detail="请选择实际训练导出的 ONNX 产物")

    split = db.scalar(
        select(RunArtifact).where(
            RunArtifact.run_id == run.id,
            RunArtifact.artifact_type == "report",
            RunArtifact.name == "training_split.json",
        )
    )
    if split is None:
        raise HTTPException(status_code=422, detail="该任务缺少原训练划分")

    version = db.scalar(
        select(DatasetVersion)
        .join(Dataset)
        .where(
            DatasetVersion.id == run.dataset_version_id,
            Dataset.project_id == project.id,
            DatasetVersion.status == ResourceVersionStatus.READY,
        )
    )
    if version is None:
        raise HTTPException(status_code=422, detail="该任务的原训练数据集版本不可用")

    storage = storage_root()
    output_root = storage / "runs" / project.id / run.id / "output"
    model = _run_artifact_path(storage, output_root, artifact)
    training_split = _run_artifact_path(storage, output_root, split)
    _validate_onnx(artifact.name, model.stat().st_size)
    try:
        dataset_root = _resolve_storage_path(storage, version.cache_uri)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="数据集目录不可用") from exc
    filename = PurePosixPath(artifact.name).name
    progress_owner = (str(project.id), str(user.id))
    db.close()
    with model.open("rb") as file:
        return _convert_model(
            filename, file, dataset_root, training_split, conversion_id, progress_owner,
        )


def _run_artifact_path(storage: Path, output_root: Path, artifact: RunArtifact) -> Path:
    try:
        path = _resolve_storage_path(storage, artifact.uri)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="训练产物文件不可用") from exc
    if not path.is_relative_to(output_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="训练产物文件不可用")
    return path


def _ensure_conversion_access(db: DbSession, project: Project, user: User) -> None:
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


def _validate_onnx(filename: str, size: int | None) -> None:
    if PurePosixPath(filename).suffix.lower() != ".onnx" or not size:
        raise HTTPException(status_code=422, detail="请选择非空的 .onnx 文件")
    if size > settings.nb_conversion_max_bytes:
        limit = settings.nb_conversion_max_bytes
        raise HTTPException(status_code=413, detail=f"ONNX 文件大小超过 {limit:,} 字节限制")


def _convert_model(
    filename: str, file: BinaryIO, dataset_root: Path, training_split: Path | None,
    conversion_id: UUID | None = None, progress_owner: tuple[str, str] | None = None,
) -> FileResponse:
    if not conversion_slot.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有模型正在转换，请稍后重试")
    workspace = None
    progress = None
    try:
        if conversion_id is not None and progress_owner is not None:
            progress = start_progress(conversion_id, *progress_owner)
        try:
            workspace = TemporaryDirectory(prefix="platform-", dir=conversion_task_root())
        except OSError as exc:
            raise HTTPException(status_code=503, detail="转换任务目录不可写，请联系管理员") from exc
        root = Path(workspace.name)
        source, output_dir = root / "model.onnx", root / "output"
        output_dir.mkdir()
        dataset = stage_calibration_dataset(dataset_root, root, training_split)
        with source.open("wb") as target:
            copyfileobj(file, target)
        if progress:
            output = convert_to_nb(source, dataset, output_dir, progress=progress)
        else:
            output = convert_to_nb(source, dataset, output_dir)
        response = FileResponse(
            output,
            media_type="application/octet-stream",
            filename=f"{PurePosixPath(filename).stem[:200]}.nb",
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(workspace.cleanup),
        )
        if progress:
            progress.succeed()
        return response
    except Exception as exc:
        if progress:
            message = (
                str(exc.detail) if isinstance(exc, HTTPException)
                else "模型转换失败，请查看转换日志"
            )
            progress.fail(message)
        if workspace is not None:
            workspace.cleanup()
        raise
    finally:
        conversion_slot.release()
