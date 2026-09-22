from __future__ import annotations

import asyncio
import shlex
from pathlib import Path, PurePosixPath

import jwt
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.projects import ensure_project_access, load_project
from app.config import settings
from app.database import SessionLocal
from app.deps import CurrentUser, DbSession
from app.models import (
    AuditLog,
    CodePackage,
    CodeVersion,
    Dataset,
    DatasetVersion,
    GpuAllocation,
    GpuDevice,
    ProjectRole,
    ResourceVersionStatus,
    RunEvent,
    RuntimeImage,
    SystemRole,
    TrainingRun,
    TrainingRunStatus,
    TrainingTemplate,
    User,
    utc_now,
)
from app.schemas import (
    GpuDeviceView,
    RunLogsView,
    RuntimeImageView,
    TrainingRunCreate,
    TrainingRunView,
    TrainingTemplateView,
)
from app.security import decode_access_token
from app.services.scheduler import _resolve_storage_path, scheduler

router = APIRouter(tags=["训练任务"])

RUN_LOAD_OPTIONS = (
    selectinload(TrainingRun.template),
    selectinload(TrainingRun.runtime_image),
    selectinload(TrainingRun.dataset_version).selectinload(DatasetVersion.dataset),
    selectinload(TrainingRun.code_version).selectinload(CodeVersion.code_package),
    selectinload(TrainingRun.events),
    selectinload(TrainingRun.gpu_allocations).selectinload(GpuAllocation.gpu_device),
)


def _ensure_write_access(db: DbSession, project_id: str, user: User) -> None:
    project = load_project(db, project_id)
    role = ensure_project_access(db, project, user)
    if user.system_role != SystemRole.ADMIN and role not in (
        ProjectRole.OWNER,
        ProjectRole.RESEARCHER,
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Write access required")


def _load_run(db: DbSession, run_id: str) -> TrainingRun:
    run = db.scalar(
        select(TrainingRun)
        .where(TrainingRun.id == run_id)
        .options(*RUN_LOAD_OPTIONS)
        .execution_options(populate_existing=True)
    )
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training run not found")
    return run


def _run_view(run: TrainingRun) -> TrainingRunView:
    return TrainingRunView(
        id=run.id,
        project_id=run.project_id,
        name=run.name,
        template_id=run.template_id,
        template_name=run.template.name,
        dataset_version_id=run.dataset_version_id,
        dataset_label=f"{run.dataset_version.dataset.name} v{run.dataset_version.version}",
        code_version_id=run.code_version_id,
        code_label=f"{run.code_version.code_package.name} v{run.code_version.version}",
        runtime_image_id=run.runtime_image_id,
        runtime_image_name=run.runtime_image.name,
        status=run.status,
        parameters=run.parameters,
        command=run.command,
        requested_gpu_count=run.requested_gpu_count,
        requested_gpu_model=run.requested_gpu_model,
        allocated_gpus=[item.gpu_device.uuid for item in run.gpu_allocations],
        priority=run.priority,
        execution_id=run.execution_id,
        exit_code=run.exit_code,
        failure_reason=run.failure_reason,
        created_by=run.created_by,
        created_at=run.created_at,
        queued_at=run.queued_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        events=run.events,
    )


def _find_yolo_dataset_config(dataset_root: Path) -> PurePosixPath:
    candidates = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name.lower() in {"data.yaml", "data.yml"}
    )
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="The selected dataset does not contain data.yaml or data.yml",
        )
    if len(candidates) > 1:
        relative_paths = [path.relative_to(dataset_root).as_posix() for path in candidates[:10]]
        raise HTTPException(
            status_code=422,
            detail=(
                "The selected dataset contains multiple YOLO configuration files: "
                + ", ".join(relative_paths)
            ),
        )
    relative_path = candidates[0].relative_to(dataset_root)
    return PurePosixPath("/workspace/dataset") / PurePosixPath(relative_path.as_posix())


def _build_command(
    template: TrainingTemplate,
    code: CodeVersion,
    dataset: DatasetVersion,
    parameters: dict,
) -> list[str]:
    if template.key == "yolo_detection":
        storage_root = Path(settings.storage_root).resolve()
        dataset_root = _resolve_storage_path(storage_root, dataset.cache_uri)
        data_config = _find_yolo_dataset_config(dataset_root)
        values = {
            "epochs": int(parameters.get("epochs", 100)),
            "batch": int(parameters.get("batch", 16)),
            "imgsz": int(parameters.get("imgsz", 640)),
            "model": str(parameters.get("model", "yolo11n.pt")),
        }
        if values["model"] != "yolo11n.pt":
            raise HTTPException(
                status_code=422, detail="Only the preloaded yolo11n.pt is supported"
            )
        if values["epochs"] < 1 or values["batch"] < 1 or values["imgsz"] < 32:
            raise HTTPException(status_code=422, detail="Invalid YOLO training parameters")
        return [
            "yolo",
            "detect",
            "train",
            f"data={data_config}",
            "model=/opt/models/yolo11n.pt",
            f"epochs={values['epochs']}",
            f"batch={values['batch']}",
            f"imgsz={values['imgsz']}",
            "project=/workspace/output",
            "name=train",
        ]
    entrypoint = code.default_entrypoint.strip()
    if not entrypoint:
        raise HTTPException(status_code=422, detail="The code version has no default entrypoint")
    try:
        command = shlex.split(entrypoint, posix=True)
        arguments = shlex.split(str(parameters.get("arguments", "")), posix=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid command syntax: {exc}") from exc
    return command + arguments


@router.get("/training-templates", response_model=list[TrainingTemplateView])
def list_templates(db: DbSession, _: CurrentUser) -> list[TrainingTemplate]:
    return list(
        db.scalars(
            select(TrainingTemplate)
            .where(TrainingTemplate.is_active.is_(True))
            .order_by(TrainingTemplate.name)
        ).all()
    )


@router.get("/runtime-images", response_model=list[RuntimeImageView])
def list_runtime_images(db: DbSession, _: CurrentUser) -> list[RuntimeImage]:
    return list(
        db.scalars(
            select(RuntimeImage).where(RuntimeImage.is_active.is_(True)).order_by(RuntimeImage.name)
        ).all()
    )


@router.get("/resources/gpus", response_model=list[GpuDeviceView])
def list_gpus(db: DbSession, _: CurrentUser) -> list[GpuDeviceView]:
    allocations = {
        allocation.gpu_device_id: allocation.run_id
        for allocation in db.scalars(select(GpuAllocation)).all()
    }
    return [
        GpuDeviceView.model_validate(device).model_copy(
            update={"allocated_run_id": allocations.get(device.id)}
        )
        for device in db.scalars(select(GpuDevice).order_by(GpuDevice.index)).all()
    ]


@router.get("/projects/{project_id}/runs", response_model=list[TrainingRunView])
def list_runs(project_id: str, db: DbSession, user: CurrentUser) -> list[TrainingRunView]:
    project = load_project(db, project_id)
    ensure_project_access(db, project, user)
    runs = (
        db.scalars(
            select(TrainingRun)
            .where(TrainingRun.project_id == project_id)
            .options(*RUN_LOAD_OPTIONS)
            .order_by(TrainingRun.created_at.desc())
        )
        .unique()
        .all()
    )
    return [_run_view(run) for run in runs]


@router.post(
    "/projects/{project_id}/runs",
    response_model=TrainingRunView,
    status_code=status.HTTP_201_CREATED,
)
def create_run(
    project_id: str, payload: TrainingRunCreate, db: DbSession, user: CurrentUser
) -> TrainingRunView:
    _ensure_write_access(db, project_id, user)
    template = db.get(TrainingTemplate, payload.template_id)
    dataset_version = db.get(DatasetVersion, payload.dataset_version_id)
    code_version = db.get(CodeVersion, payload.code_version_id)
    if template is None or not template.is_active:
        raise HTTPException(status_code=404, detail="Training template not found")
    if dataset_version is None or dataset_version.status != ResourceVersionStatus.READY:
        raise HTTPException(status_code=422, detail="Dataset version is not ready")
    if code_version is None or code_version.status != ResourceVersionStatus.READY:
        raise HTTPException(status_code=422, detail="Code version is not ready")
    dataset = db.get(Dataset, dataset_version.dataset_id)
    code_package = db.get(CodePackage, code_version.code_package_id)
    if dataset is None or code_package is None:
        raise HTTPException(status_code=422, detail="Selected resources are invalid")
    if dataset.project_id != project_id or code_package.project_id != project_id:
        raise HTTPException(status_code=422, detail="Selected resources must belong to the project")
    runtime_id = payload.runtime_image_id or template.default_runtime_image_id
    runtime = db.get(RuntimeImage, runtime_id)
    if runtime is None or not runtime.is_active:
        raise HTTPException(status_code=422, detail="Runtime image is not available")
    if payload.requested_gpu_count == 4 and payload.requested_gpu_model == "any":
        raise HTTPException(status_code=422, detail="Four-GPU heterogeneous training is disabled")
    command = _build_command(template, code_version, dataset_version, payload.parameters)
    run = TrainingRun(
        project_id=project_id,
        name=payload.name.strip(),
        template_id=template.id,
        dataset_version_id=dataset_version.id,
        code_version_id=code_version.id,
        runtime_image_id=runtime.id,
        parameters=payload.parameters,
        command=command,
        requested_gpu_count=payload.requested_gpu_count,
        requested_gpu_model=payload.requested_gpu_model,
        priority=payload.priority,
        created_by=user.id,
    )
    db.add(run)
    db.flush()
    db.add(RunEvent(run_id=run.id, event_type="status", message="训练任务草稿已创建"))
    db.add(
        AuditLog(
            actor_id=user.id,
            action="training_run.create",
            resource_type="training_run",
            resource_id=run.id,
            detail=run.name,
        )
    )
    db.commit()
    return _run_view(_load_run(db, run.id))


@router.get("/runs/{run_id}", response_model=TrainingRunView)
def get_run(run_id: str, db: DbSession, user: CurrentUser) -> TrainingRunView:
    run = _load_run(db, run_id)
    project = load_project(db, run.project_id)
    ensure_project_access(db, project, user)
    return _run_view(run)


@router.post("/runs/{run_id}/submit", response_model=TrainingRunView)
def submit_run(run_id: str, db: DbSession, user: CurrentUser) -> TrainingRunView:
    run = _load_run(db, run_id)
    _ensure_write_access(db, run.project_id, user)
    if run.status != TrainingRunStatus.DRAFT:
        raise HTTPException(status_code=409, detail="Only draft runs can be submitted")
    run.status = TrainingRunStatus.QUEUED
    run.queued_at = utc_now()
    db.add(RunEvent(run_id=run.id, event_type="status", message="训练任务已进入队列"))
    db.commit()
    return _run_view(_load_run(db, run.id))


@router.post("/runs/{run_id}/stop", response_model=TrainingRunView)
def stop_run(run_id: str, db: DbSession, user: CurrentUser) -> TrainingRunView:
    run = _load_run(db, run_id)
    _ensure_write_access(db, run.project_id, user)
    if run.status == TrainingRunStatus.QUEUED:
        run.status = TrainingRunStatus.STOPPED
        run.finished_at = utc_now()
        db.add(RunEvent(run_id=run.id, event_type="status", message="排队任务已取消"))
    elif run.status in {
        TrainingRunStatus.PREPARING,
        TrainingRunStatus.STARTING,
        TrainingRunStatus.RUNNING,
    }:
        run.status = TrainingRunStatus.CANCEL_REQUESTED
        db.add(RunEvent(run_id=run.id, event_type="status", message="已请求停止训练任务"))
    else:
        raise HTTPException(status_code=409, detail="Run cannot be stopped in its current state")
    db.commit()
    return _run_view(_load_run(db, run.id))


@router.get("/runs/{run_id}/logs", response_model=RunLogsView)
def get_logs(run_id: str, db: DbSession, user: CurrentUser) -> RunLogsView:
    run = _load_run(db, run_id)
    project = load_project(db, run.project_id)
    ensure_project_access(db, project, user)
    lines = scheduler.logs(run.execution_id)
    if not lines:
        lines = [f"[{event.event_type}] {event.message}" for event in run.events]
    return RunLogsView(status=run.status, lines=lines)


@router.websocket("/runs/{run_id}/logs/stream")
async def stream_logs(websocket: WebSocket, run_id: str, token: str) -> None:
    try:
        user_id = decode_access_token(token)
    except jwt.InvalidTokenError:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        last_payload = None
        while True:
            with SessionLocal() as db:
                user = db.get(User, user_id)
                run = _load_run(db, run_id)
                if user is None:
                    await websocket.close(code=4401)
                    return
                project = load_project(db, run.project_id)
                ensure_project_access(db, project, user)
                payload = RunLogsView(
                    status=run.status,
                    lines=scheduler.logs(run.execution_id)
                    or [f"[{event.event_type}] {event.message}" for event in run.events],
                ).model_dump(mode="json")
            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload
            if payload["status"] in {"succeeded", "failed", "stopped"}:
                return
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
