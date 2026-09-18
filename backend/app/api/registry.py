import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.projects import ensure_project_access, load_project
from app.config import settings
from app.deps import CurrentUser, DbSession
from app.models import (
    AuditLog,
    ExportJob,
    ExportStatus,
    MetricPoint,
    ModelVersion,
    ProjectRole,
    RegisteredModel,
    RunArtifact,
    SystemRole,
    TrainingRun,
    TrainingRunStatus,
    User,
    utc_now,
)
from app.schemas import (
    ExportCreate,
    ExportJobView,
    MetricPointView,
    ModelRegisterRequest,
    ModelVersionView,
    RegisteredModelView,
    RunArtifactView,
)

router = APIRouter(tags=["实验与模型"])

MODEL_OPTIONS = (
    selectinload(RegisteredModel.versions).selectinload(ModelVersion.source_run),
    selectinload(RegisteredModel.versions).selectinload(ModelVersion.source_artifact),
    selectinload(RegisteredModel.versions).selectinload(ModelVersion.exports),
)


def _ensure_write_access(db: DbSession, project_id: str, user: User) -> None:
    project = load_project(db, project_id)
    role = ensure_project_access(db, project, user)
    if user.system_role != SystemRole.ADMIN and role not in (
        ProjectRole.OWNER,
        ProjectRole.RESEARCHER,
    ):
        raise HTTPException(status_code=403, detail="Write access required")


def _load_run(db: DbSession, run_id: str, user: User) -> TrainingRun:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    ensure_project_access(db, load_project(db, run.project_id), user)
    return run


def _load_model(db: DbSession, model_id: str, user: User) -> RegisteredModel:
    model = db.scalar(
        select(RegisteredModel).where(RegisteredModel.id == model_id).options(*MODEL_OPTIONS)
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Registered model not found")
    ensure_project_access(db, load_project(db, model.project_id), user)
    return model


def _model_view(model: RegisteredModel) -> RegisteredModelView:
    return RegisteredModelView(
        id=model.id,
        project_id=model.project_id,
        name=model.name,
        description=model.description,
        created_by=model.created_by,
        created_at=model.created_at,
        versions=[
            ModelVersionView(
                id=version.id,
                version=version.version,
                source_run_id=version.source_run_id,
                source_run_name=version.source_run.name,
                source_artifact_id=version.source_artifact_id,
                source_artifact_name=version.source_artifact.name,
                stage=version.stage,
                format=version.format,
                metrics=version.metrics,
                created_at=version.created_at,
                exports=version.exports,
            )
            for version in model.versions
        ],
    )


@router.get("/runs/{run_id}/metrics", response_model=list[MetricPointView])
def list_run_metrics(run_id: str, db: DbSession, user: CurrentUser) -> list[MetricPoint]:
    _load_run(db, run_id, user)
    return list(
        db.scalars(
            select(MetricPoint)
            .where(MetricPoint.run_id == run_id)
            .order_by(MetricPoint.key, MetricPoint.step)
        ).all()
    )


@router.get("/runs/{run_id}/artifacts", response_model=list[RunArtifactView])
def list_run_artifacts(run_id: str, db: DbSession, user: CurrentUser) -> list[RunArtifact]:
    _load_run(db, run_id, user)
    return list(
        db.scalars(
            select(RunArtifact)
            .where(RunArtifact.run_id == run_id)
            .order_by(RunArtifact.created_at)
        ).all()
    )


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, db: DbSession, user: CurrentUser) -> FileResponse:
    artifact = db.get(RunArtifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    _load_run(db, artifact.run_id, user)
    path = Path(artifact.uri).resolve()
    storage_root = Path(settings.storage_root).resolve()
    if not path.is_relative_to(storage_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file is unavailable")
    return FileResponse(path, filename=artifact.name, media_type="application/octet-stream")


@router.post(
    "/runs/{run_id}/register-model",
    response_model=RegisteredModelView,
    status_code=status.HTTP_201_CREATED,
)
def register_model(
    run_id: str,
    payload: ModelRegisterRequest,
    db: DbSession,
    user: CurrentUser,
) -> RegisteredModelView:
    run = _load_run(db, run_id, user)
    _ensure_write_access(db, run.project_id, user)
    if run.status != TrainingRunStatus.SUCCEEDED:
        raise HTTPException(status_code=409, detail="Only succeeded runs can be registered")

    artifact_statement = select(RunArtifact).where(
        RunArtifact.run_id == run.id, RunArtifact.artifact_type == "checkpoint"
    )
    if payload.source_artifact_id:
        artifact_statement = artifact_statement.where(RunArtifact.id == payload.source_artifact_id)
    artifact = db.scalar(artifact_statement.order_by(RunArtifact.created_at.desc()))
    if artifact is None:
        raise HTTPException(status_code=409, detail="The run has no checkpoint artifact")

    name = payload.name.strip()
    model = db.scalar(
        select(RegisteredModel).where(
            RegisteredModel.project_id == run.project_id, RegisteredModel.name == name
        )
    )
    if model is None:
        model = RegisteredModel(
            project_id=run.project_id,
            name=name,
            description=payload.description.strip(),
            created_by=user.id,
        )
        db.add(model)
        db.flush()
    latest_version = db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.model_id == model.id)
    )
    metric_rows = db.scalars(
        select(MetricPoint)
        .where(MetricPoint.run_id == run.id)
        .order_by(MetricPoint.key, MetricPoint.step.desc())
    ).all()
    latest_metrics: dict[str, float] = {}
    for metric in metric_rows:
        latest_metrics.setdefault(metric.key, metric.value)
    version = ModelVersion(
        model_id=model.id,
        version=(latest_version or 0) + 1,
        source_run_id=run.id,
        source_artifact_id=artifact.id,
        stage=payload.stage,
        format=str(artifact.details.get("format", "pytorch")),
        metrics=latest_metrics,
        created_by=user.id,
    )
    db.add(version)
    db.add(
        AuditLog(
            actor_id=user.id,
            action="model.register",
            resource_type="registered_model",
            resource_id=model.id,
            detail=f"{name} v{version.version}",
        )
    )
    db.commit()
    return _model_view(_load_model(db, model.id, user))


@router.get("/projects/{project_id}/models", response_model=list[RegisteredModelView])
def list_models(project_id: str, db: DbSession, user: CurrentUser) -> list[RegisteredModelView]:
    ensure_project_access(db, load_project(db, project_id), user)
    models = db.scalars(
        select(RegisteredModel)
        .where(RegisteredModel.project_id == project_id)
        .options(*MODEL_OPTIONS)
        .order_by(RegisteredModel.created_at.desc())
    ).unique().all()
    return [_model_view(model) for model in models]


@router.get("/models/{model_id}", response_model=RegisteredModelView)
def get_model(model_id: str, db: DbSession, user: CurrentUser) -> RegisteredModelView:
    return _model_view(_load_model(db, model_id, user))


@router.post(
    "/model-versions/{version_id}/exports",
    response_model=ExportJobView,
    status_code=status.HTTP_201_CREATED,
)
def create_export(
    version_id: str,
    payload: ExportCreate,
    db: DbSession,
    user: CurrentUser,
) -> ExportJob:
    version = db.scalar(
        select(ModelVersion)
        .where(ModelVersion.id == version_id)
        .options(selectinload(ModelVersion.model), selectinload(ModelVersion.source_run))
    )
    if version is None:
        raise HTTPException(status_code=404, detail="Model version not found")
    _ensure_write_access(db, version.model.project_id, user)

    job = ExportJob(
        model_version_id=version.id,
        format=payload.format,
        status=ExportStatus.PENDING,
        created_by=user.id,
    )
    db.add(job)
    db.flush()
    output = (
        Path(settings.storage_root).resolve()
        / "exports"
        / version.model_id
        / f"v{version.version}"
        / job.id
        / "model.onnx"
    )
    content = (
        f"FAKE_ONNX\nmodel={version.model.name}\nversion={version.version}\n"
        f"source_run={version.source_run_id}\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    artifact = RunArtifact(
        run_id=version.source_run_id,
        artifact_type="export",
        name="model.onnx",
        uri=str(output),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        details={"format": "onnx", "development_artifact": True, "opset": 17},
    )
    db.add(artifact)
    db.flush()
    job.artifact_id = artifact.id
    job.status = ExportStatus.SUCCEEDED
    job.validation_status = "passed"
    job.max_abs_diff = 0.00012
    job.finished_at = utc_now()
    db.add(
        AuditLog(
            actor_id=user.id,
            action="model.export",
            resource_type="model_version",
            resource_id=version.id,
            detail="onnx opset=17 validation=passed",
        )
    )
    db.commit()
    db.refresh(job)
    return job
