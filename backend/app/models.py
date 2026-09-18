from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SystemRole(str, enum.Enum):
    ADMIN = "system_admin"
    USER = "user"


class ProjectRole(str, enum.Enum):
    OWNER = "project_owner"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class ResourceVersionStatus(str, enum.Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DatasetFormat(str, enum.Enum):
    YOLO = "yolo"
    COCO = "coco"
    CLASSIFICATION = "classification"
    GENERIC = "generic"


class TrainingRunStatus(str, enum.Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    PREPARING = "preparing"
    STARTING = "starting"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    STOPPING = "stopping"
    STOPPED = "stopped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GpuModelPolicy(str, enum.Enum):
    ANY = "any"
    RTX_3090 = "rtx_3090"
    RTX_4090 = "rtx_4090"


class ModelStage(str, enum.Enum):
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ExportStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(255))
    system_role: Mapped[SystemRole] = mapped_column(
        Enum(SystemRole, native_enum=False), default=SystemRole.USER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    memberships: Mapped[list["ProjectMember"]] = relationship(back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    members: Mapped[list["ProjectMember"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    code_packages: Mapped[list["CodePackage"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    registered_models: Mapped[list["RegisteredModel"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProjectMember(Base):
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_project_member"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[ProjectRole] = mapped_column(Enum(ProjectRole, native_enum=False))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    project: Mapped[Project] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="memberships")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), index=True)
    resource_type: Mapped[str] = mapped_column(String(50))
    resource_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_project_dataset_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    project: Mapped[Project] = relationship(back_populates="datasets")
    versions: Mapped[list["DatasetVersion"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetVersion.version"
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[ResourceVersionStatus] = mapped_column(
        Enum(ResourceVersionStatus, native_enum=False), default=ResourceVersionStatus.PROCESSING
    )
    format: Mapped[DatasetFormat] = mapped_column(
        Enum(DatasetFormat, native_enum=False), default=DatasetFormat.GENERIC
    )
    source_filename: Mapped[str] = mapped_column(String(255))
    archive_uri: Mapped[str] = mapped_column(String(1000), default="")
    cache_uri: Mapped[str] = mapped_column(String(1000), default="")
    manifest_uri: Mapped[str] = mapped_column(String(1000), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    archive_size: Mapped[int] = mapped_column(BigInteger, default=0)
    extracted_size: Mapped[int] = mapped_column(BigInteger, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    ready_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class CodePackage(Base):
    __tablename__ = "code_packages"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_project_code_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    project: Mapped[Project] = relationship(back_populates="code_packages")
    versions: Mapped[list["CodeVersion"]] = relationship(
        back_populates="code_package",
        cascade="all, delete-orphan",
        order_by="CodeVersion.version",
    )


class CodeVersion(Base):
    __tablename__ = "code_versions"
    __table_args__ = (UniqueConstraint("code_package_id", "version", name="uq_code_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    code_package_id: Mapped[str] = mapped_column(ForeignKey("code_packages.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[ResourceVersionStatus] = mapped_column(
        Enum(ResourceVersionStatus, native_enum=False), default=ResourceVersionStatus.PROCESSING
    )
    source_filename: Mapped[str] = mapped_column(String(255))
    archive_uri: Mapped[str] = mapped_column(String(1000), default="")
    cache_uri: Mapped[str] = mapped_column(String(1000), default="")
    manifest_uri: Mapped[str] = mapped_column(String(1000), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    archive_size: Mapped[int] = mapped_column(BigInteger, default=0)
    extracted_size: Mapped[int] = mapped_column(BigInteger, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    default_workdir: Mapped[str] = mapped_column(String(500), default=".")
    default_entrypoint: Mapped[str] = mapped_column(String(1000), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    ready_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    code_package: Mapped[CodePackage] = relationship(back_populates="versions")


class RuntimeImage(Base):
    __tablename__ = "runtime_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    image: Mapped[str] = mapped_column(String(500))
    digest: Mapped[str] = mapped_column(String(255), default="development")
    framework: Mapped[str] = mapped_column(String(80), default="pytorch")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingTemplate(Base):
    __tablename__ = "training_templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(80), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    default_runtime_image_id: Mapped[str] = mapped_column(ForeignKey("runtime_images.id"))
    parameter_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    default_runtime_image: Mapped[RuntimeImage] = relationship()


class GpuDevice(Base):
    __tablename__ = "gpu_devices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    uuid: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    index: Mapped[int] = mapped_column(Integer, unique=True)
    model: Mapped[GpuModelPolicy] = mapped_column(Enum(GpuModelPolicy, native_enum=False))
    memory_gb: Mapped[int] = mapped_column(Integer)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(160))
    template_id: Mapped[str] = mapped_column(ForeignKey("training_templates.id"))
    dataset_version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    code_version_id: Mapped[str] = mapped_column(ForeignKey("code_versions.id"))
    runtime_image_id: Mapped[str] = mapped_column(ForeignKey("runtime_images.id"))
    status: Mapped[TrainingRunStatus] = mapped_column(
        Enum(TrainingRunStatus, native_enum=False), default=TrainingRunStatus.DRAFT, index=True
    )
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    command: Mapped[list] = mapped_column(JSON, default=list)
    requested_gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    requested_gpu_model: Mapped[GpuModelPolicy] = mapped_column(
        Enum(GpuModelPolicy, native_enum=False), default=GpuModelPolicy.ANY
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    execution_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    failure_reason: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="training_runs")
    template: Mapped[TrainingTemplate] = relationship()
    dataset_version: Mapped[DatasetVersion] = relationship()
    code_version: Mapped[CodeVersion] = relationship()
    runtime_image: Mapped[RuntimeImage] = relationship()
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunEvent.created_at"
    )
    gpu_allocations: Mapped[list["GpuAllocation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["MetricPoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="MetricPoint.step"
    )
    artifacts: Mapped[list["RunArtifact"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunArtifact.created_at"
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="source_run")


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(50), default="status")
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[TrainingRun] = relationship(back_populates="events")


class GpuAllocation(Base):
    __tablename__ = "gpu_allocations"
    __table_args__ = (UniqueConstraint("gpu_device_id", name="uq_active_gpu_allocation"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    gpu_device_id: Mapped[str] = mapped_column(ForeignKey("gpu_devices.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[TrainingRun] = relationship(back_populates="gpu_allocations")
    gpu_device: Mapped[GpuDevice] = relationship()


class MetricPoint(Base):
    __tablename__ = "metric_points"
    __table_args__ = (
        UniqueConstraint("run_id", "key", "step", name="uq_run_metric_key_step"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(100), index=True)
    step: Mapped[int] = mapped_column(Integer)
    value: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[TrainingRun] = relationship(back_populates="metrics")


class RunArtifact(Base):
    __tablename__ = "run_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    artifact_type: Mapped[str] = mapped_column(String(50), index=True)
    name: Mapped[str] = mapped_column(String(255))
    uri: Mapped[str] = mapped_column(String(1000))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[TrainingRun] = relationship(back_populates="artifacts")
    model_versions: Mapped[list["ModelVersion"]] = relationship(back_populates="source_artifact")


class RegisteredModel(Base):
    __tablename__ = "registered_models"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_project_model_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    project: Mapped[Project] = relationship(back_populates="registered_models")
    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", order_by="ModelVersion.version"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_id: Mapped[str] = mapped_column(ForeignKey("registered_models.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    source_run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"))
    source_artifact_id: Mapped[str] = mapped_column(ForeignKey("run_artifacts.id"))
    stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, native_enum=False), default=ModelStage.CANDIDATE
    )
    format: Mapped[str] = mapped_column(String(50), default="pytorch")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    model: Mapped[RegisteredModel] = relationship(back_populates="versions")
    source_run: Mapped[TrainingRun] = relationship(back_populates="model_versions")
    source_artifact: Mapped[RunArtifact] = relationship(back_populates="model_versions")
    exports: Mapped[list["ExportJob"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )


class ExportJob(Base):
    __tablename__ = "export_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    format: Mapped[str] = mapped_column(String(50), default="onnx")
    status: Mapped[ExportStatus] = mapped_column(
        Enum(ExportStatus, native_enum=False), default=ExportStatus.PENDING
    )
    artifact_id: Mapped[Optional[str]] = mapped_column(ForeignKey("run_artifacts.id"))
    validation_status: Mapped[str] = mapped_column(String(50), default="pending")
    max_abs_diff: Mapped[Optional[float]]
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    model_version: Mapped[ModelVersion] = relationship(back_populates="exports")
    artifact: Mapped[Optional[RunArtifact]] = relationship()
