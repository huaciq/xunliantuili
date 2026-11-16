from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import (
    DatasetFormat,
    ExportStatus,
    GpuModelPolicy,
    ModelStage,
    ProjectRole,
    ResourceVersionStatus,
    SystemRole,
    TrainingRunStatus,
)


class LoginRequest(BaseModel):
    """用户登录请求。"""

    email: EmailStr = Field(description="登录邮箱，例如 admin@example.com")
    password: str = Field(
        min_length=8,
        max_length=128,
        description="登录密码，长度为 8 到 128 个字符",
    )


class TokenResponse(BaseModel):
    """登录成功后返回的访问令牌。"""

    access_token: str = Field(description="JWT 访问令牌，后续请求放入 Authorization 请求头")
    token_type: str = Field(default="bearer", description="令牌类型，固定为 bearer")


class UserSummary(BaseModel):
    """平台用户摘要信息。"""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="用户唯一 ID")
    email: EmailStr = Field(description="用户邮箱")
    name: str = Field(description="用户显示名称")
    system_role: SystemRole = Field(description="系统角色：user 或 system_admin")
    is_active: bool = Field(description="账号是否启用")


class ProjectCreate(BaseModel):
    """创建项目请求。"""

    name: str = Field(min_length=2, max_length=120, description="项目名称，长度为 2 到 120 个字符")
    description: str = Field(default="", max_length=2000, description="项目描述，可为空")


class ProjectMemberView(BaseModel):
    """项目成员信息。"""

    id: str = Field(description="成员关系唯一 ID")
    user_id: str = Field(description="用户唯一 ID")
    email: EmailStr = Field(description="成员邮箱")
    name: str = Field(description="成员名称")
    role: ProjectRole = Field(description="项目角色：project_owner、researcher 或 viewer")


class ProjectView(BaseModel):
    """项目详情及成员列表。"""

    id: str = Field(description="项目唯一 ID")
    name: str = Field(description="项目名称")
    description: str = Field(description="项目描述")
    created_by: str = Field(description="创建者用户 ID")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="最近更新时间")
    current_user_role: Optional[ProjectRole] = Field(
        default=None,
        description="当前用户在该项目中的角色；管理员可能没有项目成员关系",
    )
    members: list[ProjectMemberView] = Field(default_factory=list, description="项目成员列表")


class ProjectMemberAdd(BaseModel):
    """添加或更新项目成员请求。"""

    email: EmailStr = Field(description="要加入项目的用户邮箱")
    role: ProjectRole = Field(description="项目角色：researcher 或 viewer 等")


class UserCreate(BaseModel):
    """创建平台用户请求。"""

    email: EmailStr = Field(description="用户邮箱，必须是合法邮箱格式")
    name: str = Field(min_length=2, max_length=100, description="用户名称，长度为 2 到 100 个字符")
    password: str = Field(
        min_length=8,
        max_length=128,
        description="初始密码，长度为 8 到 128 个字符",
    )
    system_role: SystemRole = Field(
        default=SystemRole.USER,
        description="系统角色：user（普通用户）或 system_admin（系统管理员）",
    )


class ResourceCreate(BaseModel):
    """创建数据集或代码包请求。"""

    name: str = Field(min_length=2, max_length=120, description="资源名称，长度为 2 到 120 个字符")
    description: str = Field(default="", max_length=2000, description="资源描述，可为空")


class DatasetVersionView(BaseModel):
    """数据集版本信息。"""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="版本唯一 ID")
    dataset_id: str = Field(description="所属数据集 ID")
    version: int = Field(description="版本号，从 1 开始递增")
    status: ResourceVersionStatus = Field(description="处理状态")
    format: DatasetFormat = Field(description="识别出的数据集格式")
    root_subpath: str = Field(description="数据集在解压目录中的根路径")
    source_filename: str = Field(description="上传时的原始文件名")
    sha256: str = Field(description="压缩包 SHA-256 摘要")
    archive_size: int = Field(description="压缩包大小，单位为字节")
    extracted_size: int = Field(description="解压后大小，单位为字节")
    file_count: int = Field(description="解压后的文件数量")
    error_message: str = Field(description="处理失败时的错误信息")
    created_by: str = Field(description="创建者用户 ID")
    created_at: datetime = Field(description="创建时间")
    ready_at: Optional[datetime] = Field(default=None, description="处理完成时间")


class DatasetView(BaseModel):
    """数据集及其版本列表。"""

    id: str = Field(description="数据集唯一 ID")
    project_id: str = Field(description="所属项目 ID")
    name: str = Field(description="数据集名称")
    description: str = Field(description="数据集描述")
    created_by: str = Field(description="创建者用户 ID")
    created_at: datetime = Field(description="创建时间")
    versions: list[DatasetVersionView] = Field(default_factory=list, description="数据集版本列表")


class CodeVersionView(BaseModel):
    """代码包版本信息。"""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="版本唯一 ID")
    code_package_id: str = Field(description="所属代码包 ID")
    version: int = Field(description="版本号，从 1 开始递增")
    status: ResourceVersionStatus = Field(description="处理状态")
    source_filename: str = Field(description="上传时的原始文件名")
    sha256: str = Field(description="压缩包 SHA-256 摘要")
    archive_size: int = Field(description="压缩包大小，单位为字节")
    extracted_size: int = Field(description="解压后大小，单位为字节")
    file_count: int = Field(description="解压后的文件数量")
    default_workdir: str = Field(description="训练时默认工作目录")
    default_entrypoint: str = Field(description="默认入口命令")
    error_message: str = Field(description="处理失败时的错误信息")
    created_by: str = Field(description="创建者用户 ID")
    created_at: datetime = Field(description="创建时间")
    ready_at: Optional[datetime] = Field(default=None, description="处理完成时间")


class CodePackageView(BaseModel):
    """代码包及其版本列表。"""

    id: str = Field(description="代码包唯一 ID")
    project_id: str = Field(description="所属项目 ID")
    name: str = Field(description="代码包名称")
    description: str = Field(description="代码包描述")
    created_by: str = Field(description="创建者用户 ID")
    created_at: datetime = Field(description="创建时间")
    versions: list[CodeVersionView] = Field(default_factory=list, description="代码版本列表")


class RuntimeImageView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    image: str
    digest: str
    framework: str
    is_active: bool
    created_at: datetime


class RuntimeImageCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    image: str = Field(min_length=1, max_length=500)
    digest: str = Field(default="development", min_length=1, max_length=255)
    framework: str = Field(default="pytorch", min_length=1, max_length=80)
    is_active: bool = True


class RuntimeImageUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    image: Optional[str] = Field(default=None, min_length=1, max_length=500)
    digest: Optional[str] = Field(default=None, min_length=1, max_length=255)
    framework: Optional[str] = Field(default=None, min_length=1, max_length=80)
    is_active: Optional[bool] = None


class TrainingTemplateView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    name: str
    description: str
    default_runtime_image_id: str
    parameter_schema: dict


class GpuDeviceView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    uuid: str
    index: int
    model: GpuModelPolicy
    memory_gb: int
    is_enabled: bool
    allocated_run_id: Optional[str] = None


class TrainingRunCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    template_id: str
    dataset_version_id: str
    code_version_id: str
    runtime_image_id: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    requested_gpu_count: int = Field(default=1, ge=0, le=4)
    requested_gpu_model: GpuModelPolicy = GpuModelPolicy.ANY
    priority: int = Field(default=0, ge=-10, le=10)


class RunEventView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    event_type: str
    message: str
    created_at: datetime


class TrainingRunView(BaseModel):
    id: str
    project_id: str
    name: str
    template_id: str
    template_name: str
    dataset_version_id: str
    dataset_label: str
    code_version_id: str
    code_label: str
    runtime_image_id: str
    runtime_image_name: str
    status: TrainingRunStatus
    parameters: dict
    command: list[str]
    requested_gpu_count: int
    requested_gpu_model: GpuModelPolicy
    allocated_gpus: list[str] = Field(default_factory=list)
    priority: int
    execution_id: Optional[str]
    exit_code: Optional[int]
    failure_reason: str
    created_by: str
    created_at: datetime
    queued_at: Optional[datetime]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    events: list[RunEventView] = Field(default_factory=list)


class RunLogsView(BaseModel):
    status: TrainingRunStatus
    lines: list[str] = Field(default_factory=list)


class MetricPointView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    key: str
    step: int
    value: float
    created_at: datetime


class RunArtifactView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    artifact_type: str
    name: str
    size_bytes: int
    sha256: str
    details: dict
    created_at: datetime


class ModelRegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=2000)
    source_artifact_id: Optional[str] = None
    stage: ModelStage = ModelStage.CANDIDATE


class ExportCreate(BaseModel):
    format: str = Field(default="onnx", pattern="^onnx$")


class ExportJobView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version_id: str
    format: str
    status: ExportStatus
    artifact_id: Optional[str]
    validation_status: str
    max_abs_diff: Optional[float]
    error_message: str
    created_at: datetime
    finished_at: Optional[datetime]


class ModelVersionView(BaseModel):
    id: str
    version: int
    source_run_id: str
    source_run_name: str
    source_artifact_id: str
    source_artifact_name: str
    stage: ModelStage
    format: str
    metrics: dict
    created_at: datetime
    exports: list[ExportJobView] = Field(default_factory=list)


class RegisteredModelView(BaseModel):
    id: str
    project_id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    versions: list[ModelVersionView] = Field(default_factory=list)
