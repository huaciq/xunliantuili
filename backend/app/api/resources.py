from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.api.projects import ensure_project_access, load_project
from app.config import settings
from app.deps import CurrentUser, DbSession
from app.models import (
    AuditLog,
    CodePackage,
    CodeVersion,
    Dataset,
    DatasetVersion,
    ProjectRole,
    ResourceVersionStatus,
    SystemRole,
    utc_now,
)
from app.schemas import CodePackageView, DatasetView, ResourceCreate
from app.services.archive_ingest import ingest_archive, save_upload_to_temp, storage_root

router = APIRouter(tags=["数据集与代码"])


def _ensure_write_access(db: DbSession, project_id: str, user: CurrentUser) -> None:
    project = load_project(db, project_id)
    role = ensure_project_access(db, project, user)
    if user.system_role != SystemRole.ADMIN and role not in (
        ProjectRole.OWNER,
        ProjectRole.RESEARCHER,
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Write access required")


def _dataset_view(dataset: Dataset) -> DatasetView:
    return DatasetView(
        id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        description=dataset.description,
        created_by=dataset.created_by,
        created_at=dataset.created_at,
        versions=list(reversed(dataset.versions)),
    )


def _code_view(code_package: CodePackage) -> CodePackageView:
    return CodePackageView(
        id=code_package.id,
        project_id=code_package.project_id,
        name=code_package.name,
        description=code_package.description,
        created_by=code_package.created_by,
        created_at=code_package.created_at,
        versions=list(reversed(code_package.versions)),
    )


def _load_dataset(db: DbSession, dataset_id: str) -> Dataset:
    dataset = db.scalar(
        select(Dataset)
        .where(Dataset.id == dataset_id)
        .options(selectinload(Dataset.versions))
        .execution_options(populate_existing=True)
    )
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


def _load_code_package(db: DbSession, code_package_id: str) -> CodePackage:
    code_package = db.scalar(
        select(CodePackage)
        .where(CodePackage.id == code_package_id)
        .options(selectinload(CodePackage.versions))
        .execution_options(populate_existing=True)
    )
    if code_package is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Code package not found")
    return code_package


def _commit_resource(
    db: DbSession, resource: Dataset | CodePackage, kind: str, user_id: str
) -> None:
    db.add(resource)
    db.flush()
    db.add(
        AuditLog(
            actor_id=user_id,
            action=f"{kind}.create",
            resource_type=kind,
            resource_id=resource.id,
            detail=resource.name,
        )
    )
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A {kind.replace('_', ' ')} with this name already exists",
        ) from exc


@router.get(
    "/projects/{project_id}/datasets",
    response_model=list[DatasetView],
    summary="查询项目数据集",
    description="查询指定项目下的数据集及其版本信息。需要登录并拥有项目访问权限。",
)
def list_datasets(project_id: str, db: DbSession, user: CurrentUser) -> list[DatasetView]:
    project = load_project(db, project_id)
    ensure_project_access(db, project, user)
    datasets = db.scalars(
        select(Dataset)
        .where(Dataset.project_id == project_id)
        .options(selectinload(Dataset.versions))
        .order_by(Dataset.created_at.desc())
    ).all()
    return [_dataset_view(dataset) for dataset in datasets]


@router.post(
    "/projects/{project_id}/datasets",
    response_model=DatasetView,
    status_code=status.HTTP_201_CREATED,
    summary="创建数据集",
    description="在指定项目下创建一个数据集。项目所有者、研究人员或管理员可操作。",
)
def create_dataset(
    project_id: str, payload: ResourceCreate, db: DbSession, user: CurrentUser
) -> DatasetView:
    _ensure_write_access(db, project_id, user)
    dataset = Dataset(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description.strip(),
        created_by=user.id,
    )
    _commit_resource(db, dataset, "dataset", user.id)
    return _dataset_view(_load_dataset(db, dataset.id))


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetView,
    summary="获取数据集详情",
    description="获取数据集信息以及按最新版本优先排列的版本列表。",
)
def get_dataset(dataset_id: str, db: DbSession, user: CurrentUser) -> DatasetView:
    dataset = _load_dataset(db, dataset_id)
    project = load_project(db, dataset.project_id)
    ensure_project_access(db, project, user)
    return _dataset_view(dataset)


@router.post(
    "/datasets/{dataset_id}/versions/upload",
    response_model=DatasetView,
    summary="上传数据集版本",
    description="上传 ZIP 或 TAR.GZ 数据集压缩包，服务端会解压、扫描并生成不可变版本。",
)
async def upload_dataset_version(
    dataset_id: str,
    db: DbSession,
    user: CurrentUser,
    file: UploadFile = File(..., description="数据集压缩包，支持 ZIP 或 TAR.GZ"),
) -> DatasetView:
    dataset = _load_dataset(db, dataset_id)
    _ensure_write_access(db, dataset.project_id, user)
    temp_path, sha256, archive_size, filename = await save_upload_to_temp(file)
    next_version = (
        db.scalar(
            select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == dataset_id)
        )
        or 0
    ) + 1
    version = DatasetVersion(
        dataset_id=dataset_id,
        version=next_version,
        source_filename=filename,
        sha256=sha256,
        archive_size=archive_size,
        created_by=user.id,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    try:
        result = ingest_archive(
            temp_path=temp_path,
            destination_prefix=Path("datasets") / dataset.project_id / dataset.id / version.id,
            filename=filename,
            sha256=sha256,
            archive_size=archive_size,
            detect_dataset=True,
        )
        version.archive_uri = result.archive_uri
        version.cache_uri = result.cache_uri
        version.root_subpath = result.dataset_root_subpath
        version.manifest_uri = result.manifest_uri
        version.extracted_size = result.extracted_size
        version.file_count = result.file_count
        version.format = result.dataset_format
        version.status = ResourceVersionStatus.READY
        version.ready_at = utc_now()
    except Exception as exc:
        version.status = ResourceVersionStatus.FAILED
        version.error_message = str(exc)[:2000]
        db.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.add(
        AuditLog(
            actor_id=user.id,
            action="dataset.version.upload",
            resource_type="dataset_version",
            resource_id=version.id,
            detail=f"v{version.version}:{filename}",
        )
    )
    db.commit()
    return _dataset_view(_load_dataset(db, dataset_id))


@router.get(
    "/dataset-versions/{version_id}/manifest",
    summary="查看数据集版本清单",
    description="返回数据集版本导入时生成的文件清单。版本必须已经处理完成。",
)
def get_dataset_manifest(version_id: str, db: DbSession, user: CurrentUser) -> dict[str, object]:
    version = db.get(DatasetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    dataset = _load_dataset(db, version.dataset_id)
    project = load_project(db, dataset.project_id)
    ensure_project_access(db, project, user)
    if not version.manifest_uri:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Manifest is not ready")
    path = (storage_root() / version.manifest_uri).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


@router.get(
    "/projects/{project_id}/code-packages",
    response_model=list[CodePackageView],
    summary="查询项目代码包",
    description="查询指定项目下的代码包及其版本信息。",
)
def list_code_packages(project_id: str, db: DbSession, user: CurrentUser) -> list[CodePackageView]:
    project = load_project(db, project_id)
    ensure_project_access(db, project, user)
    packages = db.scalars(
        select(CodePackage)
        .where(CodePackage.project_id == project_id)
        .options(selectinload(CodePackage.versions))
        .order_by(CodePackage.created_at.desc())
    ).all()
    return [_code_view(code_package) for code_package in packages]


@router.post(
    "/projects/{project_id}/code-packages",
    response_model=CodePackageView,
    status_code=status.HTTP_201_CREATED,
    summary="创建代码包",
    description="在指定项目下创建一个代码包。项目所有者、研究人员或管理员可操作。",
)
def create_code_package(
    project_id: str, payload: ResourceCreate, db: DbSession, user: CurrentUser
) -> CodePackageView:
    _ensure_write_access(db, project_id, user)
    code_package = CodePackage(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description.strip(),
        created_by=user.id,
    )
    _commit_resource(db, code_package, "code_package", user.id)
    return _code_view(_load_code_package(db, code_package.id))


@router.get(
    "/code-packages/{code_package_id}",
    response_model=CodePackageView,
    summary="获取代码包详情",
    description="获取代码包信息以及代码版本列表。",
)
def get_code_package(code_package_id: str, db: DbSession, user: CurrentUser) -> CodePackageView:
    code_package = _load_code_package(db, code_package_id)
    project = load_project(db, code_package.project_id)
    ensure_project_access(db, project, user)
    return _code_view(code_package)


@router.post(
    "/code-packages/{code_package_id}/versions/upload",
    response_model=CodePackageView,
    summary="上传代码包版本",
    description="上传 ZIP 或 TAR.GZ 代码包，保存为不可变代码版本。",
)
async def upload_code_version(
    code_package_id: str,
    db: DbSession,
    user: CurrentUser,
    file: UploadFile = File(..., description="代码包压缩包，支持 ZIP 或 TAR.GZ"),
    default_workdir: str = Form(".", description="训练时默认工作目录，默认为当前目录"),
    default_entrypoint: str = Form("", description="默认入口命令，可为空"),
) -> CodePackageView:
    code_package = _load_code_package(db, code_package_id)
    _ensure_write_access(db, code_package.project_id, user)
    temp_path, sha256, archive_size, filename = await save_upload_to_temp(file)
    next_version = (
        db.scalar(
            select(func.max(CodeVersion.version)).where(
                CodeVersion.code_package_id == code_package_id
            )
        )
        or 0
    ) + 1
    version = CodeVersion(
        code_package_id=code_package_id,
        version=next_version,
        source_filename=filename,
        sha256=sha256,
        archive_size=archive_size,
        default_workdir=default_workdir.strip() or ".",
        default_entrypoint=default_entrypoint.strip(),
        created_by=user.id,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    try:
        result = ingest_archive(
            temp_path=temp_path,
            destination_prefix=(
                Path("code") / code_package.project_id / code_package.id / version.id
            ),
            filename=filename,
            sha256=sha256,
            archive_size=archive_size,
            detect_dataset=False,
        )
        version.archive_uri = result.archive_uri
        version.cache_uri = result.cache_uri
        version.manifest_uri = result.manifest_uri
        version.extracted_size = result.extracted_size
        version.file_count = result.file_count
        version.status = ResourceVersionStatus.READY
        version.ready_at = utc_now()
    except Exception as exc:
        version.status = ResourceVersionStatus.FAILED
        version.error_message = str(exc)[:2000]
        db.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.add(
        AuditLog(
            actor_id=user.id,
            action="code.version.upload",
            resource_type="code_version",
            resource_id=version.id,
            detail=f"v{version.version}:{filename}",
        )
    )
    db.commit()
    return _code_view(_load_code_package(db, code_package_id))


@router.get(
    "/code-versions/{version_id}/manifest",
    summary="查看代码版本清单",
    description="返回代码版本导入时生成的文件清单。",
)
def get_code_manifest(version_id: str, db: DbSession, user: CurrentUser) -> dict[str, object]:
    version = db.get(CodeVersion, version_id)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    code_package = _load_code_package(db, version.code_package_id)
    project = load_project(db, code_package.project_id)
    ensure_project_access(db, project, user)
    if not version.manifest_uri:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Manifest is not ready")
    path = (Path(settings.storage_root).resolve() / version.manifest_uri).resolve()
    return json.loads(path.read_text(encoding="utf-8"))
