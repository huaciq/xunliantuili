from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.deps import AdminUser, DbSession
from app.models import AuditLog, RuntimeImage
from app.schemas import RuntimeImageCreate, RuntimeImageUpdate, RuntimeImageView

router = APIRouter(prefix="/admin/runtime-images", tags=["运行镜像管理"])


def _clean(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned or any(character.isspace() for character in cleaned):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be non-empty and contain no whitespace",
        )
    return cleaned


@router.get("", response_model=list[RuntimeImageView])
def list_runtime_images(db: DbSession, _: AdminUser) -> list[RuntimeImage]:
    return list(db.scalars(select(RuntimeImage).order_by(RuntimeImage.created_at)).all())


@router.post("", response_model=RuntimeImageView, status_code=status.HTTP_201_CREATED)
def create_runtime_image(
    payload: RuntimeImageCreate, db: DbSession, user: AdminUser
) -> RuntimeImage:
    name = payload.name.strip()
    if db.scalar(select(RuntimeImage).where(RuntimeImage.name == name)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Name already exists")
    runtime = RuntimeImage(
        name=name,
        image=_clean(payload.image, "image"),
        digest=_clean(payload.digest, "digest"),
        framework=_clean(payload.framework.lower(), "framework"),
        is_active=payload.is_active,
    )
    db.add(runtime)
    db.flush()
    db.add(
        AuditLog(
            actor_id=user.id,
            action="runtime_image.create",
            resource_type="runtime_image",
            resource_id=runtime.id,
            detail=f"{runtime.name}:{runtime.image}",
        )
    )
    db.commit()
    db.refresh(runtime)
    return runtime


@router.patch("/{runtime_id}", response_model=RuntimeImageView)
def update_runtime_image(
    runtime_id: str, payload: RuntimeImageUpdate, db: DbSession, user: AdminUser
) -> RuntimeImage:
    runtime = db.get(RuntimeImage, runtime_id)
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Runtime image not found")
    updates = payload.model_dump(exclude_unset=True)
    if "name" in updates:
        name = updates["name"].strip()
        duplicate = db.scalar(
            select(RuntimeImage).where(RuntimeImage.name == name, RuntimeImage.id != runtime.id)
        )
        if duplicate:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Name already exists")
        runtime.name = name
    for field in ("image", "digest", "framework"):
        if field in updates:
            value = str(updates[field]).lower() if field == "framework" else str(updates[field])
            setattr(runtime, field, _clean(value, field))
    if "is_active" in updates:
        runtime.is_active = bool(updates["is_active"])
    db.add(
        AuditLog(
            actor_id=user.id,
            action="runtime_image.update",
            resource_type="runtime_image",
            resource_id=runtime.id,
            detail=",".join(sorted(updates)),
        )
    )
    db.commit()
    db.refresh(runtime)
    return runtime
