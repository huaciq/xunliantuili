from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.deps import AdminUser, DbSession
from app.models import User
from app.schemas import UserCreate, UserSummary
from app.security import hash_password

router = APIRouter(prefix="/users", tags=["用户管理"])


@router.get(
    "",
    response_model=list[UserSummary],
    summary="查询用户列表",
    description="查询平台用户列表。需要系统管理员权限。",
)
def list_users(db: DbSession, _: AdminUser) -> list[User]:
    return list(db.scalars(select(User).order_by(User.created_at)).all())


@router.post(
    "",
    response_model=UserSummary,
    status_code=status.HTTP_201_CREATED,
    summary="创建平台用户",
    description=(
        "创建一个新的平台用户。姓名至少 2 个字符，密码长度为 8 到 128 个字符，需要系统管理员权限。"
    ),
)
def create_user(payload: UserCreate, db: DbSession, _: AdminUser) -> User:
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    user = User(
        email=email,
        name=payload.name.strip(),
        password_hash=hash_password(payload.password),
        system_role=payload.system_role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
