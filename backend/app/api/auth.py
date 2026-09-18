from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.deps import CurrentUser, DbSession
from app.models import User
from app.schemas import LoginRequest, TokenResponse, UserSummary
from app.security import create_access_token, verify_password

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="用户登录",
    description="使用邮箱和密码登录，成功后返回 JWT Bearer 访问令牌。",
)
def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    return TokenResponse(access_token=create_access_token(user.id))


@router.get(
    "/me",
    response_model=UserSummary,
    summary="获取当前用户",
    description="根据请求头中的 Bearer Token 返回当前登录用户信息。",
)
def me(user: CurrentUser) -> User:
    return user
