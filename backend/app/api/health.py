from fastapi import APIRouter

router = APIRouter(tags=["系统状态"])


@router.get(
    "/health",
    summary="健康检查",
    description="检查 API 服务是否正常运行。正常时返回 `status=ok`。",
)
def health() -> dict[str, str]:
    return {"status": "ok"}
