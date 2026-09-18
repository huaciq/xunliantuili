import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.database import Base, engine
from app.services.bootstrap import ensure_bootstrap_admin, ensure_development_catalog
from app.services.gpu_inventory import refresh_gpu_inventory
from app.services.scheduler import scheduler

logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "系统状态", "description": "服务存活检查与运行状态。"},
    {"name": "认证", "description": "登录、当前用户和访问令牌。"},
    {"name": "用户管理", "description": "平台用户的查询和创建，仅管理员可操作。"},
    {"name": "项目与成员", "description": "项目创建、项目查询和成员管理。"},
    {"name": "数据集与代码", "description": "数据集、代码包及其版本上传和清单查询。"},
    {"name": "训练任务", "description": "训练模板、GPU 资源、任务调度、日志和停止操作。"},
    {"name": "实验与模型", "description": "指标、训练产物、模型注册、导出和一致性检查。"},
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)
    ensure_bootstrap_admin()
    ensure_development_catalog()
    if settings.executor_backend == "docker":
        try:
            count = refresh_gpu_inventory()
            logger.info("Discovered %s supported NVIDIA GPUs", count)
        except RuntimeError as exc:
            logger.warning("GPU discovery failed: %s", exc)
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(
    title="视觉模型推理训练平台 API",
    description=(
        "视觉模型训练平台的控制面 API。"
        "所有业务接口使用 `/api/v1` 前缀，并通过 Bearer Token 进行身份认证。"
    ),
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
