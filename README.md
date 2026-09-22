# 视觉模型推理训练平台

面向科研实验室的视觉模型训练、实验和模型产物管理平台。当前已完成 M1-M4 的本地开发闭环：用户与项目权限、数据集和代码包版本、GPU 训练调度、实验指标、产物下载、模型注册及 ONNX 导出。

详细设计见 [DESIGN.md](DESIGN.md)。

完整的 Windows 开发、Linux 首次部署、日常发布、权限、安全和故障排查流程见 [DEPLOYMENT_MANUAL.md](DEPLOYMENT_MANUAL.md)。

## 目录

```text
backend/       FastAPI API、SQLAlchemy 模型、Alembic 和测试
frontend/      Vue 3 管理端
infra/         PostgreSQL 与 MLflow 初始化文件
compose.yaml   PostgreSQL、Redis、MinIO、MLflow
```

开发模式下，上传的压缩包、解压缓存和 manifest 默认保存在 `backend/storage/`。支持 ZIP、TAR、TAR.GZ 和 TGZ，单包默认限制为 5 GB。

M3 默认使用内存 Fake Executor，不会在开发机上真正启动训练。Linux 实机资源按 `nvidia-smi` 动态登记，当前验证配置为 2 张 RTX 3090 + 1 张 Tesla T4，后续增加 RTX 4090 或其他支持卡无需改调度架构。Linux 部署时将 `EXECUTOR_BACKEND` 切换为 `docker`，即可使用 Docker Engine 和 NVIDIA Container Toolkit 执行真实任务。

Docker Executor 已实现，支持 NVIDIA GPU UUID、只读代码/数据挂载、资源限制、容器日志与重启对账。Linux 准备和验证步骤见 [LINUX_DEPLOYMENT.md](LINUX_DEPLOYMENT.md)。当前 Windows 开发机没有 Docker CLI，因此镜像构建与 GPU 容器运行仍需到目标服务器验收。

## 当前能力

- 管理员、项目成员与角色权限。
- 数据集和用户自备代码压缩包的版本化上传、安全解压与 manifest。
- YOLO Detection 和 Custom Command 训练模板。
- 按型号申请 1-2 张 GPU，同一任务不混用不同型号；当前支持 RTX 3090、RTX 4090 和 Tesla T4。
- 任务草稿、排队、运行、成功、失败和停止状态流转。
- WebSocket 实时日志、任务事件与 GPU 占用视图。
- 逐轮 loss、precision、recall、mAP50 指标及 checkpoint/report 产物。
- 项目模型注册表、模型版本、ONNX 导出与一致性检查结果。

Fake Executor 生成的 checkpoint 和 ONNX 是开发占位文件，界面与控制面行为完整，但不能用于真实推理。Linux Docker Executor 将负责生成真实权重、执行 `torch.onnx.export` 并回传一致性检查结果。

## 本地开发

### 后端

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。默认使用 SQLite，因此开发控制面不需要 Docker。

```powershell
Set-Location backend
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

API 文档地址：`http://127.0.0.1:8000/docs`

### 前端

需要 Node.js 20+。

```powershell
Set-Location frontend
npm install
npm run dev
```

Web 地址：`http://127.0.0.1:5173`

开发默认管理员：

```text
admin@example.com
ChangeMe123!
```

部署前必须在根目录 `.env` 中更换管理员密码和 `APP_SECRET_KEY`。

## 基础设施

目标 Linux 服务器安装 Docker 后，在项目根目录执行：

```bash
cp .env.example .env
docker compose up -d
```

随后将 `DATABASE_URL` 修改为 PostgreSQL 地址，并运行 Alembic 迁移。MinIO 控制台默认端口为 `9001`，MLflow 默认端口为 `5000`。

## 质量检查

```powershell
Set-Location backend
uv run ruff check .
uv run pytest

Set-Location ../frontend
npm run build
```
