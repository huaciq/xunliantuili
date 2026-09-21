# Linux GPU 执行层部署

## 前置条件

- Linux 服务器可正常执行 `nvidia-smi`。
- Docker Engine 已安装并由 NVIDIA Container Toolkit 配置 GPU runtime。
- 运行后端的系统用户可执行 Docker，并对平台存储目录具有读写权限。
- 当前服务器可以是 2 张 RTX 3090 + 1 张 Tesla T4；后续增加 RTX 4090 或其他已适配型号时，GPU 会按 UUID 自动登记。

先执行预检：

```bash
bash scripts/linux-preflight.sh
```

## 构建训练镜像

```bash
docker compose --profile training-images build training-pytorch
docker compose --profile training-images build training-ultralytics
```

镜像名称与平台初始化目录一致：

```text
train-platform/pytorch:development
train-platform/ultralytics:development
```

## 配置执行器

复制 `.env.example` 为 `.env`，至少修改：

```dotenv
APP_ENV=production
APP_SECRET_KEY=<随机长密钥>
EXECUTOR_BACKEND=docker
STORAGE_ROOT=/srv/train-platform/storage
DOCKER_USER=1000:1000
DOCKER_NETWORK=bridge
AUTO_CREATE_TABLES=false
```

`DOCKER_USER` 应与拥有 `/srv/train-platform/storage` 的宿主机用户 UID/GID 一致。训练容器中的代码和数据目录为只读，只有每次运行的 output 目录可写。

执行迁移并启动后端：

```bash
cd backend
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动时平台会通过 `nvidia-smi` 登记 GPU。Docker 任务使用 GPU UUID，不依赖可能变化的显示顺序。

## 训练目录映射

```text
代码缓存                  -> /workspace/code       只读
数据集缓存                -> /workspace/dataset    只读
运行配置                  -> /workspace/config     只读
runs/{project}/{run}/output -> /workspace/output   可写
```

容器启动时应用 `no-new-privileges`、`cap-drop=ALL`、非 root UID、CPU/内存/PID 限制，并且不挂载 Docker socket。
