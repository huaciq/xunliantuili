# Linux GPU 执行层部署

完整的从 Windows 开发到 Linux 长期运行手册见 [DEPLOYMENT_MANUAL.md](DEPLOYMENT_MANUAL.md)。本文保留 GPU 执行层的补充说明。

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

Ultralytics 构建不在 Docker 内访问 GitHub。官方 `yolo11n.pt` 基础权重作为固定构建资产随仓库分发；拉取代码后可先校验文件：

```bash
cd /home/zxy/workspace/xunliantuili
sha256sum infra/training/assets/yolo11n.pt
ls -lh infra/training/assets/yolo11n.pt
```

预期 SHA256 为：

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

然后构建训练镜像：

```bash
docker compose --profile training-images build training-pytorch
docker compose --profile training-images build training-ultralytics
```

Ultralytics 镜像会安装 OpenCV 所需的 `libxcb1`、`libgl1` 和相关最小运行库，将离线权重复制到 `/opt/models/yolo11n.pt`，并在构建期实际加载模型。因此构建成功本身也是一次依赖和权重完整性检查。

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

基础设施中的 MinIO 使用固定的官方 Quay 镜像 `quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z`，不再使用已经不可用的 Docker Hub `minio/minio:latest`。

执行迁移并启动后端：

```bash
cd backend
uv sync --frozen --python 3.11
uv run alembic upgrade head
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
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

## 生产服务

基础设施端口默认只监听 `127.0.0.1`。启动服务并创建 MLflow bucket：

```bash
docker compose up -d postgres redis minio mlflow
docker compose exec -T mlflow python -c "import boto3,os; s=boto3.client('s3',endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']); names=[b['Name'] for b in s.list_buckets()['Buckets']]; 'mlflow' in names or s.create_bucket(Bucket='mlflow')"
```

四个基础设施容器使用 `restart: unless-stopped`。Docker 服务随系统启动后会恢复这些容器；手动停止的容器保持停止。

后端只运行一个 Uvicorn worker，因为当前调度器与 API 位于同一进程。安装 systemd unit：

```bash
sudo cp deploy/systemd/train-platform-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now train-platform-api
sudo systemctl status train-platform-api
```

构建前端并安装 Nginx 配置：

```bash
cd frontend
npm ci
npm run build
sudo install -d -m 0755 /var/www/train-platform
sudo rsync -a --delete dist/ /var/www/train-platform/
cd ..
sudo cp deploy/nginx/train-platform.conf /etc/nginx/sites-available/train-platform
sudo ln -sfn /etc/nginx/sites-available/train-platform /etc/nginx/sites-enabled/train-platform
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

生产环境更新：

```bash
git pull --ff-only origin main
cd backend && uv sync --frozen && uv run alembic upgrade head && cd ..
cd frontend && npm ci && npm run build && sudo rsync -a --delete dist/ /var/www/train-platform/ && cd ..
sudo systemctl restart train-platform-api
curl -fsS http://127.0.0.1:8000/api/v1/health
```
