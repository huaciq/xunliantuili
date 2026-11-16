# 视觉模型推理训练平台开发与部署手册

本文档描述从 Windows 开发、提交代码到 Linux GPU 服务器长期运行的完整流程。当前部署路径约定为 `/home/zxy/workspace/xunliantuili`，运行用户为 `huaciq`。如果路径或用户名不同，需要同步修改 systemd 文件和命令。

## 1. 系统组成

```text
浏览器
  │ HTTP :80
  ▼
Nginx ───────────────► /var/www/train-platform（前端静态文件）
  │ /api/
  ▼
FastAPI + 训练调度器（127.0.0.1:8000，systemd/huaciq）
  │
  ├── PostgreSQL：业务数据库
  ├── Redis：基础设施服务，供后续队列能力使用
  ├── MinIO：对象存储，供 MLflow 使用
  └── Docker Executor ──► GPU 训练容器
```

当前 GPU 服务器已验证 `RTX 3090 + Tesla T4 + RTX 3090`。GPU 按 UUID 登记和分配，未来增加 RTX 4090 不需要修改调度架构。

组件和职责：

| 组件 | 运行方式 | 主要位置或端口 |
| --- | --- | --- |
| FastAPI、调度器 | systemd，单 worker，`huaciq` | `127.0.0.1:8000` |
| 前端 | 构建为静态文件，由 Nginx 提供 | `/var/www/train-platform` |
| PostgreSQL、Redis、MinIO、MLflow | Docker Compose，后台容器 | 默认只绑定 `127.0.0.1` |
| PyTorch、Ultralytics | Docker 训练镜像 | `train-platform/*:development` |
| 数据和训练产物 | 宿主机目录 | `/srv/train-platform/storage` |

## 2. 权限分工

日常代码、依赖和构建使用 `huaciq`：

- `git pull`、`uv sync`、Alembic 迁移。
- `npm ci`、`npm run build`。
- `docker compose` 和 Docker 训练任务。
- FastAPI 和训练调度器进程。

系统级操作使用 root：

- 安装 apt 软件包。
- 写入 `/etc/systemd/system` 和 `/etc/nginx`。
- 发布 `/var/www/train-platform`。
- 启动、停止和重载 systemd/Nginx。

不要用 root 对仓库执行 Git。仓库属于 `huaciq` 时，root 运行 Git 会触发 `dubious ownership`，强行加入 `safe.directory` 会掩盖权限问题。

## 3. Windows 开发

### 3.1 后端环境

后端固定使用 Python 3.11，版本约束位于 `backend/.python-version` 和 `backend/pyproject.toml`。不要让 uv 使用 Ubuntu 26.04 默认的 Python 3.14，否则旧版 `pydantic-core` 可能退回 Rust 编译并失败。

```powershell
Set-Location backend
uv python install 3.11
uv sync --frozen --extra dev
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

API 文档：`http://127.0.0.1:8000/docs`。

### 3.2 前端环境

需要 Node.js 20.19+ 或 22.12+，npm 随 Node.js 安装。开发时：

```powershell
Set-Location frontend
npm ci
npm run dev
```

开发页面：`http://127.0.0.1:5173`。

### 3.3 提交前检查

```powershell
Set-Location backend
uv run ruff check .
uv run pytest

Set-Location ../frontend
npm run build

Set-Location ..
git status --short
```

只有检查通过后才提交：

```powershell
git add <本次修改的文件>
git commit -m "描述本次修改"
git push origin main
git log -1 --oneline
```

`.env`、数据库、上传数据、模型权重缓存和 `node_modules` 不应提交。当前仓库已经将固定的 `infra/training/assets/yolo11n.pt` 作为构建资产纳入版本控制，并在 Dockerfile 中校验 SHA256。

## 4. Linux 首次部署

### 4.1 root：主机准备

确认驱动和架构：

```bash
nvidia-smi
uname -m
```

安装并配置 Docker Engine、NVIDIA Container Toolkit，然后确认：

```bash
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

创建或确认运行用户，并让它可以调用 Docker：

```bash
usermod -aG docker huaciq
install -d -o huaciq -g huaciq -m 0750 /srv/train-platform/storage
```

重新登录一次，使 `docker` 用户组生效。

### 4.2 huaciq：代码和配置

```bash
sudo -iu huaciq
cd /home/zxy/workspace/xunliantuili
git clone https://github.com/huaciq/xunliantuili.git .
```

如果目录已经存在，使用 `git pull --ff-only origin main`，不要重复 clone。

创建根目录 `.env`：

```bash
cp .env.example .env
chmod 600 .env
```

生产环境至少设置：

```dotenv
APP_ENV=production
APP_SECRET_KEY=<随机长密钥>
PLATFORM_BIND_ADDRESS=127.0.0.1
DATABASE_URL=postgresql+psycopg://platform:<数据库密码>@127.0.0.1:5432/platform
REDIS_URL=redis://127.0.0.1:6379/0
MINIO_ENDPOINT=127.0.0.1:9000
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
STORAGE_ROOT=/srv/train-platform/storage
EXECUTOR_BACKEND=docker
DOCKER_USER=<huaciq的UID>:<huaciq的GID>
AUTO_CREATE_TABLES=false
```

`DOCKER_USER` 使用实际值：

```bash
sed -i "s/^DOCKER_USER=.*/DOCKER_USER=$(id -u):$(id -g)/" .env
```

不要把 `.env`、密码、JWT 密钥粘贴到聊天或提交到 Git。

### 4.3 构建训练镜像

```bash
cd /home/zxy/workspace/xunliantuili
sha256sum infra/training/assets/yolo11n.pt
docker compose --profile training-images build training-pytorch
docker compose --profile training-images build training-ultralytics
```

预期 `yolo11n.pt` SHA256：

```text
0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

验证镜像：

```bash
docker run --rm --gpus all train-platform/ultralytics:development \
  python -c "import torch; from ultralytics import YOLO; YOLO('/opt/models/yolo11n.pt'); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### 4.4 启动基础设施

```bash
docker compose up -d postgres redis minio mlflow
docker compose ps
docker compose exec -T postgres pg_isready -U platform -d platform
docker compose exec -T redis redis-cli ping
curl --noproxy '*' -fsS http://127.0.0.1:9000/minio/health/live
curl --noproxy '*' -fsS http://127.0.0.1:5000/health
```

初始化 MLflow bucket：

```bash
docker compose exec -T mlflow python -c "import boto3,os; s=boto3.client('s3',endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']); names=[b['Name'] for b in s.list_buckets()['Buckets']]; 'mlflow' in names or s.create_bucket(Bucket='mlflow')"
```

### 4.5 安装 Python 依赖和迁移

```bash
cd backend
uv python install 3.11
uv venv --clear --python 3.11 .venv
uv sync --frozen --python 3.11
uv run alembic upgrade head
uv run alembic current
```

如果日志出现 `SOABI: cpython-314`，说明旧虚拟环境仍在使用 Python 3.14，重新执行 `uv venv --clear --python 3.11 .venv`。

### 4.6 安装后端 systemd

当前模板固定使用用户 `huaciq` 和路径 `/home/zxy/workspace/xunliantuili`。路径不同需要先修改模板。

退出到 root 后执行：

```bash
exit
cp /home/zxy/workspace/xunliantuili/deploy/systemd/train-platform-api.service /etc/systemd/system/train-platform-api.service
systemd-analyze verify /etc/systemd/system/train-platform-api.service
systemctl daemon-reload
systemctl enable --now train-platform-api
systemctl status train-platform-api --no-pager
curl --noproxy '*' -fsS http://127.0.0.1:8000/api/v1/health
```

后端只启用一个 Uvicorn worker，因为训练调度器与 API 当前位于同一进程。

### 4.7 构建和发布前端

如果 Node.js 已通过 NodeSource 安装，不要再次安装 Ubuntu 独立的 `npm` 包，以免产生包冲突。root 只安装 Nginx 和 rsync：

```bash
apt update
apt install -y nginx rsync
```

使用 `huaciq` 构建：

```bash
sudo -iu huaciq
cd /home/zxy/workspace/xunliantuili/frontend
npm ci
npm run build
exit
```

root 发布静态文件并配置 Nginx：

```bash
install -d -m 0755 /var/www/train-platform
rsync -a --delete /home/zxy/workspace/xunliantuili/frontend/dist/ /var/www/train-platform/
chown -R root:root /var/www/train-platform
cp /home/zxy/workspace/xunliantuili/deploy/nginx/train-platform.conf /etc/nginx/sites-available/train-platform
ln -sfn /etc/nginx/sites-available/train-platform /etc/nginx/sites-enabled/train-platform
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx
systemctl reload nginx
curl --noproxy '*' -I http://127.0.0.1/
curl --noproxy '*' -fsS http://127.0.0.1/api/v1/health
```

浏览器访问 `http://服务器IP/`。当前配置是 HTTP，公网使用前必须增加 HTTPS。

## 5. 日常发布流程

### 5.1 Windows：提交

```powershell
Set-Location backend
uv run ruff check .
uv run pytest
Set-Location ../frontend
npm run build
Set-Location ..
git status --short
git add <本次修改的文件>
git commit -m "描述本次修改"
git push origin main
```

确认 push 成功后再操作服务器。

### 5.2 Linux：huaciq 拉取和构建

```bash
sudo -iu huaciq
cd /home/zxy/workspace/xunliantuili
git status --short
git pull --ff-only origin main
git log -1 --oneline
docker compose up -d postgres redis minio mlflow

cd backend
uv sync --frozen --python 3.11
uv run alembic upgrade head

cd ../frontend
npm ci
npm run build
exit
```

`git status --short` 出现本地修改时先停止，不要直接 `reset`。如果是服务器临时修复，先保存到有名字的 stash，拉取后不要盲目 `stash pop`。

### 5.3 root：发布和重启

```bash
rsync -a --delete /home/zxy/workspace/xunliantuili/frontend/dist/ /var/www/train-platform/
chown -R root:root /var/www/train-platform
systemctl restart train-platform-api
curl --noproxy '*' -fsS http://127.0.0.1:8000/api/v1/health
curl --noproxy '*' -fsS http://127.0.0.1/api/v1/health
```

前端是带 hash 的静态资源，通常不需要重启 Nginx；浏览器用 Ctrl+F5 刷新即可。Nginx 配置变更时再执行 `nginx -t && systemctl reload nginx`。

## 6. SSH、重启和服务状态

关闭 SSH 不会停止服务：

- FastAPI 由 `train-platform-api.service` 管理。
- Nginx 由 systemd 管理。
- Compose 容器以后台模式运行。
- 前端没有 npm 常驻进程，只是 Nginx 读取静态文件。

基础设施容器使用 `restart: unless-stopped`，Docker 随系统启动后会自动恢复。检查：

```bash
systemctl is-enabled docker nginx train-platform-api
systemctl is-active docker nginx train-platform-api
docker compose ps
```

常用日志：

```bash
journalctl -u train-platform-api -n 100 --no-pager
journalctl -u train-platform-api -f
docker compose logs --tail=100 postgres redis minio mlflow
```

## 7. 功能验收

1. 浏览器登录管理员账号。
2. 查看 GPU 页面，确认 3090、T4、3090 均已登记。
3. 创建项目并上传一个包含 `data.yaml`、图片和标签的 YOLO 数据集。
4. 上传用户代码包，确认版本状态为 ready。
5. 创建 YOLO 训练任务，先用单张 3090、`epochs=1` 和小 batch。
6. 提交任务，观察实时日志和 `nvidia-smi`。
7. 测试停止任务、失败任务日志和成功任务产物。
8. 检查 checkpoint、report、指标和模型注册流程。

当前真实 Docker 训练已验证；Fake Executor、ONNX 导出和 MLflow 训练追踪仍有开发占位部分，不能把占位产物当作真实模型推理结果。

## 8. 安全和运维要点

- `.env` 使用 `chmod 600`，不提交 Git，不在聊天或日志中暴露密码。
- 对外只开放 Nginx 的 80/443 和必要的 SSH；不要开放 8000、5432、6379、9000、9001、5000。
- 当前没有 HTTPS，不适合直接暴露公网；应先使用 HTTPS、VPN 或内网访问控制。
- `huaciq` 加入 `docker` 组后，权限接近 root；这是当前 Docker Executor 架构的已知风险。
- 用户上传代码会在容器中执行，当前适用于受信任的实验室用户，不是强隔离的公有多租户服务。
- 训练容器已经使用非 root、只读代码/数据、`cap-drop=ALL`、`no-new-privileges`、资源限制，但仍共享宿主机内核。
- 当前训练容器使用 bridge 网络；不可信任务上线前应增加网络隔离和出口控制。
- 定期备份 PostgreSQL、MinIO 数据卷和 `/srv/train-platform/storage`，并验证恢复，不要只备份 Docker 镜像。
- 监控磁盘、GPU 温度/显存、Docker 日志、systemd 状态和训练失败率。

## 9. 常见故障

### `pydantic-core` 尝试 Rust 编译，日志出现 `cpython-314`

服务器使用了 Python 3.14，清理并重建项目虚拟环境：

```bash
cd /home/zxy/workspace/xunliantuili/backend
uv venv --clear --python 3.11 .venv
uv sync --frozen --python 3.11
```

### `detected dubious ownership`

不要用 root 拉代码：

```bash
sudo -iu huaciq
cd /home/zxy/workspace/xunliantuili
git pull --ff-only origin main
```

### `git pull` 提示本地修改会被覆盖

先查看 `git status --short` 和 `git diff`。确认是服务器临时改动后，再使用带说明的 `git stash push --include-untracked -m "server hotfix"`，拉取后不要直接恢复旧改动。

### systemd 服务返回 502

```bash
systemctl status train-platform-api --no-pager
journalctl -u train-platform-api -n 100 --no-pager
ss -lntp 'sport = :8000'
curl --noproxy '*' -v http://127.0.0.1:8000/api/v1/health
```

### 找不到 systemd 文件

通常是代码仍在旧提交。使用 `huaciq` 执行 `git pull --ff-only origin main`，确认 `deploy/systemd/train-platform-api.service` 存在后，再由 root 复制到 `/etc/systemd/system`。

### Docker 找不到 GPU

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

先修复 NVIDIA Container Toolkit，再检查 Docker Executor 配置。

### `apt install nodejs npm nginx` 冲突

NodeSource 的 Node.js 已经自带 npm，只安装：

```bash
apt install -y nginx rsync
```

## 10. 数据备份示例

数据库备份：

```bash
cd /home/zxy/workspace/xunliantuili
mkdir -p /srv/train-platform/backups
docker compose exec -T postgres pg_dump -U platform -d platform > /srv/train-platform/backups/platform-$(date +%F).sql
```

文件存储备份应使用带校验的备份工具或快照，将 `/srv/train-platform/storage` 复制到独立磁盘或备份服务器。备份目录不能与源目录位于同一块磁盘后就认为已经具备灾备能力。
