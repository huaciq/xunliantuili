# 视觉模型推理训练平台方案设计

版本：V1.0  
日期：2026-09-17

## 1. 建设目标

平台服务于单台四 GPU 服务器，当前 GPU 组成：

- 2 x NVIDIA RTX 3090
- 2 x NVIDIA RTX 4090

平台面向实验室成员，提供以下闭环：

1. 创建项目并管理成员。
2. 上传或导入数据集，生成不可变数据集版本。
3. 上传算法代码压缩包，生成不可变代码快照。
4. 选择训练模板、运行环境、数据集版本和 GPU 要求。
5. 排队、启动、查看日志、停止或恢复训练。
6. 查看训练指标、评估结果和产物。
7. 注册模型版本，导出 ONNX；后续支持 TensorRT 和 OpenVINO。

平台不提供 Git 托管功能。用户在平台之外维护代码，提交训练时上传 ZIP/TAR.GZ，或从管理员授权的服务器目录导入。

## 2. 技术路线

当前阶段采用单机容器调度，不部署 Kubernetes。

| 模块 | 技术选型 |
| --- | --- |
| Web 前端 | Vue 3 + TypeScript + Vite + Pinia + Vue Router |
| UI 组件 | Naive UI |
| API 服务 | FastAPI + SQLAlchemy 2 + Alembic + Pydantic 2 |
| 主数据库 | PostgreSQL 16 |
| 队列与事件 | Redis 7 |
| 后台任务 | Celery |
| 训练执行 | Docker Engine API + NVIDIA Container Toolkit |
| 对象存储 | MinIO |
| 实验追踪 | MLflow Tracking Server |
| 实时日志 | Docker logs + WebSocket；归档到 MinIO |
| 主机监控 | Prometheus + NVIDIA DCGM Exporter + Grafana |
| 反向代理 | Nginx |
| 身份认证 | 平台 JWT；后续可接 LDAP/OIDC |

暂不引入 Harbor。训练镜像先使用服务器本地 Docker 镜像，由管理员维护允许使用的镜像列表。需要多人构建环境或增加服务器时再部署 Harbor。

## 3. 部署拓扑

```text
Browser
   |
 Nginx
   |-------------------------|
   v                         v
Web static                FastAPI API
                             |
          |------------------|------------------|
          v                  v                  v
      PostgreSQL           Redis              MinIO
                             |                  |
                             v                  |
                       Scheduler/Worker         |
                             |                  |
                             v                  |
                       Docker Engine <----------|
                             |
              |--------------|--------------|
              v              v              v
          Training       Evaluation       Export
          Container       Container       Container
              |
       NVIDIA GPU 0..3

MLflow 使用 PostgreSQL 保存元数据，使用 MinIO 保存 artifact。
Prometheus/Grafana 读取 API、主机和 GPU 指标。
```

控制面服务可以使用 Docker Compose 部署。训练容器由 Worker 通过 Docker API 动态创建，不加入控制面 Compose 生命周期。

## 4. 系统边界

### 4.1 平台负责

- 用户、项目、角色和配额。
- 数据集、代码包和训练产物的版本管理。
- 训练模板和参数表单。
- GPU 资源分配和任务排队。
- 容器创建、停止、日志、状态和退出码采集。
- 实验元数据和指标入口。
- checkpoint、模型注册和导出。

### 4.2 用户负责

- 在平台外开发和测试算法代码。
- 提供可执行的代码包。
- 声明入口命令、依赖环境和参数。
- 保证代码兼容所选训练镜像。

### 4.3 管理员负责

- 创建和验证基础训练镜像。
- 配置可导入的服务器目录白名单。
- 配置 GPU、用户配额和任务优先级。
- 维护 CUDA、驱动、Docker 和平台服务。

## 5. 核心业务模块

### 5.1 用户与项目

角色分为：

- `system_admin`：平台和服务器管理。
- `project_owner`：项目成员、资源和全部实验管理。
- `researcher`：数据、代码、训练和模型操作。
- `viewer`：只读查看。

所有数据、代码、运行和模型均归属于项目。第一阶段不做复杂组织树。

### 5.2 数据集管理

支持两种导入方式：

1. 浏览器上传 ZIP/TAR.GZ。
2. 从管理员配置的服务器目录导入。

导入过程由后台任务完成：

1. 校验文件类型和大小。
2. 安全解压，拒绝绝对路径、`..` 路径和符号链接逃逸。
3. 扫描文件并生成 manifest。
4. 计算文件大小、数量和内容摘要。
5. 识别 YOLO、COCO、分类目录等格式。
6. 生成不可变版本并上传 MinIO。

数据集版本一旦进入 `READY` 状态不得原地修改。任何增删文件都会产生新版本。

对象存储布局：

```text
datasets/{project_id}/{dataset_id}/{version}/archive.tar
datasets/{project_id}/{dataset_id}/{version}/manifest.json
datasets/{project_id}/{dataset_id}/{version}/preview/*
```

第一阶段训练容器启动前将数据版本解压到本机只读缓存：

```text
/srv/train-platform/cache/datasets/{dataset_version_id}/
```

容器中统一挂载为 `/workspace/dataset:ro`。缓存按版本复用，避免每次训练重复解压。

### 5.3 代码包管理

代码不使用 Git，由平台保存提交时的不可变快照。

提交方式：

- 上传 ZIP/TAR.GZ。
- 导入白名单目录，平台打包后保存快照。

每个代码版本保存：

- 原始文件名。
- SHA-256。
- 文件清单。
- 默认工作目录。
- 默认入口命令。
- 创建者和说明。

对象存储布局：

```text
code/{project_id}/{code_package_id}/{version}/source.tar.gz
code/{project_id}/{code_package_id}/{version}/manifest.json
```

训练时解压到独立运行目录，容器中挂载为 `/workspace/code:ro`。用户代码不得直接挂载宿主机任意路径。

### 5.4 运行环境

第一阶段的环境由管理员预制镜像，建议至少提供：

- `pytorch-cu12-base`：通用 PyTorch 环境。
- `ultralytics-yolo`：固定版本的 Ultralytics 环境。
- `mmdetection`：固定版本的 MMEngine/MMDetection 环境。
- `anomalib`：异常检测环境。

数据库记录镜像名称和 digest。运行时固定 digest，不使用浮动的 `latest` 作为可复现实验依据。

用户第一阶段不能任意填写 Docker 镜像或 Dockerfile，避免恶意镜像和不可控依赖。后续增加“环境构建申请”流程。

### 5.5 训练模板

训练模板不是训练代码，而是平台理解某类任务的适配协议。

首批模板：

- YOLO Detection。
- YOLO Segmentation。
- Image Classification。
- Custom PyTorch。
- Custom Command。

每个模板包含：

- 参数 JSON Schema。
- 参数到命令行的转换器。
- 默认镜像。
- 入口命令模板。
- 指标读取方式。
- checkpoint 发现规则。
- 支持的模型导出器。

自定义命令模板示例：

```text
python train.py \
  --data /workspace/dataset/data.yaml \
  --output /workspace/output \
  --epochs 100
```

平台不使用 Shell 拼接未转义字符串。命令在数据库和 Docker API 中保存为参数数组。

### 5.6 实验追踪

每次运行创建平台 `training_run`，并关联一个 MLflow run。

平台数据库保存业务状态和关键摘要；MLflow 保存：

- 超参数。
- epoch 指标。
- loss、precision、recall、mAP 等曲线。
- 环境信息。
- 图表、混淆矩阵和评估报告。

训练代码可以主动调用 MLflow SDK。对于 YOLO 模板，平台适配器负责注入 MLflow 配置。对于没有接入 SDK 的自定义代码，平台仍保存 stdout、退出码和最终产物。

### 5.7 模型与导出

训练任务输出目录统一为：

```text
/workspace/output
```

训练完成后 Worker 扫描并上传：

```text
runs/{project_id}/{run_id}/logs/
runs/{project_id}/{run_id}/checkpoints/
runs/{project_id}/{run_id}/artifacts/
runs/{project_id}/{run_id}/metadata.json
```

用户从 checkpoint 创建模型版本。模型版本状态：

```text
CANDIDATE -> VALIDATED -> RELEASED -> ARCHIVED
```

模型导出是独立 `export_job`，不在 API 进程内执行。第一阶段支持 PyTorch 权重到 ONNX，并执行一次源模型与 ONNX Runtime 的数值一致性检查。

## 6. GPU 调度设计

### 6.1 GPU 资源登记

启动时 Agent 通过 NVML 发现 GPU，并保存稳定标识：

- GPU UUID。
- 型号。
- 总显存。
- 当前健康状态。
- 是否被管理员禁用。

数据库和调度器使用 GPU UUID，不使用可能因重启变化的显示顺序作为唯一标识。

### 6.2 第一阶段分配单位

一张物理 GPU 同时只分配给一个训练任务，不做显存切片或超卖。原因是训练显存波动大，混部容易产生不可预测的 OOM 和性能干扰。

任务可声明：

- `gpu_count`：0、1、2 或 4。
- `gpu_model_policy`：`ANY`、`RTX_3090`、`RTX_4090`。
- `min_gpu_memory_gb`。
- `cpu_limit`。
- `memory_limit_gb`。
- `priority`。

### 6.3 异构卡规则

- 单卡任务默认选择满足条件的空闲 GPU。
- 双卡任务默认要求同型号 GPU，因此可分配两张 3090 或两张 4090。
- 默认禁止把 3090 和 4090 放进同一个分布式训练任务。
- 四卡异构训练仅允许管理员显式开启，并向用户展示性能和兼容性警告。
- 4090 优先分配给明确要求 4090 的任务；`ANY` 任务优先使用当前更容易形成连续同型组合的卡组。

### 6.4 排队策略

第一阶段使用带老化的优先级 FIFO：

```text
effective_priority = base_priority + waiting_age_score
```

约束：

- 每个用户默认最多运行 1 个 GPU 任务。
- 每个用户默认最多排队 3 个任务。
- 项目可设置 GPU 并发上限。
- 管理员可以调整优先级、暂停队列或禁用 GPU。

Scheduler 使用 PostgreSQL 事务和行锁领取任务；Redis 负责唤醒与事件广播，不作为资源分配的最终事实来源。

### 6.5 GPU 锁与故障恢复

`gpu_allocation` 表对活动 GPU UUID 设置唯一约束。启动容器前创建租约，容器退出后释放。

Scheduler/Worker 重启时执行对账：

1. 查询数据库中的活动任务。
2. 查询带平台标签的 Docker 容器。
3. 重新绑定仍在运行的容器。
4. 对已消失的容器补记退出状态。
5. 回收没有对应容器的过期 GPU 租约。

## 7. 容器执行协议

每个训练容器使用非 root 用户运行，并设置：

- 指定 GPU UUID。
- CPU 和内存限制。
- `no-new-privileges`。
- 丢弃不需要的 Linux capabilities。
- 不挂载 Docker socket。
- 不使用 privileged 模式。
- 代码和数据只读挂载。
- 仅输出目录可写。
- 平台内部服务使用独立网络。

标准目录：

```text
/workspace/code       # 只读代码
/workspace/dataset    # 只读数据
/workspace/output     # 可写输出
/workspace/config     # 本次运行配置
```

标准环境变量：

```text
PLATFORM_RUN_ID
PLATFORM_PROJECT_ID
DATASET_DIR=/workspace/dataset
OUTPUT_DIR=/workspace/output
MLFLOW_TRACKING_URI
MLFLOW_RUN_ID
```

所有容器增加标签：

```text
train-platform.managed=true
train-platform.run-id={run_id}
train-platform.project-id={project_id}
```

## 8. 任务状态机

```text
DRAFT
  -> QUEUED
  -> PREPARING
  -> STARTING
  -> RUNNING
  -> SUCCEEDED

QUEUED/PREPARING/STARTING/RUNNING
  -> CANCEL_REQUESTED
  -> STOPPING
  -> STOPPED

PREPARING/STARTING/RUNNING/STOPPING
  -> FAILED
```

停止分为：

- 优雅停止：向容器主进程发送 `SIGTERM`，等待模板配置的超时时间。
- 强制停止：超时后发送 `SIGKILL`。

恢复训练会创建新的 run，并记录 `parent_run_id` 和 `resume_checkpoint_id`，不修改原运行记录。

## 9. 核心数据表

第一阶段主要表：

```text
users
projects
project_members
datasets
dataset_versions
code_packages
code_versions
runtime_images
training_templates
training_runs
run_events
run_artifacts
gpu_devices
gpu_allocations
model_versions
export_jobs
audit_logs
```

`training_runs` 关键字段：

```text
id
project_id
name
template_id
dataset_version_id
code_version_id
runtime_image_id
status
parameters_json
command_json
requested_gpu_count
requested_gpu_model
allocated_gpu_uuids
priority
container_id
mlflow_run_id
parent_run_id
resume_checkpoint_id
created_by
queued_at
started_at
finished_at
exit_code
failure_reason
version
```

`version` 用于乐观锁，防止状态更新覆盖。

## 10. 第一阶段 API

```text
POST   /api/v1/auth/login
GET    /api/v1/users/me

GET    /api/v1/projects
POST   /api/v1/projects
GET    /api/v1/projects/{id}
POST   /api/v1/projects/{id}/members

GET    /api/v1/projects/{id}/datasets
POST   /api/v1/projects/{id}/datasets
POST   /api/v1/datasets/{id}/versions/upload
POST   /api/v1/datasets/{id}/versions/import
GET    /api/v1/dataset-versions/{id}

GET    /api/v1/projects/{id}/code-packages
POST   /api/v1/projects/{id}/code-packages
POST   /api/v1/code-packages/{id}/versions/upload
POST   /api/v1/code-packages/{id}/versions/import

GET    /api/v1/runtime-images
GET    /api/v1/training-templates

GET    /api/v1/projects/{id}/runs
POST   /api/v1/projects/{id}/runs
GET    /api/v1/runs/{id}
POST   /api/v1/runs/{id}/submit
POST   /api/v1/runs/{id}/stop
POST   /api/v1/runs/{id}/force-stop
POST   /api/v1/runs/{id}/resume
GET    /api/v1/runs/{id}/artifacts
GET    /api/v1/runs/{id}/logs
WS     /api/v1/runs/{id}/logs/stream

GET    /api/v1/projects/{id}/models
POST   /api/v1/projects/{id}/models
POST   /api/v1/models/{id}/exports

GET    /api/v1/resources/gpus
GET    /api/v1/resources/queue
```

## 11. 页面范围

第一阶段页面：

1. 登录。
2. 项目列表和项目概览。
3. 项目成员。
4. 数据集列表、版本上传和版本详情。
5. 代码包列表、版本上传和版本详情。
6. 新建训练任务向导。
7. 训练队列。
8. 运行详情：状态、参数、GPU、实时日志、指标、产物。
9. 模型列表和模型详情。
10. ONNX 导出任务。
11. 管理员 GPU、镜像和模板管理。

## 12. MVP 开发顺序

### M1：平台骨架

- Monorepo、Docker Compose 和配置管理。
- 用户登录、项目和成员权限。
- PostgreSQL、Redis、MinIO 接入。
- 审计日志基础结构。

### M2：数据与代码

- 分片上传或预签名上传。
- 数据集/代码版本后台导入。
- manifest、安全解压和本地缓存。
- 镜像与训练模板管理。

### M3：训练闭环

- [x] 四张目标 GPU 的开发资源登记与状态展示。
- [x] Scheduler、GPU 独占分配和统一 Executor 接口。
- [x] 创建、排队、启动、实时日志和停止。
- [x] Custom Command 与 YOLO Detection 模板。
- [x] Linux GPU 发现、Docker Executor、安全挂载和容器重启对账实现。
- [ ] 在目标服务器完成 NVIDIA Container Toolkit 与四张 GPU 实机联调。

### M4：实验和模型

- [x] 平台原生实验指标采集与指标页面。
- [x] checkpoint、report 与 export artifact 收集和下载。
- [x] 项目级模型注册表与版本管理。
- [x] Fake Executor 的 ONNX 导出和一致性检查闭环。
- [ ] Linux 实机 MLflow 对接、真实 ONNX 导出和数值一致性检查。

### M5：稳定性

- 进程重启对账。
- 配额、优先级老化和审计。
- 磁盘空间保护、缓存清理和失败重试。
- 单元、集成和端到端测试。

## 13. 第一版明确不做

- Git 服务或网页代码编辑器。
- Kubernetes。
- 多服务器调度。
- GPU 共享和显存超卖。
- Notebook 环境。
- 在线推理服务。
- 用户任意构建或运行镜像。
- 自动标注和复杂数据清洗流水线。
- 多租户计费。

## 14. 演进路径

执行层必须实现统一接口：

```text
Executor.submit(run_spec) -> execution_id
Executor.stop(execution_id, grace_period)
Executor.inspect(execution_id) -> execution_status
Executor.stream_logs(execution_id)
Executor.collect_artifacts(execution_id)
```

第一阶段实现 `DockerExecutor`。未来增加服务器时实现 `KubernetesExecutor`，上层项目、数据、代码、运行和模型 API 保持不变。

## 15. 开发前提

目标服务器建议使用 Linux，安装 NVIDIA 驱动、Docker Engine、Docker Compose Plugin 和 NVIDIA Container Toolkit。生产部署不建议以 Windows Docker Desktop 作为训练执行环境。

开发阶段可以在无 GPU 的机器上运行控制面，并提供 `FakeExecutor` 模拟排队、日志、成功和失败；GPU 集成测试在目标服务器执行。
