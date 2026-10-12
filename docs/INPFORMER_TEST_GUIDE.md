# INP-Former 平台全流程测试

核对日期：2026-10-12。本文针对本仓库现有 `inpformer_multiclass` 模板。

## 1. 测试路线和账户

先用真实 MVTec AD 的小子集跑通：构造数据 → 上传数据版本 → 上传代码和预训练权重 → 准备并登记 Docker 镜像 → 提交训练 → 查看评估 → 下载 checkpoint → 注册模型。最后单独测试停止任务。

| 操作 | 执行位置与账户 |
| --- | --- |
| 下载、构造数据集 | Windows 普通账户，或 Linux 的 `huaciq` |
| 拉平台代码、准备算法代码、构建镜像 | Linux 的 `huaciq` |
| 上传数据、代码、创建任务 | 浏览器中的平台用户，拥有项目写权限 |
| 登记运行镜像 | 浏览器中的平台管理员 |
| 检查或重启系统服务 | Linux 的 `root`，或有 sudo 权限的用户 |

服务器项目路径沿用 `/home/zxy/workspace/xunliantuili`。算法准备目录使用 `/home/huaciq/inpformer-acceptance`，算法权重、数据不提交到平台 Git 仓库。

当前能力边界：

- 此模板强制要求 MVTec AD 的 15 类目录齐全；上传单独的 `bottle` 不能使用此模板。
- 官方多类别入口使用单张 GPU。首次选择 1 张 RTX 3090；申请 2 张不会自动获得分布式训练。
- 代码和数据目录在训练容器内只读，结果目录 `/workspace/output` 可写。编码器权重必须提前放入代码包。
- 官方入口在最后一轮评估完成后保存 `model.pth`。提前停止通常没有 checkpoint，也没有自动断点续训。
- 平台当前只解析特定 `results.csv` 的常见训练指标，尚未适配 INP-Former AUROC、AP、AUPRO 的曲线。先从训练日志验收指标。
- 模型注册可以验收；平台 ONNX 导出目前写入 `FAKE_ONNX` 占位内容，不能作为真实推理模型验收。
- 本地 Windows 没有 Docker CLI，本文的候选依赖环境尚未在 Linux GPU 上实际构建或训练；以下容器检查是服务器验收的一部分。

## 2. 检查服务器

如果当前是 root，先切换：

```bash
sudo -iu huaciq
cd /home/zxy/workspace/xunliantuili
git status --short
```

工作区没有需要处理的本地修改时拉取：

```bash
git pull --ff-only origin main
curl --noproxy '*' -fsS http://127.0.0.1/api/v1/health
docker compose ps
nvidia-smi
docker ps
cd backend
.venv/bin/python -c "from app.config import settings; print('executor=', settings.executor_backend); print('docker_user=', settings.docker_user); print('storage=', settings.storage_root)"
id
```

健康检查应返回 `{"status":"ok"}`，执行器应为 `docker`。`DOCKER_USER` 应与 huaciq 的 UID/GID 一致，存储目录需要允许这个 UID 写入。不要把带密码的整个 `.env` 发到聊天里。平台已有服务正常时无需重启。

## 3. 下载并理解数据集

从 [MVTec AD 官方下载页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads) 下载原始 MVTec AD，选择整个数据集或分别下载 15 类，解压到一个统一目录。这里使用原版 MVTec AD，而非 AD2、3D-AD 或 LOCO AD。

官方标注完整包约 4.9 GB，接近平台默认 5 GiB 的单包上限。首次上传使用下面的小子集。下载并解压完整数据时请预留足够磁盘空间；平台导入时还会保存原始压缩包和解压缓存。

所有类别名必须为：

```text
bottle cable capsule carpet grid hazelnut leather metal_nut
pill screw tile toothbrush transistor wood zipper
```

每类的结构如下：

```text
mvtec_anomaly_detection/
  bottle/
    train/good/000.png
    test/good/000.png
    test/broken_large/000.png
    ground_truth/broken_large/000_mask.png
  cable/
  ... 其余 13 类
```

`train/good` 只放正常图片；`test/good` 放独立的正常测试图片；其他 `test/<缺陷类型>` 放异常图片；`ground_truth/<同名缺陷类型>` 放相应像素掩码，使用原始尺寸，正常像素为 0，异常区域为非零。正常测试图不需要掩码。不要把同一张图片同时用于训练和测试。

首次使用官方真实图片，避免生成随机图片、复制一个类别填满 15 类或用空目录凑结构。那些做法无法验证数据语义及评估流程。

## 4. 构造首次测试 ZIP

仓库脚本 `scripts/prepare_mvtec_smoke.py` 从每类抽取 16 张正常训练图、4 张正常测试图，以及一种缺陷的 4 张异常图和 4 张对应掩码。总计 15 类、420 个文件。它检查目录和掩码配对，不改源数据，不覆盖现有 ZIP。

### 4.1 Windows 构造，浏览器上传

下面源路径是示例，替换成你实际解压后、直接包含 `bottle` 等 15 类的目录：

```powershell
Set-Location 'F:\推理训练平台'
conda run -n pytorch python -S scripts/prepare_mvtec_smoke.py 'D:\datasets\mvtec_anomaly_detection' 'D:\datasets\mvtec-smoke-v1.zip'
```

脚本仅使用 Python 标准库，`-S` 跳过本地第三方包初始化。成功时会打印 `15 categories, 420 files`。ZIP 内包含 `mvtec_smoke/<类别>/...`，平台会自动识别外层目录。Windows 构造后可直接在 Windows 浏览器上传，无需先把 ZIP 传到服务器。

### 4.2 Linux 构造

若完整数据已在服务器，huaciq 可运行：

```bash
cd /home/zxy/workspace/xunliantuili
backend/.venv/bin/python scripts/prepare_mvtec_smoke.py /home/huaciq/datasets/mvtec_anomaly_detection /home/huaciq/datasets/mvtec-smoke-v1.zip
```

浏览器文件选择器只能选择浏览器所在电脑的文件。如果 ZIP 在服务器而浏览器在 Windows，需要先通过你现有的文件传输方式下载到 Windows，或通过服务器上的认证 API 上传；不要直接复制文件到平台 `storage` 来代替导入。

## 5. 在网页上传数据集

1. 登录平台，创建项目 `INP-Former 验收`，进入项目。
2. 进入“数据集”，新建 `MVTec AD Smoke`。
3. 点击“上传新版本”，选择 `mvtec-smoke-v1.zip`，点击“上传并处理”。
4. 等待版本显示 `v1`、状态 `ready`、格式 `mvtec_ad`，文件数应为 420。

`ready` 表示解压和目录识别成功，不代表完整图像解码、标注语义或模型运行已通过。若格式是 `generic` 或 `classification`，先检查 ZIP 内类别层级和 `train/test/ground_truth` 结构，重新上传一个版本。

## 6. 准备用户算法代码包

平台不管理用户算法的 Git。你自行下载 [官方 INP-Former 代码](https://github.com/luow23/INP-Former)；下面的 clone 仅用于准备外部算法，不会加入平台仓库。

Linux，huaciq：

```bash
mkdir -p /home/huaciq/inpformer-acceptance
cd /home/huaciq/inpformer-acceptance
git clone https://github.com/luow23/INP-Former.git
cd INP-Former
git rev-parse HEAD
test -f INP_Former_Multi_Class.py
mkdir -p backbones/weights
curl -fL --retry 3 -o backbones/weights/dinov2_vitb14_reg4_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
ls -lh backbones/weights/dinov2_vitb14_reg4_pretrain.pth
sha256sum backbones/weights/dinov2_vitb14_reg4_pretrain.pth
```

记录官方代码提交和权重 SHA256，之后复现实验使用同一份代码包。下载失败时不要打包残缺权重；重新下载成功后继续。这里使用带 4 个 register tokens 的 ViT-B/14 权重，与网页编码器 `dinov2reg_vit_base_14` 对应，不要换成不带 `reg4` 的文件。

代码包必须包含全部算法模块，至少有：

```text
INP-Former/
  INP_Former_Multi_Class.py
  dataset.py
  utils.py
  optimizers.py
  aug_funcs.py
  models/
  dinov1/
  dinov2/
  beit/
  backbones/weights/dinov2_vitb14_reg4_pretrain.pth
```

打包时不包含 `.git`，保留已下载的权重：

```bash
cd /home/huaciq/inpformer-acceptance
tar --exclude='INP-Former/.git' --exclude='*/__pycache__' -czf inpformer-code-v1.tar.gz INP-Former
tar -tzf inpformer-code-v1.tar.gz | head -n 20
```

也可以在 Windows 下载、准备并压缩同样的目录。若包在 Linux，先传到浏览器所在电脑再网页上传。

网页进入项目“代码包”，新建 `INP-Former`，上传新版本，填写：

| 字段 | 值 |
| --- | --- |
| 默认工作目录 | `INP-Former` |
| 默认入口命令 | `python INP_Former_Multi_Class.py` |

这里的工作目录与上面的 TAR 根目录一致。如果你打包的是目录内容而非整个文件夹，工作目录应为 `.`。入口文件名使用下划线 `INP_Former`；应以实际文件名为准。上传后确认状态 `ready`。

## 7. 准备 INP-Former 环境镜像

本仓库未内置 INP-Former 镜像，不能只登记一个不存在的镜像名，也不能直接用 YOLO 镜像代替。

官方 requirements 使用 Torch 2.0.0/CUDA 11.8 和较旧的科学计算依赖；不能把其中的旧 NumPy/SciPy 固定版本直接安装进平台后端 Python 3.11。环境在独立 Docker 镜像中准备。

下面是供服务器验证的候选环境，保留官方 Torch 2.0.0/CUDA 11.8，科学计算包调整为该 Python 环境可用的版本。它不是作者原始环境的逐项复现。正式论文对照需要进一步锁定算法和依赖版本。

Linux，huaciq：

```bash
mkdir -p /home/huaciq/inpformer-acceptance/runtime
nano /home/huaciq/inpformer-acceptance/runtime/Dockerfile
```

文件内容：

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libxcb1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    "torch==2.0.0" "torchvision==0.15.1" \
    "numpy==1.24.4" "scipy==1.10.1" "pandas==1.5.3" \
    "scikit-learn==1.2.2" "scikit-image==0.21.0" \
    "matplotlib==3.7.5" "Pillow==9.5.0" "opencv-python==4.6.0.66" \
    "tabulate==0.9.0" "tqdm==4.66.5" "timm==0.9.12" \
    "kornia==0.7.3" "adeval==1.1.0" \
    && python -m pip check \
    && python -c "import torch, torchvision, cv2, timm, kornia, adeval"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/workspace/output XDG_CACHE_HOME=/workspace/output/.cache \
    MPLCONFIGDIR=/workspace/output/.cache/matplotlib
WORKDIR /workspace/code
USER 10001:10001
CMD ["python", "--version"]
```

镜像中的默认 UID 会被平台 `.env` 的 `DOCKER_USER` 覆盖，因此平台运行容器时应使用与输出目录匹配的 huaciq UID/GID。

构建：

```bash
docker build -t train-platform/inpformer:torch2.0-cu118-v1 /home/huaciq/inpformer-acceptance/runtime
docker image inspect train-platform/inpformer:torch2.0-cu118-v1 --format '{{.Id}}'
```

检查 GPU 及所有导入。先执行 `nvidia-smi --query-gpu=uuid,name --format=csv,noheader`，把下面 `GPU-实际3090的UUID` 替换成一张 RTX 3090 的实际 UUID，确认该卡当前没有平台任务在使用：

```bash
docker run --rm --gpus device=GPU-实际3090的UUID --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/.cache --env MPLCONFIGDIR=/tmp/matplotlib train-platform/inpformer:torch2.0-cu118-v1 python -c "import torch, torchvision, cv2, timm, kornia, adeval; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

再检查算法导入、入口和预训练权重的真实加载：

```bash
docker run --rm --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/.cache --env MPLCONFIGDIR=/tmp/matplotlib -v /home/huaciq/inpformer-acceptance/INP-Former:/workspace/code:ro -w /workspace/code train-platform/inpformer:torch2.0-cu118-v1 python INP_Former_Multi_Class.py --help
docker run --rm --user "$(id -u):$(id -g)" --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/.cache --env MPLCONFIGDIR=/tmp/matplotlib -v /home/huaciq/inpformer-acceptance/INP-Former:/workspace/code:ro -w /workspace/code train-platform/inpformer:torch2.0-cu118-v1 python -c "from models import vit_encoder; model = vit_encoder.load('dinov2reg_vit_base_14'); print(type(model).__name__)"
```

这些命令成功后再登记。失败时保留完整构建或导入错误，先解决依赖问题。不能把构建完成等同于 GPU 训练已经验证通过。

## 8. 在网页登记镜像

平台管理员进入“运行镜像”，登记：

| 字段 | 值 |
| --- | --- |
| 名称 | `INP-Former Torch2.0 CUDA11.8 v1` |
| Docker 镜像引用 | `train-platform/inpformer:torch2.0-cu118-v1` |
| 镜像摘要或版本标识 | 上一步 inspect 输出的 `sha256:...` |
| 框架标识 | `inpformer` |
| 允许创建任务 | 开启 |

摘要字段目前只是记录信息；真正执行使用镜像引用，所以实验期间不要用不同环境覆盖同一个 tag。停用原来的 `INP-Former Runtime (configure me)` 占位镜像，以免被自动选中。

## 9. 首次训练

项目 → “训练任务” → “新建训练”，填写：

| 字段 | 首次验收值 |
| --- | --- |
| 任务名称 | `inpformer-smoke-001` |
| 训练模板 | `INP-Former 多类别异常检测` |
| 数据集版本 | `MVTec AD Smoke v1 · mvtec_ad` |
| 代码版本 | `INP-Former v1` |
| 运行镜像 | 刚登记的 INP-Former 镜像 |
| GPU 型号 / 数量 | `RTX 3090` / `1 张` |
| 优先级 | `0` |
| 训练轮数 / 批量大小 | `1` / `4` |
| 输入尺寸 / 裁剪尺寸 | `448` / `392` |
| INP 数量 / 编码器 | `6` / `DINOv2 Reg ViT-B/14` |
| 结果名称 | `smoke001` |

点击“创建并提交”。平台会构造数据挂载路径和输出路径，不要在入口命令里重复写训练参数。

预期过程：排队、准备并启动容器、运行、最后成功。页面详情中的命令应包含 `--dataset MVTec-AD`、`--data_path /workspace/dataset/mvtec_smoke`、`--save_dir /workspace/output` 和 `--total_epochs 1`。

## 10. 查看日志与验收产物

页面日志应逐步显示：

- `train image number:240`，训练共 240 张正常图，batch=4 时通常 60 个训练 batch。
- 第一轮的 loss，随后逐类测试评估。
- 15 类的图像级和像素级 AUROC/AP/F1，以及像素 AUPRO、平均结果。

一轮训练结束后还要完成 15 类评估，任务才会退出。小数据的指标不用于论文结论。

Linux，huaciq 可以同时观察：

```bash
watch -n 2 nvidia-smi
docker ps --filter label=train-platform.managed=true
```

页面有“执行 ID”后，可查看同一个容器：

```bash
docker logs --tail 100 -f 实际执行ID
docker inspect 实际执行ID --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
```

验收成功需要同时满足：任务 `succeeded`、退出码 `0`、日志含评估结果、训练产物包含一个非空的 `.pth`，能够下载。官方保存目录名称会追加 dataset、encoder、尺寸等参数，产物可能显示为 `<长结果目录>/model.pth`，不一定叫 `best.pt`。

点击产物下载，再在 Windows 核对 SHA256：

```powershell
Get-FileHash 'C:\Users\DELL\Downloads\model.pth' -Algorithm SHA256
```

路径按实际下载位置调整。应与平台记录的完整 SHA256 对应；页面目前显示的是其前缀。若要检查 checkpoint 可反序列化，使用相同运行镜像在服务器加载，而不是要求 Windows 本地安装另一套训练依赖。

官方日志文件 `log.txt` 会写进输出目录，但当前产物收集器不收集 `.txt`；它通常只在容器日志或服务器输出目录可见。网页指标面板为空不代表训练失败。

## 11. 注册模型、停止和正式训练

成功任务页面点击“注册模型”，名称填 `INPFormer-MVTec-Smoke`、阶段选“候选”，确认项目“模型注册表”中出现 v1 和其来源任务。原始 checkpoint 下载与 ONNX 导出是不同操作，本次验收以 `.pth` 为交付物。

再新建一条相同配置但轮数为 10 的独立任务，命名 `inpformer-stop-test`。看到训练 batch 运行后点击“停止”，等待状态 `stopped`，确认容器不再运行、GPU 释放，之后能再次提交任务。不要用这条被中止的任务验收模型文件。

整套小数据流程通过后，将完整官方数据集上传为新的版本，选择它新建正式任务，保留模型原有输入/裁剪设置，并逐步调整 batch 和训练轮数。1 轮只是执行链路验收，正式轮数按固定版本的官方配置或实验方案设定。

如果后续使用实验室自定义类别，当前硬编码的 15 类校验和官方入口都需要适配。不要把自定义类别重命名成 MVTec 类别来绕过校验。

## 12. 常见故障定位

| 现象 | 首先检查 |
| --- | --- |
| 数据集选项为空 | 模板选对；版本 `ready` 且格式 `mvtec_ad` |
| missing category structure | 必须齐全 15 类，各有 train/test/ground_truth |
| Required backbone weight is missing | 权重是否位于所填工作目录下的 `backbones/weights`，且编码器匹配 |
| Entrypoint script not found | TAR 根目录、默认工作目录、下划线文件名是否一致 |
| pull access denied / image not found | 训练服务器是否已成功构建所登记的 tag |
| No module named ... / lib...so missing | 看镜像依赖检查；不要往后端虚拟环境安装算法依赖 |
| Permission denied | 代码/数据只读；结果应写 output；DOCKER_USER 是否有输出目录权限 |
| 卡在 queued | 是否存在可用且启用的 3090，是否被其他平台任务占用 |
| CUDA out of memory | 首先将 batch=4 调低为 2 或 1；以日志确定显存不足 |
| ExitCode=137 | 查看 OOMKilled；可能是宿主机/容器内存限制，不等同于显存不足 |
| test and ground truth pair | 每张异常图需要相应缺陷目录下的真实 mask |
| roc_auc / AUPRO 异常 | 每类测试要有正常和异常图，mask 必须有真实异常像素 |
| 训练完没有 pth | 是否完成最终评估；写入是否在 output；是否达到产物数量/大小限制 |

失败时记录任务 ID、运行命令、镜像 tag、代码版本、数据版本和首段完整 traceback。不要只提供日志最后一行。

## 13. 核对来源

- [INP-Former 官方仓库](https://github.com/luow23/INP-Former)：代码和环境说明。
- [官方多类别入口](https://github.com/luow23/INP-Former/blob/main/INP_Former_Multi_Class.py)：类别、训练、评估和保存行为。
- [官方编码器加载](https://github.com/luow23/INP-Former/blob/main/models/vit_encoder.py)：预训练权重位置和下载来源。
- [官方数据加载](https://github.com/luow23/INP-Former/blob/main/dataset.py)：图片和掩码目录。
- [官方依赖列表](https://github.com/luow23/INP-Former/blob/main/requirements.txt)：原始版本参考。
- [MVTec AD 下载页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads)：官方数据和目录含义。

平台行为核对自 `backend/app/api/runs.py`、`services/archive_ingest.py`、`services/scheduler.py`、`services/results.py`、`api/registry.py` 以及相应前端视图。
