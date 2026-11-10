# 在平台训练三个拉链异常检测模型

三个任务共用一个代码版本和一个运行镜像，但分别读取自己的数据集、从同一 DINOv2 编码器开始训练、各自输出一个模型。正常图用于学习正常外观；异常图及留出的正常图用于图像级评估。不需要像素掩码，不提供像素级评估，也不使用既有的 `model.zip`。

## 1. 上传数据（Windows 浏览器）

这次已在 `F:\zipper_pictures\platform_upload_v1` 生成三个 ZIP。若要在另一台 Windows 机器重建，使用 `conda run -n pytorch python -S scripts/prepare_zipper_datasets.py <源目录> <新的输出目录>`，源目录应包含下表三套原始数据。脚本不会改动原图。

| 网页数据集名称 | ZIP | 正常训练 / 正常测试 / 异常测试 |
| --- | --- | --- |
| zipper | `zipper-v1.zip` | 62 / 26 / 100 |
| Zipperback | `Zipperback-v1.zip` | 27 / 7 / 16 |
| Zipperfront | `Zipperfront-v1.zip` | 19 / 5 / 5 |

登录平台 → 新建项目 `拉链异常检测` → 数据集 → 新建数据集 → 上传新版本。分别创建并上传上述三个 ZIP；每个版本状态应为 `ready`、格式应为 `classification`。ZIP 根目录直接是 `train/good`、`test/good` 和 `test/bad`，附一份 `split_manifest.json`。

拆分留出较晚的正常帧，训练与正常测试之间隔开 12 帧，并排除与任意异常图共享帧号的正常训练候选。保留全部异常图用于初步评估；异常帧仍可能来自同一段视频并与训练帧相邻。正式验收应新增独立批次或视频，在独立数据上定阈值和复测，尤其是正面数据只有 5 张异常图。

## 2. 构建训练镜像（Linux，huaciq）

在 Linux 已登录 `root` 时先执行 `sudo -iu huaciq`，随后：

```bash
cd /home/zxy/workspace/xunliantuili
git status --short --branch
git pull --ff-only origin main
docker image inspect train-platform/pytorch:development --format '{{.Id}}'
```

若基础镜像不存在，先执行 `docker compose build training-pytorch`。然后：

```bash
docker build -f infra/training/inpformer.Dockerfile -t train-platform/inpformer:zipper-v1 .
docker run --rm --gpus all train-platform/inpformer:zipper-v1 python -c 'import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

若构建日志在 `pip install` 阶段显示 `Network is unreachable` 或 `Read timed out`，可只为这次构建换用服务器可访问的 HTTPS PyPI 源。例如：[清华 TUNA 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)：

```bash
docker build --progress=plain \
  --build-arg PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -f infra/training/inpformer.Dockerfile -t train-platform/inpformer:zipper-v1 .
```

构建默认的 pip 网络超时为 60 秒，可用 `--build-arg PIP_TIMEOUT=90` 调整。用同一个 Docker 构建器重试时可复用 pip 缓存，无需清理缓存或加 `--no-cache`。若两个源都无法访问，检查 Docker 构建环境的外网或实验室代理配置。

若已经构建了之前手册中的 INP-Former 镜像，可以复用，但需要通过上面的 GPU 检查和下一节的实际代码/权重加载检查，镜像名替换为真实 tag。不要选 YOLO 镜像或 `configure-me` 占位镜像。

打开平台网页，以**平台管理员**进入「运行镜像」→「登记运行镜像」。名称填 `INP-Former Zipper v1`，Docker 镜像引用填 `train-platform/inpformer:zipper-v1`，框架标识填 `inpformer`，打开「允许创建任务」。版本标识可填 `zipper-v1`。不需要重启 API 或 Nginx。

## 3. 准备代码包（Windows PowerShell）

如果 Windows 的 `F:\zipper_pictures\platform_upload_v1` 中已有 `inpformer-zipper-code-v1.tar`，可直接跳到本节末尾上传。以下命令仅用于重建代码包，将 [官方 INP-Former 代码](https://github.com/luow23/INP-Former)、平台入口脚本和 [DINOv2 Reg ViT-B/14 官方预训练权重](https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub) 打包。数据或模型不会上传到 Git。重建时用没有同名 `INP-Former` 目录的新工作目录：

```powershell
New-Item -ItemType Directory -Path 'F:\zipper_pictures\code_rebuild'
Set-Location 'F:\zipper_pictures\code_rebuild'
git clone https://github.com/luow23/INP-Former.git INP-Former
git -C .\INP-Former checkout --detach 17d265381d9b323a2ef6e05aab0665a85edebe84
git -C .\INP-Former rev-parse HEAD
Copy-Item 'F:\推理训练平台\scripts\train_inpformer_image_labels.py' '.\INP-Former\'
New-Item -ItemType Directory -Path '.\INP-Former\backbones\weights' -Force
curl.exe -fL --retry 3 -o '.\INP-Former\backbones\weights\dinov2_vitb14_reg4_pretrain.pth' 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth'
```

先确认 `curl.exe` 成功退出，权重文件完整，再执行：

```powershell
Get-FileHash '.\INP-Former\backbones\weights\dinov2_vitb14_reg4_pretrain.pth' -Algorithm SHA256
tar.exe -cf inpformer-zipper-code-v1.tar --exclude=INP-Former/.git --exclude=*/__pycache__ INP-Former
```

若 Windows 下载有网络问题，可在 Linux 以 `huaciq` 在自己的目录做同样准备，入口脚本从 `/home/zxy/workspace/xunliantuili/scripts/train_inpformer_image_labels.py` 复制，再把生成的 TAR 传到浏览器所在电脑。权重下载失败时不要打包空文件、截断文件或改用旧的训练产物。记录算法提交和权重 SHA256，供复现。

如果代码同时在 Linux 可用，建议在创建平台任务前检查编码器可加载。命令里的 `/home/huaciq/INP-Former` 换成实际算法目录：

```bash
docker run --rm -v /home/huaciq/INP-Former:/workspace/code:ro -w /workspace/code -e MPLCONFIGDIR=/tmp/mpl train-platform/inpformer:zipper-v1 python -c "from models import vit_encoder; print(vit_encoder.load('dinov2reg_vit_base_14').embed_dim)"
```

网页进入同一项目 → 代码包 → 新建 `INP-Former Zipper` → 上传新版本 → 选 `inpformer-zipper-code-v1.tar`。按上面命令打包时，**默认工作目录**填 `INP-Former`，**默认入口命令**填 `python train_inpformer_image_labels.py`。等待 `ready`。

## 4. 提交三个任务（网页）

项目 → 训练任务 → 新建训练。先提交 `zipper-smoke`，确认通过后再提交实际训练任务。

| 字段 | 首次验收 |
| --- | --- |
| 训练模板 | **自定义命令** |
| 数据集版本 | `zipper v1` |
| 代码版本 | `INP-Former Zipper v1` |
| 运行镜像 | `INP-Former Zipper v1` |
| GPU 型号 / 数量 | RTX 3090 / 1 张 |
| 附加命令参数 | `--epochs 1 --batch-size 2 --input-size 224` |

现有「INP-Former 多类别异常检测」模板专用于 15 类 MVTec AD，此处使用「自定义命令」。平台自动挂载 `/workspace/dataset` 和 `/workspace/output`，无需在网页填写宿主机路径。224 可被 ViT 的 patch size 14 整除；拉链图使用完整视野，不额外中心裁剪。后续可按缺陷大小对比 336 或 448，注意显存和输入缩放。

验收通过后，新建三个任务，分别选择 `zipper v1`、`Zipperback v1`、`Zipperfront v1`，建议任务名 `inpformer-zipper`、`inpformer-zipper-back`、`inpformer-zipper-front`。其余配置相同，参数可先用 `--epochs 30 --batch-size 2 --input-size 224` 作为初始基线。每个任务只申请一张卡，有空闲 3090 的任务会启动，其余排队。第一次基线不应用测试集反复挑参数或生产阈值。

## 5. 下载、注册与排错（网页）

任务详情应出现 GPU、数据数量和每轮 `loss`。`succeeded`、退出码 0 后，查看“训练产物”：

- `model.pth`：训练后的模型权重，下载时按任务分别保存，避免三个同名文件混淆。推理需匹配编码器、INP 数量及输入预处理；`metrics.json` 记录这些参数。
- `metrics.json`：图像级 AUROC、AP 以及训练、测试数量。没有像素掩码，因此不报告像素级 AUROC 或 AUPRO。
- `predictions.csv`：每张测试图的标签和异常分数；`results.csv`：逐轮损失。下载模型后可以点击「注册模型」，分别注册为候选模型。当前平台的 ONNX 按钮仍是占位功能。

若任务未启动，先看“失败原因”及“执行 ID”。`image not found` 核对服务器 Docker tag；`FileNotFoundError` 核对 `backbones/weights`；`CUDA out of memory` 将 batch 降到 1。平台未报告成功前不能认为模型已产出。
