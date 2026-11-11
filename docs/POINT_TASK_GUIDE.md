# ZDP 点位任务：从数据版本到真实 ONNX

## 1. 本次任务是什么

输入源为 `F:\zdp脚本`。原脚本是 MobileNetV2 的目标存在性、两类分类和单点坐标联合训练，不是 YOLO 检测，也不是 INP-Former 好坏异常检测。

- 200 张图片，200 份同名 TXT；空 TXT 为背景，非空格式 `class_id x y`，坐标归一化到 0～1。
- 背景 127 张，类别 0 为 41 张，类别 1 为 32 张；并非简单的好/坏标签。
- 按历史清单固定 160 张训练、40 张验证，不重新随机拆分。
- 全主干训练、batch=16、lr=0.0003、分类损失权重=0.3、seed=42；先 1 轮验收，再 10 轮基线。
- 保留算法的 ONNX 包装：置信度 sigmoid、两类概率 softmax，再补齐到 8 类概率，加上 x/y，总输出 11 维。

适配不修改原始 `pointModel-batch1.py`。额外的 `platform_train.py` 导入该脚本的定义，设置挂载路径和参数，然后调用原 `train_model`、原 `export_onnx`，最后用 ONNX Runtime 核验真实输出。原算法的网络、损失、优化器、早停和导出处理均保留。

本次环境基于平台 PyTorch 2.5.1/CUDA12.4、torchvision0.20.1，属于经本地验证的接入环境，不是历史说明中的 torch2.7.1/cu118 环境复刻。固定了数据成员不代表随机增强、训练结果与历史任务逐位相同；预先分目录后读取顺序也可能与原随机列表不同。输入预处理保留原脚本 RGB、128×128、除以255，不套用 ImageNet 标准化。

## 2. 准备文件（Windows）

本次生成目录：`F:\zdp脚本\platform_upload_v1`。若已有以下两个 ZIP，可以直接进入网页上传，无需再次运行脚本。

| 文件 | 用途 |
| --- | --- |
| `point-dataset-v1.zip` | `dataset200_point/train`、`val`，400 个图像/标注文件及划分清单 |
| `point-code-v1.zip` | 原训练脚本、适配入口、ImageNet V1 预训练权重和来源元数据 |

需要重建时，在新的输出目录执行（权重路径按实际缓存位置填写）：

```powershell
Set-Location 'F:\推理训练平台'
conda run -n pytorch python -S scripts/prepare_point_task.py 'F:\zdp脚本' 'F:\zdp脚本\platform_upload_v2' --weights 'C:\Users\DELL\.cache\torch\hub\checkpoints\mobilenet_v2-b0353104.pth'
```

打包脚本检查原数据 SHA256 和字节数、同名标注、一图一个点、坐标范围、划分覆盖和不重叠、跨 train/val 完全相同图片，以及官方预训练权重哈希前缀。未划分的数据和原脚本保持原样。清单只能排除完全重复，正式精度测试还要独立检查相近拍摄场景是否泄漏。

## 3. 构建环境（Linux，huaciq）

若当前为 root，先 `sudo -iu huaciq`。工作区有本地改动时先处理，不要覆盖。

```bash
cd /home/zxy/workspace/xunliantuili
git status --short --branch
git pull --ff-only origin main
docker build --progress=plain \
  --build-arg PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -f infra/training/point.Dockerfile \
  -t train-platform/point:128-v1 .
```

基础镜像 `train-platform/pytorch:development` 已在先前任务中准备；若不存在，先 `docker compose build training-pytorch`。本镜像补充 ONNX、ONNX Runtime CPU、OpenCV headless 等依赖，并使用修复元数据的 Ninja 1.11.1.2。构建阶段不会训练模型。

GPU 自检：

```bash
docker run --rm --gpus all train-platform/point:128-v1 \
  python -c 'import torch, torchvision, onnx, onnxruntime, cv2; print(torch.__version__, torchvision.__version__); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

Docker 镜像还需在此 Linux 服务器实际构建和运行验收，本地 Windows GPU 通过不等于 Linux 镜像已经通过。不需要迁移数据库或重启后端/Nginx，因为平台使用已有的自定义命令接口。

## 4. 网页登记及上传

以平台管理员进入「运行镜像」，登记：

| 字段 | 值 |
| --- | --- |
| 名称 | `Point MobileNetV2 128 v1` |
| Docker 镜像引用 | `train-platform/point:128-v1` |
| 版本标识 | `128-v1` |
| 框架标识 | `pytorch` |
| 允许创建任务 | 开启 |

新建项目 `ZDP 点位定位`，在项目内：

1. 数据集 → 新建 `Point200 fixed160-40` → 上传 `point-dataset-v1.zip`。
2. 等待 `v1` 为 `ready`，文件数为 401，格式预计为 `generic`，这对自定义命令正常。不要伪装成 YOLO 格式。
3. 代码包 → 新建 `Point Trainer` → 上传 `point-code-v1.zip`。
4. 默认工作目录填 `.`；默认入口填 `python platform_train.py`。代码在 ZIP 根目录，无额外外层文件夹。

不要上传旧 `run_train.sh` 作为平台入口：它的输出目标在代码目录，而代码挂载只读。也不要把整份原始目录作为一个混合代码包反复上传，数据和代码分别版本化。

## 5. 首次训练及导出

训练任务 → 新建训练，填写：

| 字段 | 值 |
| --- | --- |
| 名称 | `point-smoke-001` |
| 模板 | **自定义命令** |
| 数据版本 | `Point200 fixed160-40 v1` |
| 代码版本 | `Point Trainer v1` |
| 镜像 | `Point MobileNetV2 128 v1` |
| GPU | RTX3090，1张 |
| 附加参数 | `--epochs 1 --batch-size 16 --lr 0.0003 --cls-weight 0.3` |

入口默认从 `/workspace/dataset/dataset200_point` 读已划分目录，向 `/workspace/output` 写文件。它通过 Torch Hub 缓存读取随代码提供的 ImageNet V1 权重，不需要训练时下载。

日志分三段：`[stage] train`、`[stage] export`、`[stage] validate`。数据数目应为 train=160、val=40。训练每轮显示 Train/Val Loss，保存最佳验证损失的 checkpoint，然后使用该 checkpoint 导出 ONNX。

最后用 40 张验证图片比较 ONNX Runtime CPU 与含 sigmoid/softmax/补齐的 PyTorch 包装模型，检查有限值、输入输出名字和固定形状；逐元素容差 `atol=1e-4, rtol=1e-3`。比较的是同一张图片经相同预处理后的同一输出含义，而不是用原始 logits 比概率。

## 6. 验收输出

任务应 `succeeded`、退出码0，并出现：

| 产物 | 验收要点 |
| --- | --- |
| `mobilenetv2_point_128.pth` | 原训练器保存的最佳 checkpoint，包含模型/优化器状态和训练信息 |
| `dataset200_point.onnx` | 算法真实导出，输入 input[1,3,128,128]、输出 output[1,11]、opset11 |
| `onnx_validation.json` | status=passed、samples=40、实际最大绝对误差和逐图输出 |
| `results.csv` | 每轮 train/loss、val/loss；从原日志转录，精度为原打印的4位小数 |
| `training_split.json` | 原算法实际使用的容器内绝对路径，可与清单对应 |
| `split_manifest.json` | 源相对路径清单、原文件SHA及各划分的类别统计 |
| `run_summary.json` | 实际参数、依赖版本、GPU、最佳轮次、验证结果及源代码/权重SHA |

数值一致不代表生产精度达标。`val/loss` 是训练联合损失，不是准确率；当前没有独立 test 集或任务级业务精度报告。正样本定位误差、类别准确率、背景误报率需单独设计后续评估，不能由 ONNX 数值校验替代。

先下载 `.pth` 和 `.onnx`，再从成功任务注册候选模型。真实 ONNX 从本次任务产物下载；当前模型注册表的「ONNX导出」接口仍是占位实现，不要再次点它来覆盖或替代这里的真实文件。

## 7. 再跑10轮与问题排查

首次通过后，新建 `point-baseline-10ep`，仍选择同一数据、代码、镜像，参数改为 `--epochs 10 --batch-size 16 --lr 0.0003 --cls-weight 0.3`。创建新任务不会自动加载上一任务 checkpoint；每次从随包 ImageNet 预训练权重开始。

若导出或验证失败，整个任务会失败，不应标为成功。已完成的 checkpoint 仍可能留在宿主机的本次输出目录，但平台目前只自动采集成功任务的文件。用 huaciq 查看容器日志和 `.env` 的 STORAGE_ROOT 下 `runs/<项目ID>/<任务ID>/output`，不要删除并重新训练来掩盖导出错误。

常见故障：

- 预训练文件哈希不符：使用本次生成的完整代码包，不要拿另一个 MobileNet 权重改名。
- CUDA显存不足：先将 batch 降为8；显卡不支持或不可见先检查镜像GPU自检。
- 缺依赖：检查所选是否是 `point:128-v1`，不是 INP-Former 镜像。
- 写入拒绝：输出应在 output，代码/数据只读；核对服务器 DOCKER_USER 与存储权限。
- ONNX校验失败：记录原始错误和报告，核对 PyTorch、ONNX Runtime 版本和输入输出；不要直接跳过检查。

## 8. 后续统一的方向

当前按 [训练任务接入约定](TRAINING_TASK_CONTRACT.md) 接入。平台统一任务生命周期和外部交接，算法提供实际训练、划分语义和导出方法；此次没有在后端增加只为 ZDP 服务的专用分支。

如需网页分别启动“数据准备/训练/导出/验证”，下一步应开发通用阶段编排和结构化参数接口，并让算法提供相应入口。目前这些阶段仍在同一个训练容器中顺序执行。
