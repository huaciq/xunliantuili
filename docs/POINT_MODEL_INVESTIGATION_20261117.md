# 振动盘点位模型差异排查（2026-11-17）

已在本地复现一项平台适配差异：训练/验证图片成员不变，但原始平铺数据经 `train_test_split` 得到的列表顺序，被分目录后的文件名排序替换。保留相同 seed 无法弥补索引对应图片的改变。本次修复入口恢复 `source_split` 的原始顺序，原训练脚本保持不变。

线上异常任务的 ONNX、checkpoint、实际参数和推理样本尚未提供。下面的实验证明了适配差异及其修复，不能据此认定线上异常的全部原因，也没有完成 NB 板端精度验收。

## 1. 源码和数据检查

- 原始代码目录：`F:\zdp脚本`；训练器：`pointModel-batch1.py`。
- 原始训练器、`platform_upload_v1/point-code-v1.zip` 内的训练器及 `platform_validation_v1/code` 内的训练器，SHA256 均为 `3dd6d624e03372de3ecd08a2fe9b9a18a494317db2b4900847cf8864dd04d133`。
- 旧平台入口在仓库、ZIP、本地验证目录中一致，SHA256 为 `f88ac981b56b631d13af581235742463baf2374db0769d0d47d55276885032d5`。
- 随包 ImageNet V1 权重 SHA256 为 `b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`。
- 原始清单的 400 个图片/标注文件全部通过大小和 SHA256 核对；原始目录、数据 ZIP 和本地平台验证目录的 400 个文件内容完全一致。
- 原始脚本使用 RGB、PIL resize 到 128×128、除以 255；`MEAN=[0,0,0]`、`STD=[1,1,1]`。平台没有额外套用 ImageNet 标准化。
- 训练/验证成员均与原始 seed=42 的划分一致，分别 160/40 张；训练列表有 160/160 个位置改变，验证列表有 38/40 个位置改变。

训练索引的前几张图：

| 流程 | 前五张训练图 |
| --- | --- |
| 原始平铺目录 | auto_250、auto_97、auto_178、auto_156、auto_330 |
| 旧平台分目录排序 | auto_102、auto_103、auto_104、auto_105、auto_106 |

`DataLoader(shuffle=True)` 打乱的是索引。初始索引顺序改变后，同一个随机排列会选择不同图片，ColorJitter 随机数也会作用于不同图片，后续优化轨迹随之改变。

## 2. 三组 10 轮对照实验

三组均调用同一份未经修改的原训练器，使用同一台 GTX 1650、PyTorch 2.5.1、torchvision 0.20.1、同一 ImageNet V1 权重、同一组 200 张图片及标注。参数统一为 epochs=10、batch=16、lr=0.0003、cls_weight=0.3、num_classes=2、seed=42，全主干训练，CUDA AMP。

为隔离读取顺序的影响，三组统一设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`cudnn.benchmark=False`、`cudnn.deterministic=True`、`torch.use_deterministic_algorithms(True)`。这是本地受控实验设置，区别于默认启用 benchmark 的运行环境。

| 流程 | 最佳轮次 | 最佳验证损失 | 与原始流程相同的模型状态张量 |
| --- | ---: | ---: | ---: |
| 原始平铺目录 | 9 | 0.021369842812418938 | 318/318 |
| 旧平台分目录排序 | 10 | 0.05311569944024086 | 0/318 |
| 分目录数据恢复原始顺序 | 9 | 0.021369842812418938 | 318/318 |

原始流程与恢复顺序后的 ONNX SHA256 均为 `4177065b8af980bbe35171686f6b4356c24e09027990f51422795973e7e4f069`；40 张验证图片的 11 维逐元素输出差异为 0。旧平台排序方式导出的 SHA256 为 `fe4baa92b8b2c6ff55d320c7cd5e65a900ad2ec203db8944ee1757ee593c4e36`，与原始模型最大输出差异为 0.7995504。

使用同样的 40 张验证图片、置信度阈值 0.5：

| 指标 | 原始顺序 | 旧平台排序 | 恢复原始顺序 |
| --- | ---: | ---: | ---: |
| 正样本检出数 / 正样本数 | 17/18 | 18/18 | 17/18 |
| 背景误检数 / 背景数 | 0/22 | 1/22 | 0/22 |
| 全部正样本分类正确率 | 100% | 94.44% | 100% |
| 全部正样本平均点位误差（128×128 输入像素） | 5.7703 | 6.2366 | 5.7703 |

排序变化会改变模型行为；这组结果也没有显示所有指标都沿同一方向变化。40 张验证图来自既有容量测试样本，不能作为生产精度估计。此处分类及点位指标覆盖全部有标注的正样本，包括低于置信度阈值的样本。

## 3. 训练轮次、导出与数据类型

三个入口的参数来源不同，核对历史正常模型时要先确认实际启动命令：

| 入口 | 默认轮次 | 类别数 |
| --- | ---: | ---: |
| 直接运行原 `pointModel-batch1.py` | 55 | 8 |
| 原目录附带 `run_train.sh` | 显式指定 10 | 显式指定 2 |
| 旧/新平台入口 | 1（链路验证） | 固定 2 |

本地现存 `platform_validation_v1/output` 的 `run_summary.json` 和 checkpoint 确认只训练了 1 轮。用该 checkpoint 重新调用原始导出函数，得到的 ONNX 与已有平台 ONNX 字节相同，SHA256 均为 `b969a1cc8850933bfed69e4cdb465f6aa28fbbbffe50945cf1e67121e7ab19cf`。40 张验证图片上：

- 两份 ONNX 的最大输出差异为 0。
- ONNX 与含 sigmoid、softmax、补齐类别的 PyTorch 包装模型最大差异为 `1.2107261054827845e-06`。
- 阈值 0.5 下，该 1 轮模型漏检全部 18 张正样本；分类正确率 50%，平均点位误差 20.3566 输入像素。

因此，本地这份模型存在识别质量问题，但同 checkpoint 的导出数值保持一致。`onnx_validation.status=passed` 仅验证导出一致性，不能当作精度合格。

本次检查的 ONNX 实际输入类型为 `tensor(float)`，输出类型为 `tensor(double)`。原始代码重新导出和平台导出相同；旧检查只验证形状及数值，没有记录输出类型。新报告补充输入/输出实际类型。生产推理程序应按模型实际类型读取输出；是否与历史正常 ONNX 有类型差异，需要拿到那份文件后实测。

平台模型注册表的 `POST /model-versions/{id}/exports` 仍生成 `FAKE_ONNX` 开头的开发占位文件。点位模型应使用训练任务产物中的真实 ONNX。新增比较工具会识别这种占位文件。

## 4. 本次修复及使用

涉及 `scripts/run_point_task.py`、`scripts/prepare_point_task.py`：

1. 原训练器创建 DataLoader 后，按数据清单 `source_split` 恢复图片及对应标注的原始索引，再开始迭代。保留原 DataLoader、增强、模型、损失、优化器、早停及导出函数。
2. 校验清单成员与实际数据完全一致，缺失、重复或成员变化直接报错；全部划分通过校验后再应用顺序。
3. `training_split.json` 记录修复后实际顺序，NB 校准仍可追溯到实际训练图片。
4. 包元数据增加入口 SHA256；新入口核对该 SHA。`run_summary.json` 增加 `data_order`、入口 SHA、类别数、seed、Pillow 版本；ONNX 验证报告增加实际数据类型。
5. 没有历史清单的自备分目录数据仍使用文件名顺序，并明确记录 `data_order=directory_sorted`，不声称与某个历史训练过程等价。

修复后的上传代码包位于：

```text
F:\zdp脚本\platform_upload_v2_orderfix_20261117\point-code-v1.zip
```

文件名沿用打包器的命名；请在平台将它作为新的代码版本上传，原数据版本可以继续使用。仅拉取 Git 不会替换已有的上传代码版本，也不会改变旧模型。新任务如需对照附带训练命令，附加参数应显式填写：

```text
--epochs 10 --batch-size 16 --lr 0.0003 --cls-weight 0.3
```

历史正常模型若使用其他轮次、类别数或数据集，应按该模型的实际参数重新对齐；当前入口的两类设置不能代表任意八类训练。

## 5. 正常模型与异常模型的逐图比较

新增 `scripts/compare_point_onnx.py`。它将同一份 PIL RGB、双线性 resize、float32 NCHW /255 输入同时送入两份模型，报告真实输入/输出类型、SHA256、11 维原始输出、置信度变化、类别变化和点位偏移。图片目录旁有 `labels` 时自动读取；缺失标签会报错，避免误当作背景。

Windows 示例（替换两份模型的实际路径）：

```powershell
conda run -n pytorch --no-capture-output python scripts/compare_point_onnx.py `
  --reference 'F:\模型对照\正常.onnx' `
  --candidate 'F:\模型对照\平台.onnx' `
  --images 'F:\zdp脚本\platform_validation_v1\data\dataset200_point\val\images' `
  --threshold 0.5 `
  --output 'F:\模型对照\comparison.json'
```

依赖为 numpy、Pillow、onnxruntime，点位训练镜像已包含。输出报告必须是新路径。模型不同是正常的比较结果，不会仅因为差异超出容差就将命令标为失败；`all_outputs_close` 和业务指标分别记录。输出语义应为 `[confidence, class0..class7, x, y]`，两份模型都应使用固定输入 `[1,3,128,128]`。

本地详细实验产物在 `F:\zdp脚本\investigation_20261117`，包含 `static_audit.json`、`model_audit.json`、`controlled_training.json`、`fixed_package_verification.json`、各组 checkpoint/ONNX、训练日志及 `comparison_*.json`。模型二进制和实验数据没有加入 Git。

## 6. NB 及线上仍需核对的证据

远程 `origin/main` 的 `00fc5ec` 将代码树恢复到 `15ae4ff`，不包含 NB 接口；NB 集成保留在 `origin/zys`。本次检查了该分支的 API/校准数据准备代码以及原转换包，不切换或合并这两个分支的功能。部署前先确认服务器的分支，避免把训练入口更新误当作 NB 功能发布。

现有 NB 集成按原训练划分准备图片，并调用 `/opt/glh-converter` 的转换器。原转换包会比较 ONNX 输出头清理前后的数值，但 float32/quantized SDK sanity inference 只运行首张图片，没有对整套验证集设置量化误差或业务精度门槛。API 转换成功也不能代替板端精度验收。本地未找到这次线上异常 NB 和正常 NB，尚不能判定其量化或板端输入输出问题。

后续需要：

- 用户确认表现正常的 ONNX，以及本次平台真实训练产物 ONNX。
- 对应任务的 `.pth`、`run_summary.json`、`training_split.json`、`onnx_validation.json`、`results.csv`、实际代码版本和训练命令。
- 同一批实际推理图片或 ROI、推理端的 RGB/BGR、resize/裁剪、归一化、输出数据类型和 11 维字段解析方式。
- 若 ONNX 同图结果符合预期而 NB 异常，再检查该 NB 的 `conversion_report.json`、`nbg_meta.json`、校准清单及板端输出，分离训练、量化、输入预处理和输出读取的问题。

## 7. 修复包验证

- 点位接入及比较工具的针对性测试：11 项通过；修改涉及的 Python 文件通过 Ruff。
- 使用实际打包后的 `platform_train.py`，保留默认 CUDA benchmark 设置，实际完成 10 轮训练、原始导出及 40 张验证图的 PyTorch/ONNX 一致性检查，最大输出差异 `2.3180889e-06`。实际 `training_split.json` 恢复源清单顺序，`run_summary.json` 记录 `data_order=source_split`。该次最佳验证损失为 `0.05236037001013756`，不能将受控实验的损失直接套用到默认非确定性训练。
- 再以与三组对照实验相同的确定性设置，调用这个实际打包入口完成 10 轮训练：最佳轮次 9、最佳验证损失 `0.021369842812418938`，318/318 个模型状态张量与原始平铺流程一致，ONNX SHA256 同为 `4177065b8af980bbe35171686f6b4356c24e09027990f51422795973e7e4f069`。这验证了交付包中的实际修复入口。
- 本次保留运行环境默认的 CUDA benchmark 设置，没有为追求某次损失数值而修改原训练算法。相同数据顺序是复现所需条件之一，跨运行的确定性仍需单独控制。
