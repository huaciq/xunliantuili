# 点位 ONNX 转 NB

这里保存平台使用的 AMD64 宿主转换源码。版本为 `point-pcint8-rgb-u8-v2`，通过原有三参数入口接收 ONNX、数据集目录和 NB 输出路径。

校准和验证使用 Pillow 11.0.0 的 RGB 双线性缩放。`training_split.json` 与 ONNX 同目录，train 用于校准，val 用于逐图核对原 ONNX、SDK 浮点和 SDK INT8 的预测。SDK 输入张量、输出有限值和误差上限均通过后，再导出 NB 并检查实际二进制接口。

NB 输入为 UINT8 NCHW `[1,3,128,128]`，scale 为 1/255，zero point 为 0。输出为 FLOAT16 `[1,11]`。默认逐图概率偏差上限为 0.10，定位偏差上限为 3 像素（按 128×128 输入计）。配置文件中的 `validation_limits.max_probability_delta` 和 `validation_limits.max_coordinate_delta_px` 可调整这些上限。

完整逐图报告保存在 SDK 工作目录的 `quality_validation.json` 中。报告包括置信度过 0.5 的判断变化和类别变化。源码与本地回归已验证，真实 SDK 编译结果仍需在 Linux 转换服务器运行验收。

## 部署 zys 分支

服务器沿用已安装的官方 SDK Docker 镜像。安装脚本从 `/opt/glh-converter/wheelhouse` 读取已有离线依赖，从本仓库读取 Pillow wheel，为 `/opt/glh-converter-zys` 创建独立虚拟环境。

按项目现有部署路径执行以下命令。执行前应结束正在运行的转换任务。若服务器已有本地 Git 改动，应先保存这些改动。

```bash
cd /home/zxy/workspace/xunliantuili
git status --short --branch
git switch zys
git pull --ff-only origin zys
sudo python3 scripts/deploy_point_converter.py \
  --env-file /home/zxy/workspace/xunliantuili/.env
sudo systemctl restart train-platform-api
sudo systemctl is-active train-platform-api
curl --fail --silent --show-error http://127.0.0.1:8000/api/v1/health
```

服务器本地尚无 zys 分支时，可先执行 `git fetch origin zys` 和 `git switch --track origin/zys`。

安装脚本会将指定 `.env` 的转换工作目录设为 `/opt/glh-converter-zys`，转换命令保持三参数形式，并把转换超时增加到至少 1800 秒。旧转换源码和 `.env` 备份保存在该目录的 `deployment-backups/<时间>/` 下。部署脚本核对 API 账号能无密码调用 Docker，完成依赖检查和模块导入检查后才更新 `.env`。API 服务重启后加载这项配置。

如项目目录、服务名或运行账号不同，按实际实例填写。安装器支持 `--converter-root`、`--seed-root` 和 `--user`。这里的独立目录用于让 zys 实例选择本版本，运行镜像和离线依赖由已有服务器安装提供。

## 部署后的转换验收

使用此前同一份 ONNX 和数据版本重新转换。成功时检查新报告中的浮点及 INT8 偏差，再进行板端测试。如果转换被拒绝，可查看 `journalctl -u train-platform-api` 和 `/opt/glh-converter-zys/converter/sdk-workspace/workspace/training/converter-jobs/` 下对应任务的报告。

宿主逐图数值校验覆盖验证集中提供的输入。最终 NB 在目标板上的执行结果需要板端实测。
