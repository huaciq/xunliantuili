# 点位 ONNX 转 NB：原版转换流程

按用户要求恢复同学提供的 `amd64_conversion_only_20261112` 原版，版本为 `point-pcint8-rgb-u8-v1`。恢复来源是第一次修改前的备份，各文件已与原始包的 `SHA256SUMS.txt` 核对。文件校验记录见 `original_source.json`。

`convert_nb.py`、`point_quantization.py`、`sdk_point_entry.py` 与原始文件逐字节一致。平台只在 Docker 启动和容器清理入口把 `sudo -n docker` 改为 `docker`，使用 API 服务已有的 Docker 用户组权限。三参数调用协议、路径隔离和 SDK 工作目录机制均沿用原版。

## 原版的转换配置

- OpenCV 解码与 `INTER_LINEAR` 缩放至 128×128，RGB、NCHW、1/255 归一化。
- 从 `training_split.json` 的 train 列表读取校准图，保持训练与验证划分分离。
- 原版 ONNX 输出头等价整理与 ONNX Runtime 等价检查。
- 官方 `ubuntu-npu:v2.0.10.2` 镜像，ACUITY 逐通道 INT8，`--compute-entropy --rebuild`。
- 原版 UINT8 输入参数、`force_float32=True` 输出配置及 `VIP9000NANODI_PLUS_PID0X1000003B` 导出目标。
- 导出前各执行一次浮点及量化试运行，核对 NB 存在、体积与原版元数据接口要求后发布文件。

本次撤回后来新增的 Pillow 预处理、逐图误差门槛和自动混合量化，恢复原版的完整行为。旧的 `validation_limits` 配置不参与原版转换。转换成功表示原版流程执行完成，精度需要使用相同输入在目标板上复测。

## 部署 zys 分支

执行前结束正在运行的转换任务。在当前服务器 root 终端执行：

```bash
cd /home/zxy/workspace/xunliantuili &&
sudo -u huaciq git pull --ff-only origin zys &&
/usr/bin/python3 scripts/deploy_point_converter.py --env-file .env &&
systemctl restart train-platform-api
```

安装成功会打印 `POINT_CONVERTER_ORIGINAL_IMPORT_OK`，并核对已安装源码的校验值。转换目录仍为 `/opt/glh-converter-zys`，离线依赖仍从 `/opt/glh-converter/wheelhouse` 获取。旧源码、被撤回的附加模块和 `.env` 保存在 `deployment-backups/<时间>/`，安装检查失败会恢复。

```bash
curl --noproxy '*' --retry 10 --retry-connrefused --retry-delay 2 -fsS http://127.0.0.1:8000/api/v1/health
```

若服务器访问 GitHub 失败，可用本次提交的 Git bundle 更新 zys 分支，再执行相同部署命令。安装器也支持通过 `--converter-root`、`--seed-root`、`--user` 指定其他实例。

## 验证范围

本地核对原版来源校验值、Docker 权限适配的全部差异、OpenCV 输入一致性、原版 SDK 调用顺序、导出失败处理，以及部署备份和回滚。真实 SDK 转换和目标板精度仍需在服务器及目标板验收。转换报告为原版 `conversion_report.json`，服务器 SDK 工作目录为 `/opt/glh-converter-zys/converter/sdk-workspace/workspace/training/converter-jobs/`。

## 平台转换进度

平台在原转换请求运行期间读取现有转换日志，每秒更新训练详情和项目上传入口中的阶段进度条，显示准备数据、模型导入、量化校准、试运行和生成 NB 等阶段及实际已用时。长阶段保持当前百分比并显示处理中；百分比表示阶段进度，不预测剩余时间。只有 NB 文件生成并通过原有检查后才显示完成，失败和超时保留错误状态。

进度功能只修改平台 API 和前端；本目录的转换主程序、预处理、量化和 SDK 导出文件保持原版校验值。进度记录位于转换工作目录的 `converter/tasks/.progress`，仅请求创建者可读取，已完成记录保留 24 小时。进度读取或写入失败不取消转换。

已运行原版转换器的服务器只需更新平台、重启 API 并构建发布前端，无需再次安装转换器或修改 `.env`：

```bash
cd /home/zxy/workspace/xunliantuili &&
sudo -u huaciq git pull --ff-only origin zys &&
sudo -iu huaciq bash -lc 'cd /home/zxy/workspace/xunliantuili/frontend && npm run build' &&
rsync -a frontend/dist/ /var/www/train-platform/ &&
systemctl restart train-platform-api
```

健康检查返回正常后，在浏览器刷新页面，再发起转换即可看到进度条。
