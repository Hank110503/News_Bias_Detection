# Scripts

这里存放一次性工具脚本和本地诊断脚本，不放主线训练、推理、评估入口。

当前结构：

- `scripts/data/`: 数据清洗、完整性检查等数据工具
- `scripts/diagnostics/`: 网络、环境、PyTorch 可用性等本地诊断工具
- `scripts/notebooks/`: 数据收集阶段留下的 notebook 脚本

使用原则：

- 主线代码放 `src/bias_vl/`
- 兼容入口保留在仓库根目录
- 一次性脚本和本地诊断脚本统一收口到这里
