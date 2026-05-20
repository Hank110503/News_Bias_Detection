# Data

这个目录只放数据相关内容和少量数据代码，不再承接分析产物。

当前约定：

- `data/raw/`: 原始采集数据
- `data/processed/`: 处理中或清洗后的数据
- `data/visualization/`: 数据可视化脚本代码

说明：

- 由 `data/visualization/visualization.py` 生成的图表和汇总表，统一输出到 `outputs/analysis/data_visualization/`
- 一次性数据脚本已经收口到 `scripts/data/`
- 如果后续需要保留演示样例，建议新增 `data/samples/`
