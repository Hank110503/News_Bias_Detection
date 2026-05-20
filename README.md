# War News Bias Detection

用于战争新闻偏见检测与分析的多模态项目，当前同时包含：

- 现役 Python 主线：`src/bias_vl/`
- 配置文件：`configs/`
- 前端与桌面壳：`frontend/`
- 历史训练管线：`legacy/old_pipeline/`

## 当前主线

当前建议维护和继续开发的代码路径是：

- `src/bias_vl/`
- `configs/`
- `frontend/`

根目录下的这些脚本目前保留为兼容入口，真实实现位于 `src/bias_vl/`：

- `train_stage1.py`
- `train_stage2.py`
- `infer.py`
- `infer_stream.py`
- `evaluate.py`
- `eval_stage1_batch.py`
- `review_stage1.py`
- `filter_complete_cues.py`
- `generate_stage2_predicted_obs.py`
- `convert_final_label.py`
- `schema.py`

## 历史代码

旧版训练管线已经归档到 `legacy/old_pipeline/`。这一部分保留是为了：

- 回溯早期实验实现
- 兼容旧训练入口 `python train.py`
- 对比新旧训练流程

如果你要继续迭代当前项目，优先使用 `src/bias_vl/`，不要基于旧管线继续扩展。

## 目录说明

- `configs/`: 当前主线训练和评估配置
- `data/`: 原始数据、处理中数据、样例与数据脚本
- `docs/`: 产品方案、实验整理、训练文档、规划文档
- `frontend/`: React + Vite + Tauri 前端
- `legacy/`: 历史实现归档
- `scripts/`: 一次性工具脚本和本地诊断脚本
- `src/`: 当前主线 Python 包

说明：此前仓库里存在过 `apps/`、`evaluations/` 等目录，但当前工作区中它们已经不在顶层现役结构里。

## 安装

基础依赖：

```bash
pip install -r requirements.txt
```

训练环境依赖：

```bash
pip install -r requirements-train.txt
```

如果需要本地前端：

```bash
cd frontend
npm install
```

## 常用命令

当前主线：

```bash
python train_stage1.py
python train_stage2.py
python infer.py
python evaluate.py
```

旧版训练入口：

```bash
python train.py
```

前端：

```bash
cd frontend
npm run dev
```

一次性工具脚本：

```bash
python scripts/data/clean_bias_labels.py
python scripts/data/check_data_integrity.py
python scripts/diagnostics/check_net.py
python scripts/diagnostics/test_pytorch.py
```

数据收集阶段 notebooks：

```bash
scripts/notebooks/
```

## 文档

- 规划与整理：[docs/plans/](/opt/data/private/war_news_bias_detection/docs/plans)
- 训练文档：[docs/training/](/opt/data/private/war_news_bias_detection/docs/training)
- 实验整理：[docs/experiments/](/opt/data/private/war_news_bias_detection/docs/experiments)
- 产品方案：[docs/product/](/opt/data/private/war_news_bias_detection/docs/product)

## 数据与产物

仓库里存在大量本地数据、训练输出和分析导出物。当前已经通过 `.gitignore` 尽量隔离这些生成物，但目录本身仍在本地保留，后续建议继续收敛到统一的产物目录。

当前约定：

- `data/visualization/` 只保留可视化脚本代码
- `outputs/analysis/data_visualization/` 存放该脚本生成的图表和汇总表
- `data/README.md` 记录 `data/` 目录的角色边界
- 历史实验运行结果保留在本地目录中，不纳入当前 GitHub 版本
