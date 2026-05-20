# GitHub Upload Checklist

这个仓库在上传 GitHub 前，建议按下面的边界处理。

## 应该保留在仓库里的内容

- `src/`
- `configs/`
- `frontend/` 的源码与配置
- `scripts/`
- `docs/`
- `legacy/`
- 根目录兼容入口脚本
- `README.md`
- `requirements.txt`
- `requirements-train.txt`

## 不应该上传到 GitHub 的内容

- `data/raw/`
- `data/processed/`
- `checkpoints/`
- `outputs/`
- 本地历史实验运行结果目录，例如 `test/` 或 `legacy/test/`
- `frontend/public/demo/` 下的大型演示 JSON
- 本地环境目录：`myenv/`、`torch-gpu/`
- 本地工具状态：`.gradio/`、`.vscode/`

## 当前特别需要注意的点

1. `frontend/public/demo/eval_cases.json` 约 260 MB，不适合直接进普通 GitHub 仓库
2. 本地实验运行目录不属于这次 GitHub 上传范围
3. `.gradio/certificate.pem` 和 `.vscode/settings.json` 这类本地文件不应继续保留在版本历史中

## 上传前建议动作

1. 先执行 `git status`
2. 确认删除项主要是旧实验日志、旧可视化产物、旧 notebook 和旧训练管线文件
3. 确认未跟踪文件主要是新结构下的 `src/`、`configs/`、`docs/`、`scripts/`、`frontend/`、`legacy/`
4. 不要把本地数据、模型、日志和前端大演示 JSON 加入版本控制

## 推荐补充

- 如果要公开发布，补一个明确的 `LICENSE`
- 如果要让别人复现，补一个最小样例数据说明或 `data/samples/`
- 如果前端演示必须保留，建议把 `frontend/public/demo/` 换成小样本文件
