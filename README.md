# War News Bias Detection

这是一个用于检测战争新闻偏见的项目，使用多模态模型（文本和图像）来分析新闻的偏见程度。

## 项目结构

- `apps/`: 应用程序，包括标注工具和可视化工具
- `data/`: 数据集和可视化脚本
- `models/`: 模型定义，包括文本编码器、图像编码器和多模态融合
- `training/`: 训练相关代码
- `utils/`: 工具函数
- `evaluations/`: 评估脚本
- `notebook/`: Jupyter笔记本，用于实验和分析

## 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/Hank110503/News_Bias_Detection.git
   cd News_Bias_Detection
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装CLIP（如果需要）：
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## 使用

### 训练模型
```bash
python train.py
```

### 运行演示
```bash
python training/training_pipeline_demo.py
```

### 评估模型
```bash
python evaluations/evaluate_model.py
```

## 数据

数据文件夹包含原始数据和处理后的数据。请注意，原始数据和模型检查点不会上传到GitHub。

## 许可证

[添加许可证信息]