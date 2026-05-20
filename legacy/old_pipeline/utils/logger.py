import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, base_log_dir="test", exp_name=None):
        # 1. 创建带时间戳的实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_log_dir, exp_name if exp_name else f"/run_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # 2. 配置标准 Logging
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler (防止多次实例化时日志重复记录)
        if not self.logger.handlers:
            log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            
            # 文件输出
            fh = logging.FileHandler(os.path.join(self.exp_dir, "train.log"))
            fh.setFormatter(log_format)
            self.logger.addHandler(fh)
            
            # 控制台输出
            ch = logging.StreamHandler()
            ch.setFormatter(log_format)
            self.logger.addHandler(ch)

        # 3. 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tb_logs"))
        
        self.info(f"Experiment initialized at: {self.exp_dir}")

    def info(self, message):
        self.logger.info(message)

    def log_metrics(self, epoch, metrics, step_type="Train"):
        """
        metrics: 字典类型, 如 {'loss': 0.5, 'f1': 0.8}
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"{step_type}/{name}", value, epoch)

    def save_checkpoint(self, model, is_best=False):
        import torch
        path = os.path.join(self.exp_dir, "latest_model.pt")
        torch.save(model.state_dict(), path)
        if is_best:
            best_path = os.path.join(self.exp_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            self.info(f"✔ Best model saved to {best_path}")

    def close(self):
        self.writer.close()