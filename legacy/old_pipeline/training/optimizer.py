# Optimizer.py
import torch
from typing import Tuple

def create_optimizer(
    model: torch.nn.Module, 
    base_lr: float = 1e-5, 
    fusion_lr: float = 5e-4, 
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999)
) -> torch.optim.Optimizer:
    """
    构建分层学习率的优化器。
    
    策略：
    1. presence_fusion_weights: 使用较大的学习率 (fusion_lr)，且不使用权重衰减 (weight_decay=0)。
    2. 其他所有参数 (Backbone, Heads): 使用基础学习率 (base_lr) 和常规权重衰减。
    
    兼容 DataParallel (DP) 和 DistributedDataParallel (DDP)。
    """
    
    # 1. 处理多卡包装情况 (DDP/DP)
    # 如果模型被包裹在 DDP 或 DP 中，我们需要访问内部的 .module 来获取参数
    if hasattr(model, 'module'):
        model_to_optimize = model.module
    else:
        model_to_optimize = model

    # 2. 检查关键参数是否存在，防止报错
    if not hasattr(model_to_optimize, 'presence_fusion_weights'):
        raise AttributeError(
            "模型中未找到 'presence_fusion_weights' 属性。"
            "请确认 MultimodalBiasModel 初始化正确，或者该参数名称是否变更。"
        )

    # 3. 分离参数
    # 获取特殊的融合权重参数对象
    fusion_param = model_to_optimize.presence_fusion_weights
    
    # 获取其余所有参数 (通过对象身份 'is not' 排除融合权重)
    # 注意：named_parameters() 返回的是 (name, parameter) 元组
    other_params = [
        p for n, p in model_to_optimize.named_parameters() 
        if p is not fusion_param
    ]

    # 4. 构建参数组列表
    param_groups = [
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'backbone_and_heads'
        },
        {
            'params': [fusion_param], # 放入列表
            'lr': fusion_lr,
            'weight_decay': 0.0,      # 关键：不对少量融合权重做正则化
            'name': 'fusion_weights'
        }
    ]

    # 5. 创建优化器
    optimizer = torch.optim.AdamW(param_groups, betas=betas)


    return optimizer