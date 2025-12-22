#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扩散模型训练器
基于条件扩散模型方案.md实现
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any


class DiffusionTrainer:
    """
    扩散模型训练器
    """
    
    def __init__(self, model, scheduler, device: str = "cpu"):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        执行一个训练步骤
        
        Args:
            batch: 包含'window'和'cond'的批次数据
            
        Returns:
            损失值
        """
        # 获取数据
        window_data = batch['window'].to(self.device)  # (B, 100, 5)
        cond = batch['cond'].to(self.device)           # (B, 23)
        
        # 转换window数据格式: 从(B, 100, 5)到(B, 4, 100)，忽略时间戳列
        # 列顺序：`[ts, del_up, del_dn, loss_up, loss_dn]` -> `[del_up, del_dn, loss_up, loss_dn]`
        sample = window_data[:, :, 1:].transpose(1, 2)  # (B, 4, 100)
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (sample.shape[0],), 
            device=self.device
        ).long()
        
        # 生成噪声
        noise = torch.randn_like(sample)
        
        # 添加噪声
        noisy_samples = self.scheduler.add_noise(sample, noise, timesteps)
        
        # 模型预测
        model_output = self.model(noisy_samples, timesteps, cond).sample
        
        # 计算损失
        loss = F.mse_loss(model_output, noise)
        
        return loss