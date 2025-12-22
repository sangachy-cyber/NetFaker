#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扩散模型采样器
基于条件扩散模型方案.md实现
"""

import torch
from typing import Dict, Any


class DiffusionSampler:
    """
    扩散模型采样器
    """
    
    def __init__(self, model, scheduler, device: str = "cpu"):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
    def sample(self, cond: torch.Tensor, num_inference_steps: int = 1000) -> torch.Tensor:
        """
        从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_inference_steps: 推理步数
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        # 初始化噪声样本
        batch_size = cond.shape[0]
        sample = torch.randn(batch_size, 4, 100, device=self.device)
        
        # 设置调度器推理步数
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 迭代去噪
        for t in self.scheduler.timesteps:
            # 获取模型预测
            model_output = self.model(sample, t, cond).sample
            
            # 执行去噪步骤
            sample = self.scheduler.step(model_output, t, sample).prev_sample
        
        return sample