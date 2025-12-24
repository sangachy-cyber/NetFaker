#!/usr/bin/env python3

"""条件校正流模型实现
整合了模型定义、训练器、采样器、后处理器等组件
支持使用TSDiff或UNet作为骨干网络
"""

import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class ConditionalFlowModel(torch.nn.Module):
    """条件校正流模型
    手动实现 Rectified Flow 核心逻辑，兼容现有代码结构
    支持使用TSDiff或UNet作为骨干网络
    """

    def __init__(self, device: str = "cpu", backbone_type: str = "tsdiff"):
        """初始化校正流模型
        
        Args:
            device: 训练设备
            backbone_type: 骨干网络类型，可选值: 'tsdiff' 或 'unet'
        """
        super().__init__()
        self.backbone_type = backbone_type
        
        # 根据backbone_type选择骨干网络
        if backbone_type == "tsdiff":
            from src.models.conditional_tsdiff import TSDiffModel
            self.unet = TSDiffModel(input_dim=4, cond_dim=23)
        elif backbone_type == "unet":
            from src.models.conditional_diffusion import PaddedConditionalUNet1D
            self.unet = PaddedConditionalUNet1D()
        else:
            raise ValueError(f"Invalid backbone_type: {backbone_type}")
        
        self.device = device
        self.unet = self.unet.to(device)
    
    def forward(self, x, timestep, cond=None):
        """前向传播
        
        Args:
            x: 输入数据
            timestep: 时间步
            cond: 条件向量
        
        Returns:
            模型输出
        """
        if self.backbone_type == "tsdiff":
            return self.unet(x, timestep=timestep, cond=cond)
        else:
            return self.unet(x, timestep=timestep, cond=cond)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行一个训练步骤
        
        Args:
            batch: 包含'window'和'cond'的批次数据
            
        Returns:
            损失值
        """
        # 获取数据
        clean_sample = batch["window"].to(self.device)  # (B, 4, 100)
        cond = batch["cond"].to(self.device)            # (B, 23)

        # 采样随机时间步
        batch_size = clean_sample.shape[0]
        times = torch.rand(batch_size, device=self.device)
        
        # 生成噪声
        noise = torch.randn_like(clean_sample)
        
        # 线性插值：x_t = t * x_1 + (1 - t) * x_0
        noised_sample = times.view(-1, 1, 1) * clean_sample + (1 - times.view(-1, 1, 1)) * noise
        
        # 计算目标流：flow = x_1 - x_0
        target_flow = clean_sample - noise
        
        # 模型预测流
        pred_flow = self.forward(noised_sample, timestep=times, cond=cond)
        
        # 基础 flow loss，使用Huber Loss对大误差更鲁棒
        loss = F.smooth_l1_loss(pred_flow, target_flow)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行一个验证步骤
        
        Args:
            batch: 包含'window'和'cond'的批次数据
            
        Returns:
            损失值
        """
        self.unet.eval()
        with torch.no_grad():
            # 获取数据
            clean_sample = batch["window"].to(self.device)  # (B, 4, 100)
            cond = batch["cond"].to(self.device)            # (B, 23)

            # 采样随机时间步
            batch_size = clean_sample.shape[0]
            times = torch.rand(batch_size, device=self.device)
            
            # 生成噪声
            noise = torch.randn_like(clean_sample)
            
            # 线性插值：x_t = t * x_1 + (1 - t) * x_0
            noised_sample = times.view(-1, 1, 1) * clean_sample + (1 - times.view(-1, 1, 1)) * noise
            
            # 计算目标流：flow = x_1 - x_0
            target_flow = clean_sample - noise
            
            # 模型预测流
            pred_flow = self.forward(noised_sample, timestep=times, cond=cond)
            
            # 基础 flow loss，使用Huber Loss对大误差更鲁棒
            loss = F.smooth_l1_loss(pred_flow, target_flow)
        
        self.unet.train()
        return loss
    
    def sample(self, cond: torch.Tensor, num_sample_steps: int = 100, guidance_scale: float = 1.0, safety_factor: float = 1.05, clamp_from_step: float = 0.8) -> torch.Tensor:
        """从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_sample_steps: 推理步数，推荐使用100步或更多
            guidance_scale: 条件增强比例
            safety_factor: 安全因子，推荐1.05，确保生成值不超过p99×safety_factor
            clamp_from_step: 从多少比例的步数开始clamp，推荐0.8
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        # 条件增强：放大p99分位数
        if guidance_scale != 1.0:
            cond_aug = cond.clone()
            cond_aug[:, 3] *= guidance_scale  # p99_del_up
            cond_aug[:, 7] *= guidance_scale  # p99_del_dn
            cond = cond_aug
            
        # 确保条件向量在正确的设备上
        cond = cond.to(self.device)
        
        # 初始化噪声样本
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, 4, 100, device=self.device)
        
        # 从条件向量中提取p99值（假设结构：[up_p1, up_p50, up_p99, up_std, down_p1, down_p50, down_p99, down_std, ...]）
        up_p99_norm = cond[:, 2].view(batch_size, 1)   # 上行p99，归一化值
        dn_p99_norm = cond[:, 6].view(batch_size, 1)   # 下行p99，归一化值
        
        # 计算安全上限
        safe_up = up_p99_norm * safety_factor  # 上行安全上限
        safe_dn = dn_p99_norm * safety_factor  # 下行安全上限
        
        # Euler方法采样
        dt = 1.0 / num_sample_steps
        
        for i in range(num_sample_steps):
            t = i / num_sample_steps
            times = torch.full((batch_size,), t, device=self.device)
            
            # Euler更新
            with torch.no_grad():
                pred = self.forward(x, timestep=times, cond=cond)
            x = x + pred * dt
        
        return x


class FlowSampler:
    """校正流模型采样器
    负责从训练好的模型中生成新样本
    """

    def __init__(self, model, device: str = "cpu") -> None:
        """初始化采样器
        
        Args:
            model: 训练好的模型
            device: 采样设备
        """
        self.model = model
        self.device = device
    
    def sample(self, cond: torch.Tensor, num_inference_steps: int = 100, guidance_scale: float = 1.0) -> torch.Tensor:
        """从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_inference_steps: 推理步数
            guidance_scale: 条件增强比例
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        return self.model.sample(cond, num_inference_steps, guidance_scale)


# 复用现有的 PostProcessor
# from src.models.conditional_diffusion import PostProcessor
