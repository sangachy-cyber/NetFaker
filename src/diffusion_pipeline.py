#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扩散过程管理器和训练流程
基于条件扩散模型方案.md实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
try:
    from conditional_diffusion_model import ConditionalUNet1D
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: ConditionalUNet1D not available")


class DDPMScheduler:
    """
    DDPM调度器实现
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 创建beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程中的常用项
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 计算后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        向原始样本添加噪声
        
        Args:
            original_samples: 原始样本 [B, C, L]
            noise: 噪声 [B, C, L]
            timesteps: 时间步 [B]
            
        Returns:
            加噪后的样本
        """
        # 获取对应时间步的系数
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1).unsqueeze(2)
        
        # 执行前向扩散过程: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_velocity(self, sample: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        计算速度 (velocity)
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].unsqueeze(1).unsqueeze(2)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1).unsqueeze(2)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor) -> Dict[str, torch.FloatTensor]:
        """
        执行一步去噪过程
        
        Args:
            model_output: 模型预测的噪声
            timestep: 当前时间步
            sample: 当前样本
            
        Returns:
            包含去噪后样本和其他信息的字典
        """
        t = timestep
        prev_t = t - 1
        
        # 获取alpha值
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 计算当前方差
        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]
        
        # 预测原始样本
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 计算预测噪声
        pred_epsilon = model_output
        
        # 计算后验均值系数
        posterior_mean_coef1 = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        posterior_mean_coef2 = (current_alpha_t ** (0.5) * beta_prod_t_prev) / beta_prod_t
        
        # 计算后验均值
        posterior_mean = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
        
        # 获取后验方差
        posterior_variance = self.posterior_variance[t]
        
        # 生成新样本
        if t > 0:
            noise = torch.randn_like(sample)
        else:
            noise = torch.zeros_like(sample)
            
        pred_prev_sample = posterior_mean + (posterior_variance ** 0.5) * noise
        
        return {
            "prev_sample": pred_prev_sample,
            "pred_original_sample": pred_original_sample
        }


class DiffusionTrainer:
    """
    扩散模型训练器
    """
    
    def __init__(self, model: ConditionalUNet1D, scheduler: DDPMScheduler, device: str = "cpu"):
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
    
    def sample(self, cond: torch.Tensor, num_inference_steps: int = 1000) -> torch.Tensor:
        """
        从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_inference_steps: 推理步数
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        # 设置调度器推理步数
        self.scheduler.num_inference_steps = num_inference_steps
        
        # 初始化噪声样本
        batch_size = cond.shape[0]
        sample = torch.randn(batch_size, 4, 100, device=self.device)
        
        # 迭代去噪
        for t in reversed(range(num_inference_steps)):
            # 获取模型预测
            model_output = self.model(sample, torch.full((batch_size,), t, device=self.device).long(), cond).sample
            
            # 执行去噪步骤
            step_result = self.scheduler.step(model_output, t, sample)
            sample = step_result["prev_sample"]
        
        return sample


# 示例用法
if __name__ == "__main__":
    print("扩散管道模块已加载")
    print(f"模型可用: {MODEL_AVAILABLE}")