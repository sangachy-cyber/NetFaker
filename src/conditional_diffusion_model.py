#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
条件扩散模型实现
基于预处理后的网络轨迹数据进行训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码，用于时间步嵌入
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalDiffusionModel(nn.Module):
    """
    条件扩散模型
    
    Args:
        window_size: 窗口大小（默认100）
        feature_dim: 特征维度（del_up, del_dn, loss_up, loss_dn, gap = 5）
        cond_dim: 条件向量维度（23维）
        hidden_dim: 隐藏层维度
        timesteps: 扩散步骤数
    """
    
    def __init__(
        self, 
        window_size: int = 100,
        feature_dim: int = 5,  # timestamp, del_up, del_dn, loss_up, loss_dn, gap
        cond_dim: int = 23,
        hidden_dim: int = 128,
        timesteps: int = 1000
    ):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 条件向量嵌入
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 输入数据嵌入
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, window_size, feature_dim]
            t: 时间步 [batch_size]
            cond: 条件向量 [batch_size, cond_dim]
            
        Returns:
            输出数据 [batch_size, window_size, feature_dim]
        """
        batch_size = x.size(0)
        
        # 时间嵌入
        t_emb = self.time_mlp(t)  # [batch_size, hidden_dim]
        t_emb = t_emb.unsqueeze(1).repeat(1, self.window_size, 1)  # [batch_size, window_size, hidden_dim]
        
        # 条件向量嵌入
        cond_emb = self.cond_mlp(cond)  # [batch_size, hidden_dim]
        cond_emb = cond_emb.unsqueeze(1).repeat(1, self.window_size, 1)  # [batch_size, window_size, hidden_dim]
        
        # 输入数据嵌入
        x_emb = self.input_projection(x)  # [batch_size, window_size, hidden_dim]
        
        # 合并所有嵌入
        combined_emb = x_emb + t_emb + cond_emb  # [batch_size, window_size, hidden_dim]
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(combined_emb)  # [batch_size, window_size, hidden_dim]
        
        # 输出投影
        output = self.output_projection(transformer_out)  # [batch_size, window_size, feature_dim]
        
        return output


class DiffusionPipeline:
    """
    扩散过程管理器
    """
    
    def __init__(self, model: ConditionalDiffusionModel, timesteps: int = 1000):
        self.model = model
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """
        余弦调度计算beta值
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        正向扩散过程：向数据中添加噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple):
        """
        从一维张量a中提取t时间步的值，并调整形状以匹配x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def p_losses(self, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        计算训练损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, cond)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, cond: torch.Tensor):
        """
        逆向扩散过程：从噪声中恢复数据
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        # 预测噪声
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple, cond: torch.Tensor):
        """
        完整的逆向扩散采样过程
        """
        device = next(self.model.parameters()).device
        
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((shape[0],), i, device=device, dtype=torch.long), cond)
            imgs.append(img.cpu().numpy())
        
        return imgs
    
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, batch_size: int = 1):
        """
        从条件向量生成新样本
        """
        shape = (batch_size, self.model.window_size, self.model.feature_dim)
        imgs = self.p_sample_loop(shape, cond)
        return imgs[-1]


# 示例用法
if __name__ == "__main__":
    # 创建模型实例
    model = ConditionalDiffusionModel()
    diffusion = DiffusionPipeline(model)
    
    print("条件扩散模型已创建")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")