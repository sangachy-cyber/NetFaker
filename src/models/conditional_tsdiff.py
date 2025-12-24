#!/usr/bin/env python3

"""条件TSDiff模型实现
基于Fourier MLP的时间序列扩散模型，支持条件注入
"""

import numpy as np
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入
    
    Args:
        dim: 嵌入维度
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """前向传播
        
        Args:
            x: 时间步张量 (B,)
            
        Returns:
            位置嵌入 (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TSDiffModel(nn.Module):
    """基于MLP的条件时间序列扩散模型
    
    实现了一个简单的MLP模型，支持条件注入，适合保留尖峰特征
    采用Scale-Conditioned Output Head，显式将p99作为尺度因子注入输出层
    
    Args:
        input_dim: 输入通道数，默认4（del_up, del_dn, loss_up, loss_dn）
        cond_dim: 条件向量维度，默认23
        hidden_dim: 隐藏层维度，默认512
    """
    
    def __init__(self, input_dim=4, cond_dim=23, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 条件投影（关键！）
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        else:
            self.cond_proj = None

        # 主干 MLP（输出两倍：residual 和 log_scale）
        self.main = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim * 2)  # 输出两倍：[residual, log_scale]
        )

        # p99在条件向量中的位置
        self.up_p99_idx = 3   # 上行p99位置（正确索引）
        self.dn_p99_idx = 7   # 下行p99位置（正确索引）
        
        # 可学习的scale_alpha参数
        self.scale_alpha = nn.Parameter(torch.tensor(1.1))

    def forward(self, x, timestep, cond=None):
        """前向传播
        
        Args:
            x: 输入数据 (B, C, T) - e.g., (B, 4, 100)
            timestep: 扩散时间步 (B,) - 范围 [0, 1]
            cond: 全局条件向量 (B, cond_dim) - 包含p1, p50, p99, std等
            
        Returns:
            预测的噪声或流 (B, C, T)
        """
        B, C, T = x.shape
        assert C == self.input_dim, f"输入通道不匹配: {C} vs {self.input_dim}"

        # 展平为 (BT, C)
        x_flat = x.permute(0, 2, 1).reshape(B * T, C)  # (BT, C)

        # 时间嵌入 -> (B, hidden_dim)
        timestep_emb = self.time_mlp(timestep)  # (B, hidden_dim)
        timestep_emb = timestep_emb.repeat_interleave(T, dim=0)  # (BT, hidden_dim)

        # 条件嵌入（加到时间嵌入上）
        if cond is not None and self.cond_proj is not None:
            assert cond.shape[1] == self.cond_dim, f"条件维度不匹配: {cond.shape[1]} vs {self.cond_dim}"
            c_emb = self.cond_proj(cond)  # (B, hidden_dim)
            c_emb = c_emb.repeat_interleave(T, dim=0)  # (BT, hidden_dim)
            h_global = timestep_emb + c_emb  # (BT, hidden_dim)
        else:
            h_global = timestep_emb

        # 拼接输入
        h = torch.cat([x_flat, h_global], dim=-1)  # (BT, C + hidden_dim)

        # 主干网络
        out_flat = self.main(h)  # (BT, 2*C)

        # 拆分输出
        residual_flat = out_flat[:, :C]      # (BT, C) - 标准化的扰动模式
        log_scale_flat = out_flat[:, C:]     # (BT, C) - 对数尺度

        # 从条件向量中提取p99值
        up_p99 = cond[:, self.up_p99_idx]  # (B,)
        dn_p99 = cond[:, self.dn_p99_idx]  # (B,)
        
        # 扩展到BT长度
        up_p99_expanded = up_p99.repeat_interleave(T)  # (BT,)
        dn_p99_expanded = dn_p99.repeat_interleave(T)  # (BT,)

        # 构造scale向量（对每个通道）
        scale_vec = torch.ones_like(residual_flat)  # (BT, C)
        scale_vec[:, 0] = up_p99_expanded * self.scale_alpha  # 上行时延用up_p99缩放
        scale_vec[:, 1] = dn_p99_expanded * self.scale_alpha  # 下行时延用dn_p99缩放
        # loss通道（通道2和3）保持scale=1

        # 最终输出 = residual * scale
        out_flat_scaled = residual_flat * scale_vec

        # 重塑回 (B, C, T)
        out = out_flat_scaled.reshape(B, T, C).permute(0, 2, 1)  # (B, C, T)
        return out