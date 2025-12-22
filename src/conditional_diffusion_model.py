#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
条件扩散模型实现
基于预处理后的网络轨迹数据进行训练
严格按照条件扩散模型方案.md实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import warnings


class ConditionalUNet1D(nn.Module):
    """
    UNet1D with conditioning on:
      - 22 continuous features (Z-scored)
      - 1 discrete network state ID (embedded)
    Injects condition via `timestep_cond` using `addition_embed_type="text"`.
    """

    def __init__(
        self,
        sample_size: int = 100,  # 序列长度
        in_channels: int = 4,    # 输入通道数 [del_up, del_dn, loss_up, loss_dn]
        out_channels: int = 4,   # 输出通道数
        num_network_states: int = 8,
        state_embed_dim: int = 32,
        time_embedding_dim: int = 256,
        layers_per_block: int = 2,
        block_out_channels: tuple = (32, 64, 128, 256),
        down_block_types: tuple = (
            "DownBlock1D",
            "AttnDownBlock1D",
            "AttnDownBlock1D",
            "AttnDownBlock1D",
        ),
        up_block_types: tuple = (
            "AttnUpBlock1D",
            "AttnUpBlock1D",
            "AttnUpBlock1D",
            "UpBlock1D",
        ),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size
        self.time_embedding_dim = time_embedding_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim // 4, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # 创建时间嵌入的位置编码
        self.time_proj = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim // 4),
        )
        
        # Embed discrete state ID
        self.state_embed = nn.Embedding(num_network_states, state_embed_dim)

        # Project combined condition to addition_time_embed_dim
        cont_dim = 22  # 12 global + 10 local
        proj_in = cont_dim + state_embed_dim
        proj_out = time_embedding_dim

        self.add_embedding = nn.Sequential(
            nn.Linear(proj_in, proj_out),
            nn.SiLU(),
            nn.Linear(proj_out, proj_out)
        )
        
        # 简化的UNet结构
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        # 解码器
        self.dec3 = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(384, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(192, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, 3, padding=1)
        )
        
        # 下采样和上采样
        self.downsample = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 注意力机制
        self.attn_enc2 = Attention(128)
        self.attn_enc3 = Attention(256)
        self.attn_middle = Attention(256)
        self.attn_dec3 = Attention(256)
        self.attn_dec2 = Attention(128)

    def forward(self, sample: torch.Tensor, timestep, cond: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            sample: 输入张量 (B, 4, 100)
            timestep: 时间步
            cond: 条件向量 (B, 23)
        """
        # 时间嵌入
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embed(t_emb)
        
        # 条件嵌入
        if cond is not None:
            # Split cond: skip index 11 (state_id)
            global_part = torch.cat([cond[:, :11], cond[:, 12:13]], dim=-1)  # (B, 12)
            local_part = cond[:, 13:23]                                      # (B, 10)
            cont = torch.cat([global_part, local_part], dim=-1)               # (B, 22)

            # Safely convert state_id from float to integer
            state_id_raw = cond[:, 11]
            state_id = state_id_raw.long()  # truncate towards zero
            if not torch.allclose(state_id_raw, state_id.float(), atol=1e-5):
                warnings.warn(f"Non-integer network_state_id detected: {state_id_raw}")
            state_id = state_id.clamp(0, self.state_embed.num_embeddings - 1)
            state_emb = self.state_embed(state_id)

            timestep_cond = self.add_embedding(torch.cat([cont, state_emb], dim=-1))
        else:
            timestep_cond = None

        # 将时间条件与时间嵌入结合
        if timestep_cond is not None:
            t_emb = t_emb + timestep_cond

        # 编码器
        h1 = self.enc1(sample)  # (B, 64, 100)
        h1_down = self.downsample(h1)  # (B, 64, 50)
        
        h2 = self.enc2(h1_down)  # (B, 128, 50)
        h2 = self.attn_enc2(h2)
        h2_down = self.downsample(h2)  # (B, 128, 25)
        
        h3 = self.enc3(h2_down)  # (B, 256, 25)
        h3 = self.attn_enc3(h3)
        h3_down = self.downsample(h3)  # (B, 256, 12)
        
        # 中间层
        h_middle = self.middle(h3_down)  # (B, 256, 12)
        h_middle = self.attn_middle(h_middle)
        
        # 解码器
        h_middle_up = self.upsample(h_middle)  # (B, 256, 24)
        # 调整尺寸以匹配
        if h_middle_up.size(2) != h3.size(2):
            h_middle_up = F.interpolate(h_middle_up, size=h3.size(2), mode='nearest')
        h_dec3 = torch.cat([h_middle_up, h3], dim=1)  # (B, 512, 24) -> (B, 256, 24)
        h_dec3 = self.dec3(h_dec3)
        h_dec3 = self.attn_dec3(h_dec3)
        
        h_dec3_up = self.upsample(h_dec3)  # (B, 256, 48)
        # 调整尺寸以匹配
        if h_dec3_up.size(2) != h2.size(2):
            h_dec3_up = F.interpolate(h_dec3_up, size=h2.size(2), mode='nearest')
        h_dec2 = torch.cat([h_dec3_up, h2], dim=1)  # (B, 384, 48) -> (B, 128, 48)
        h_dec2 = self.dec2(h_dec2)
        h_dec2 = self.attn_dec2(h_dec2)
        
        h_dec2_up = self.upsample(h_dec2)  # (B, 128, 96)
        # 调整尺寸以匹配
        if h_dec2_up.size(2) != h1.size(2):
            h_dec2_up = F.interpolate(h_dec2_up, size=h1.size(2), mode='nearest')
        
        h_dec1 = torch.cat([h_dec2_up, h1], dim=1)  # (B, 192, 100) -> (B, 64, 100)
        output = self.dec1(h_dec1)  # (B, 4, 100)
        
        # 创建一个类似diffusers输出的对象
        class Output:
            def __init__(self, sample):
                self.sample = sample
        
        return Output(output)


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
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Attention(nn.Module):
    """
    简单的注意力机制
    """
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, N = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, N).permute(1, 0, 2, 3)  # (3, B, C, N)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, C, N)
        
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = self.proj(out)
        return out + x  # Residual connection


def validate_cond(cond: torch.Tensor, num_network_states: int = 8):
    """Validate cond tensor format"""
    if cond.dim() != 2 or cond.shape[-1] != 23:
        raise ValueError(f"Expected cond shape (B, 23), got {cond.shape}")
    
    state_id = cond[:, 11]
    if not torch.allclose(state_id, state_id.round(), atol=1e-5):
        warnings.warn(f"Non-integer network_state_id detected: {state_id.tolist()}")
    
    if (state_id < 0).any() or (state_id >= num_network_states).any():
        raise ValueError(f"network_state_id out of range [0, {num_network_states})")


# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = ConditionalUNet1D()
    
    print("条件扩散模型已创建")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    sample = torch.randn(batch_size, 4, 100)  # (B, channels, length)
    timestep = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, 23)
    
    # 设置网络状态ID为有效值
    cond[:, 11] = torch.randint(0, 8, (batch_size,)).float()
    
    with torch.no_grad():
        output = model(sample, timestep, cond)
        print(f"输入形状: {sample.shape}")
        print(f"输出形状: {output.sample.shape}")
        print("前向传播测试通过")