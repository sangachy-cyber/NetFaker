#!/usr/bin/env python3

"""CSDI模型实现
条件分数扩散模型，用于网络轨迹生成
基于CSDI（Conditional Score-based Diffusion Models for Imputation）框架
"""

import torch
import torch.nn as nn
from einops import rearrange


class TimeEmbedding(nn.Module):
    """时间嵌入层
    
    将时间步嵌入到高维空间
    """
    
    def __init__(self, hidden_dim):
        """初始化时间嵌入层
        
        Args:
            hidden_dim: 嵌入维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
    
    def forward(self, t):
        """前向传播
        
        Args:
            t: [B, L] 时间步
        
        Returns:
            [B, L, H] 时间嵌入
        """
        t = t.unsqueeze(-1).float()  # [B, L, 1]
        return self.mlp(t)  # [B, L, H]


class CSDIModel(nn.Module):
    """条件分数扩散模型
    
    用于网络轨迹生成的CSDI模型
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        cond_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """初始化CSDI模型
        
        Args:
            input_dim: 输入通道数（4个变量：up_delay, dn_delay, up_loss, dn_loss）
            cond_dim: 条件向量维度（23维）
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.value_emb = nn.Linear(1, hidden_dim)  # 值嵌入
        self.time_emb = TimeEmbedding(hidden_dim)  # 时间嵌入
        self.feature_emb = nn.Embedding(
            input_dim, hidden_dim
        )  # 特征嵌入，区分4个变量
        
        # 条件投影层
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        
        # 输出头
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        values: torch.Tensor,
        time_stamps: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            values: [B, K, L] 输入值，K=4变量，L=序列长度（包括虚拟观测点）
            time_stamps: [B, L] 时间戳
            mask: [B, K, L] 掩码（1=已观测，0=待生成）
            cond: [B, cond_dim] 条件向量
        
        Returns:
            [B, K, L] 预测的噪声
        """
        B, K, L = values.shape
        
        # 展平为 [B, K, L, 1]
        x = values.unsqueeze(-1)
        
        # 嵌入
        value_emb = self.value_emb(x)  # [B, K, L, H]
        time_emb = self.time_emb(time_stamps).unsqueeze(1)  # [B, 1, L, H]
        feature_ids = torch.arange(K, device=x.device).view(1, K, 1)  # [1, K, 1]
        feature_emb = self.feature_emb(feature_ids)  # [1, K, 1, H]
        
        # 合并嵌入
        h = value_emb + time_emb + feature_emb  # [B, K, L, H]
        # 重排为 [B*K, L, H]，便于Transformer处理
        h = rearrange(h, "b k l h -> (b k) l h")
        
        # 条件嵌入
        cond_emb = self.cond_proj(cond)  # [B, H]
        # 扩展为 [B*K, 1, H]
        cond_emb = cond_emb.repeat_interleave(K, dim=0).unsqueeze(1)
        
        # Cross-attention融合条件
        h = h + cond_emb
        
        # Transformer编码
        h = self.encoder(h)  # [B*K, L, H]
        
        # 输出预测
        out = self.head(h)  # [B*K, L, 1]
        # 重排回 [B, K, L]
        out = rearrange(out, "(b k) l 1 -> b k l", b=B, k=K)
        
        return out
