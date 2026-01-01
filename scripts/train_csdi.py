#!/usr/bin/env python3

"""CSDI模型训练与测试脚本
整合到现有项目流程中
"""

import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Tuple
from einops import rearrange

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子，确保结果可复现
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None


class TimeEmbedding(nn.Module):
    """时间嵌入层
    
    将时间步嵌入到高维空间
    """
    
    def __init__(self, hidden_dim: int):
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
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
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
        cond_dim: int = 15,  # 15维条件向量（1 ID + 14 分位点）
        hidden_dim: int = 64,  # 降低隐藏层维度，简化模型
        num_layers: int = 2,  # 减少Transformer层数，简化模型
        num_heads: int = 4,  # 减少注意力头数，简化模型
        dropout: float = 0.1,
        num_behavior_ids: int = 8  # 最大行为ID数量，最多8个
    ):
        """初始化CSDI模型
        
        Args:
            input_dim: 输入通道数（4个变量：up_delay, dn_delay, up_loss, dn_loss）
            cond_dim: 条件向量维度（15维）
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout率
            num_behavior_ids: 最大行为ID数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        
        # 嵌入层
        self.value_emb = nn.Linear(1, hidden_dim)  # 值嵌入
        self.time_emb = TimeEmbedding(hidden_dim)  # 时间嵌入
        self.feature_emb = nn.Embedding(
            input_dim, hidden_dim
        )  # 特征嵌入，区分4个变量
        
        # 添加behavior ID嵌入层
        self.behavior_id_emb = nn.Embedding(num_behavior_ids, hidden_dim)
        
        # 分布嵌入：使用排序+差分+MLP方案
        # 每个分布（上行/下行）的特征：原始值(7) + 差分(6) + 均值(1) + 极差(1) + 标准差(1) = 16
        # 上下行共32，加上行为ID嵌入(hidden_dim)，总输入32 + hidden_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),  # 行为ID嵌入 + 32维分布特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 添加可学习的无条件嵌入，用于处理CFG的无条件分支
        self.uncond_embedding = nn.Parameter(torch.randn(hidden_dim))
        
        # 简化Transformer编码器
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
            cond: [B, cond_dim] 条件向量 - 15维：1行为ID + 14分位点
        
        Returns:
            [B, K, L] 预测的噪声
        """
        B, K, L = values.shape
        
        # Debug: Check if cond is being used
        if torch.rand(1) < 0.05:  # 5% chance to print debug info for better visibility
            print(f"[Debug] CSDIModel.forward - cond: {cond[0].tolist()}")
            print(f"[Debug] CSDIModel.forward - cond shape: {cond.shape}")
        
        # 展平为 [B, K, L, 1]
        x = values.unsqueeze(-1)
        
        # 嵌入
        value_emb = self.value_emb(x)  # [B, K, L, H]
        time_emb = self.time_emb(time_stamps).unsqueeze(1)  # [B, 1, L, H]
        
        # 创建feature_ids，使用values的设备
        feature_ids = torch.arange(K, device=values.device).view(1, K, 1)  # [1, K, 1]
        feature_emb = self.feature_emb(feature_ids)  # [1, K, 1, H]
        
        # 条件嵌入：分离处理behavior ID和分位点条件
        # 检查是否是无条件请求（使用特殊标记-100.0）
        is_uncond = torch.all(cond == -100.0)  # 检查所有元素是否都是-100.0
        
        if is_uncond:
            # 使用可学习的无条件嵌入
            batch_size = cond.shape[0]
            cond_emb = self.uncond_embedding.unsqueeze(0).expand(batch_size, -1)  # [B, H]
            cond_emb = cond_emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
        else:
            # 1. 提取behavior ID（第0维）和分位点特征（1-14维）
            behavior_id = cond[:, 0].long()  # [B]
            
            # 实现分布嵌入：排序 + 差分 + MLP方案
            # 1. 提取上行和下行分位点（各7个）
            # cond维度：15 = 1行为ID + 7上行分位点 + 7下行分位点
            up_quantiles = cond[:, 1:8]      # [B, 7] 上行分位点：p1, p10, p25, p50, p75, p90, p99
            dn_quantiles = cond[:, 8:15]     # [B, 7] 下行分位点：p1, p10, p25, p50, p75, p90, p99
            
            # 2. 分布特征编码函数
            def encode_distribution(q):
                # q: [B, 7] 分位点值
                # 1. 原始值（已排序）
                raw = q  # [B, 7]
                # 2. 差分特征：相邻分位点的差值，反映局部密度
                diffs = torch.diff(q, dim=-1)  # [B, 6]
                # 3. 全局统计特征
                mean_q = q.mean(dim=-1, keepdim=True)      # [B, 1] 均值
                range_q = (q[:, -1] - q[:, 0]).unsqueeze(-1)  # [B, 1] 极差
                std_q = q.std(dim=-1, keepdim=True)        # [B, 1] 标准差
                
                # 拼接所有特征：7原始 + 6差分 + 1均值 + 1极差 + 1标准差 = 16维
                dist_feat = torch.cat([raw, diffs, mean_q, range_q, std_q], dim=-1)  # [B, 16]
                return dist_feat
            
            # 3. 编码上行和下行分布
            up_dist_feat = encode_distribution(up_quantiles)  # [B, 16]
            dn_dist_feat = encode_distribution(dn_quantiles)  # [B, 16]
            
            # 4. 合并上下行分布特征
            dist_feat = torch.cat([up_dist_feat, dn_dist_feat], dim=-1)  # [B, 32]
            
            # 5. 行为ID嵌入
            behavior_emb = self.behavior_id_emb(behavior_id)  # [B, H]
            
            # 6. 拼接行为ID嵌入和分布特征，然后通过MLP投影
            full_cond_feat = torch.cat([behavior_emb, dist_feat], dim=-1)  # [B, H + 32]
            cond_emb = self.cond_proj(full_cond_feat)  # [B, H]
            cond_emb = cond_emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
            # 移除额外的条件嵌入权重，由guidance_scale单独控制条件引导强度
        
        # Debug: Check cond_emb
        if torch.rand(1) < 0.05:  # 5% chance to print debug info
            print(f"[Debug] CSDIModel.forward - cond_emb shape: {cond_emb.shape}")
            print(f"[Debug] CSDIModel.forward - cond_emb mean: {cond_emb.mean().item():.4f}")
            print(f"[Debug] CSDIModel.forward - cond_emb std: {cond_emb.std().item():.4f}")
        
        # 合并所有嵌入，确保条件嵌入有足够的影响
        # 将条件嵌入添加到所有其他嵌入中，增强其影响力
        # 首先将条件嵌入与时间嵌入、特征嵌入深度融合
        combined_emb = time_emb + feature_emb + cond_emb
        # 然后与值嵌入相加，并再次添加条件嵌入，确保条件信息占主导地位
        # 增加条件嵌入的权重，确保条件信息被充分利用
        h = value_emb + combined_emb + cond_emb * 2.0  # [B, K, L, H]，增加条件嵌入权重
        
        # 重排为 [B*K, L, H]，便于Transformer处理
        h = rearrange(h, "b k l h -> (b k) l h")
        
        # Transformer编码
        h = self.encoder(h)  # [B*K, L, H]
        
        # 输出预测
        out = self.head(h)  # [B*K, L, 1]
        # 重排回 [B, K, L]
        out = rearrange(out, "(b k) l 1 -> b k l", b=B, k=K)
        
        return out


class SimpleSDE:
    """简单的Variance Preserving SDE
    
    实现了方差保持SDE的边际分布和去噪步骤
    """
    
    def __init__(self, beta_min: float = 0.05, beta_max: float = 10.0):
        """初始化SDE
        
        Args:
            beta_min: 最小噪声强度
            beta_max: 最大噪声强度
        """
        self.beta_min = beta_min  # 降低最小噪声强度，适合uniform空间
        self.beta_max = beta_max  # 降低最大噪声强度，减少扩散过程的极端值
    
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算边际分布
        
        Args:
            x0: [B, K, L] 初始值
            t: [B] 时间步
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 边际分布的均值和标准差
        """
        # 数值稳定计算：添加clamp防止log_mean_coeff过于极端
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        # 添加数值稳定性约束
        log_mean_coeff = torch.clamp(log_mean_coeff, min=-20.0, max=20.0)  # 防止指数溢出
        
        mean = torch.exp(log_mean_coeff).unsqueeze(-1).unsqueeze(-1) * x0
        # 添加小epsilon防止sqrt(0)问题
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff).clamp(min=0.0) + 1e-8).unsqueeze(-1).unsqueeze(-1)
        return mean, std
    
    def denoise_step(self, x: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """基于score的Probability Flow ODE去噪步骤
        
        Args:
            x: [B, K, L] 当前值
            pred_noise: [B, K, L] 预测的噪声
            t: [B] 当前时间步
            next_t: [B] 下一个时间步
            deterministic: 是否使用确定性采样，True则不添加随机项
        
        Returns:
            torch.Tensor: 更新后的值
        """
        # 计算当前时间步的参数，添加数值稳定性约束
        log_mean_coeff_t = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff_t = torch.clamp(log_mean_coeff_t, min=-20.0, max=20.0)  # 防止指数溢出
        
        alpha_t = torch.exp(2.0 * log_mean_coeff_t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        # 添加数值稳定性约束，确保alpha_t在合理范围内
        alpha_t = torch.clamp(alpha_t, min=1e-8, max=1.0 - 1e-8)
        # 添加小epsilon防止除以零
        std_t = torch.sqrt(1. - alpha_t + 1e-8)  # 当前时间步的标准差
        
        # 计算score：-pred_noise / std_t（根据SDE理论，score是负的噪声除以标准差）
        score = -pred_noise / (std_t + 1e-8)  # 添加epsilon防止除以零
        
        # 计算β(t) = β_min + t*(β_max - β_min)
        beta_t = (self.beta_min + t * (self.beta_max - self.beta_min)).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        beta_t = torch.clamp(beta_t, min=1e-8)  # 防止β(t)过小
        
        # 计算时间步长 dt = next_t - t
        dt = (next_t - t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # Probability Flow ODE 更新：dx = -0.5 * β(t) * (x + score) * dt
        x_next = x - 0.5 * beta_t * (x + score) * dt
        
        # 只有非确定性模式下才加入少量随机性
        if not deterministic and next_t.min() > 0:
            # 计算扩散项，添加数值稳定性约束
            diffusion_coeff = torch.sqrt(beta_t)
            noise = torch.randn_like(x)
            # 更安全的写法：使用绝对值，确保sqrt输入为正
            x_next += diffusion_coeff * noise * torch.sqrt(torch.abs(dt))  # 使用欧拉-马尔可夫方法添加噪声
        
        return x_next


class CondIndex:
    """条件向量各维度索引常量
    
    定义条件向量中各统计特征和行为ID的索引位置
    条件向量共15维：1个behavior_id + 14个分位点特征
    """
    BEHAVIOR_ID = 0
    # 上行分位点（7个）
    UP_P1 = 1    # 1%分位点
    UP_P10 = 2   # 10%分位点
    UP_P25 = 3   # 25%分位点
    UP_P50 = 4   # 50%分位点
    UP_P75 = 5   # 75%分位点
    UP_P90 = 6   # 90%分位点
    UP_P99 = 7   # 99%分位点
    # 下行分位点（7个）
    DN_P1 = 8    # 1%分位点
    DN_P10 = 9   # 10%分位点
    DN_P25 = 10  # 25%分位点
    DN_P50 = 11  # 50%分位点
    DN_P75 = 12  # 75%分位点
    DN_P90 = 13  # 90%分位点
    DN_P99 = 14  # 99%分位点


class NetworkTraceCSDIDataset(Dataset):
    """网络轨迹CSDI数据集
    
    用于CSDI模型训练和采样的数据集
    将原始窗口转换为包含虚拟观测点的CSDI输入格式
    """
    
    def __init__(self, windows: List[Dict[str, Any]], seq_len: int = 100):
        """初始化CSDI数据集
        
        Args:
            windows: 归一化后的窗口列表
            seq_len: 原始序列长度
        """
        self.windows = windows
        self.seq_len = seq_len
        # 只保留原始序列长度，虚拟点放在前100个位置内
        self.L_total = seq_len
    
    def __len__(self) -> int:
        """返回数据集长度
        
        Returns:
            int: 数据集样本数量
        """
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本
        
        Args:
            idx: 样本索引
        
        Returns:
            Dict[str, torch.Tensor]: 包含以下键值对：
                - values: [4, 100] 序列值
                - mask: [4, 100] 掩码（1=已观测，0=待生成）
                - time_stamps: [100] 时间戳
                - cond: [15] 条件向量（1个行为ID + 7个up分位点 + 7个dn分位点）
        """
        window_meta = self.windows[idx]
        cond = window_meta["cond"]  # [15] 条件向量（1个行为ID + 7个up分位点 + 7个dn分位点）
        
        # 处理窗口数据
        window = window_meta["window"]
        
        # 提取4个变量的值 [100, 4]
        if isinstance(window, pd.DataFrame):
            # 如果window是DataFrame，直接提取值
            window_values = window[["delay_up_qt", "delay_down_qt", "loss_up", "loss_dn"]].values.T  # [4, 100]
        else:
            # 处理从JSONL加载的窗口数据
            # 从window数组中提取4个变量的值
            window_values = np.zeros((4, self.seq_len))
            
            for i, row in enumerate(window):
                if i >= self.seq_len:
                    break
                window_values[0, i] = row.get("delay_up_qt", 0.0)      # up_delay_qt
                window_values[1, i] = row.get("delay_down_qt", 0.0)    # dn_delay_qt
                window_values[2, i] = row.get("loss_up", 0.0)       # up_loss
                window_values[3, i] = row.get("loss_dn", 0.0)       # dn_loss
        
        # 修复条件向量合法性
        # 确保p1 <= p99
        if cond[CondIndex.UP_P1] > cond[CondIndex.UP_P99]:
            cond = cond.copy()
            cond[CondIndex.UP_P1] = cond[CondIndex.UP_P99]
        if cond[CondIndex.DN_P1] > cond[CondIndex.DN_P99]:
            cond = cond.copy()
            cond[CondIndex.DN_P1] = cond[CondIndex.DN_P99]
        
        # 只保留原始序列长度
        values = np.zeros((4, self.L_total))
        mask = np.zeros((4, self.L_total))
        
        # 填充原始窗口（待生成）
        values[:, :self.seq_len] = window_values  # [4, 100]
        mask = np.zeros((4, self.L_total))  # 全0，无观测点
        # 彻底移除所有虚拟点，让模型完全通过条件向量和diffusion学习自然分布
        
        # 时间戳 [0.0, ..., 1.0]，统一时间范围为[0, 1]
        time_stamps = np.linspace(0.0, 1.0, self.L_total)
        
        # 转换为torch张量
        return {
            "values": torch.tensor(values, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "time_stamps": torch.tensor(time_stamps, dtype=torch.float32),
            "cond": torch.tensor(cond, dtype=torch.float32),
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批次处理函数
        
        Args:
            batch: 样本列表
        
        Returns:
            Dict[str, torch.Tensor]: 批次数据
        """
        return {
            "values": torch.stack([sample["values"] for sample in batch]),
            "mask": torch.stack([sample["mask"] for sample in batch]),
            "time_stamps": torch.stack([sample["time_stamps"] for sample in batch]),
            "cond": torch.stack([sample["cond"] for sample in batch]),
        }


def create_data_loader(
    windows: List[Dict[str, Any]],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """创建数据加载器
    
    Args:
        windows: 窗口数据列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = NetworkTraceCSDIDataset(windows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=NetworkTraceCSDIDataset.collate_fn
    )


class CSDITrainer:
    """CSDI模型训练器
    
    负责CSDI模型的训练和验证
    """
    
    def __init__(
        self,
        model: CSDIModel,
        sde: SimpleSDE,
        device: str = "cpu",
        patience: int = 30,  # 增加耐心值到30，容忍30个epoch无改进
        min_epochs: int = 150,  # 最小训练轮数增加到150，确保模型充分学习
        delta: float = 1e-4  # 最小改进阈值，保持不变
    ):
        """初始化CSDI训练器
        
        Args:
            model: CSDI模型
            sde: SDE对象
            device: 训练设备
            patience: 早停耐心值
            min_epochs: 最小训练轮数
            delta: 最小改进阈值
        """
        self.model = model.to(device)
        self.sde = sde
        self.device = device
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.epoch_count = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行一个训练步骤
        
        Args:
            batch: 批次数据
        
        Returns:
            torch.Tensor: 损失值
        """
        # 获取数据
        values = batch["values"].to(self.device)      # [B, 4, 100]
        mask = batch["mask"].to(self.device)          # [B, 4, 100]
        time_stamps = batch["time_stamps"].to(self.device)  # [B, 100]
        full_cond = batch["cond"].to(self.device)     # [B, cond_dim]
        
        # 直接使用完整条件向量，不再简化
        cond = full_cond  # [B, cond_dim]
        
        B = values.shape[0]
        
        # 采样随机时间步
        t = torch.rand(B, device=self.device) * (1 - 1e-4) + 1e-4
        
        # 只对missing位置加噪，observed位置保持原值
        noise = torch.randn_like(values)
        mean, std = self.sde.marginal_prob(values, t)
        perturbed_values = torch.where(
            mask == 0,
            mean + std * noise,
            values  # observed位置保持原值
        )
        

        # Classifier-Free Guidance (CFG) 训练
        # 1. 以一定概率生成无条件样本
        # 使用0.1的概率进行无条件训练（与常见CFG实现一致）
        do_uncond = torch.rand(1) < 0.1
        if do_uncond:
            # 无条件训练：使用特殊标记-100.0表示无条件
            uncond_cond = torch.full_like(cond, -100.0)  # 使用-100.0作为无条件标记
            pred_noise = self.model(perturbed_values, time_stamps, mask, uncond_cond)
        else:
            # 条件训练：使用真实条件
            pred_noise = self.model(perturbed_values, time_stamps, mask, cond)
        
        # 原有 loss：噪声预测 MSE
        noise_loss = F.mse_loss(pred_noise[mask == 0], noise[mask == 0])
        
        # === 新增：用预测的噪声重建 x0（去噪估计）=== 
        # 根据 SDE 公式：x0 ≈ (x_t - std * pred_noise) / mean_coeff
        log_mean_coeff = -0.25 * t ** 2 * (self.sde.beta_max - self.sde.beta_min) - 0.5 * t * self.sde.beta_min
        # 添加数值稳定性约束，防止log_mean_coeff过于极端导致exp溢出
        log_mean_coeff = torch.clamp(log_mean_coeff, min=-20.0, max=20.0)
        mean_coeff = torch.exp(log_mean_coeff).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # 重建 x0（对所有位置都使用模型预测）
        x0_pred_all = (perturbed_values - std * pred_noise) / (mean_coeff + 1e-8)
        # 但对于observed位置，我们仍然只计算missing位置的噪声损失
        x0_pred = torch.where(
            mask == 1,
            values,  # observed 位置用真实值（用于其他损失计算）
            x0_pred_all
        )
        
        # 移除输出裁剪，让模型自由生成，避免限制模型的表达能力
        # x0_pred = torch.clamp(x0_pred, min=-10.0, max=10.0)
        
        # 基础噪声损失
        noise_loss = F.mse_loss(pred_noise[mask == 0], noise[mask == 0])
        
        # === 新增：分布感知损失（Distribution-Aware Loss）=== 
        # 总损失：只使用噪声损失，信任CSDI + cond引导
        # 移除stat_loss，避免不同空间的比较导致MSE爆炸
        total_loss = noise_loss
        
        # 添加debug信息
        if torch.rand(1) < 0.01:  # 1%概率打印，避免日志过多
            print(f"[Debug] Train Step - Cond p99: {cond[0, CondIndex.UP_P99].item():.4f}")
            print(f"[Debug] Train Step - Pred noise mean: {pred_noise.mean().item():.4f}")
            print(f"[Debug] Train Step - Noise Loss: {noise_loss.item():.4f}")
            print(f"[Debug] Train Step - Total Loss: {total_loss.item():.4f}")
        
        # 梯度检查（0.5%概率）
        if torch.rand(1) < 0.005:
            print("\n[Debug] 梯度检查：")
            print(f"noise_loss: {noise_loss.item():.4f}, requires_grad: {noise_loss.requires_grad}")
            print(f"total_loss: {total_loss.item():.4f}, requires_grad: {total_loss.requires_grad}")
            # 检查pred_noise的梯度（在backward之后才能看到实际梯度值）
            print(f"pred_noise requires_grad: {pred_noise.requires_grad}")
        
        # 打印真实噪声和预测噪声的标准差（0.5%概率）
        if torch.rand(1) < 0.005:
            print("\n[Debug] 噪声统计信息：")
            print(f"真实噪声 std: {noise[mask == 0].std().item():.4f}")
            print(f"预测噪声 std: {pred_noise[mask == 0].std().item():.4f}")
            print(f"MSE loss: {noise_loss.item():.4f}")
            print()
        
        # 检查虚拟点处理（0.5%概率）
        if torch.rand(1) < 0.005:
            print("\n[Debug] 虚拟点处理检查：")
            # 获取虚拟点位置 - 只有t=0.5（中间位置）
            p50_pos = 50  # t=0.5
            
            # 打印第一个样本的虚拟点相关信息
            print(f"样本1 - values[0, 0, {p50_pos}]: {values[0, 0, p50_pos].item():.4f} (up_delay_qt p50 at t=0.5)")
            print(f"样本1 - values[0, 1, {p50_pos}]: {values[0, 1, p50_pos].item():.4f} (dn_delay_qt p50 at t=0.5)")
            
            print(f"样本1 - mask[0, 0, {p50_pos}]: {mask[0, 0, p50_pos].item()} (up_delay_qt p50 mask at t=0.5)")
            print(f"样本1 - mask[0, 1, {p50_pos}]: {mask[0, 1, p50_pos].item()} (dn_delay_qt p50 mask at t=0.5)")
            
            print(f"样本1 - perturbed_values[0, 0, {p50_pos}]: {perturbed_values[0, 0, p50_pos].item():.4f} (up_delay_qt p50 扰动后 at t=0.5)")
            print(f"样本1 - perturbed_values[0, 1, {p50_pos}]: {perturbed_values[0, 1, p50_pos].item():.4f} (dn_delay_qt p50 扰动后 at t=0.5)")
            
            # 检查是否相等（允许微小浮点误差）
            is_up_p50_unchanged = torch.allclose(perturbed_values[0, 0, p50_pos], values[0, 0, p50_pos], atol=1e-6)
            is_dn_p50_unchanged = torch.allclose(perturbed_values[0, 1, p50_pos], values[0, 1, p50_pos], atol=1e-6)
            
            print(f"样本1 - 虚拟点是否未被加噪: up_p50_t0.5={is_up_p50_unchanged}, dn_p50_t0.5={is_dn_p50_unchanged}")
            
            # 打印 x0_pred 的 debug 信息
            print(f"样本1 - x0_pred[0, 0, {p50_pos}]: {x0_pred[0, 0, p50_pos].item():.4f} (x0_pred up_delay_qt at t=0.5)")
            print(f"样本1 - x0_pred[0, 1, {p50_pos}]: {x0_pred[0, 1, p50_pos].item():.4f} (x0_pred dn_delay_qt at t=0.5)")
            print()
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行一个验证步骤
        
        Args:
            batch: 批次数据
        
        Returns:
            torch.Tensor: 损失值
        """
        self.model.eval()
        with torch.no_grad():
            # 获取数据
            values = batch["values"].to(self.device)
            mask = batch["mask"].to(self.device)
            time_stamps = batch["time_stamps"].to(self.device)
            full_cond = batch["cond"].to(self.device)
            
            # 直接使用完整条件向量，不再简化
            cond = full_cond  # [B, cond_dim]
            
            B = values.shape[0]
            
            # 使用多个固定时间步而不是随机采样，减少验证损失波动
            # 选择具有代表性的时间步，覆盖从早期到晚期的扩散过程
            fixed_times = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=self.device)
            
            total_loss = 0.0
            num_time_steps = len(fixed_times)
            
            for t_val in fixed_times:
                # 使用固定时间步，对整个批次使用相同的时间步
                t = t_val.repeat(B)
                
                # 只对missing位置加噪，observed位置保持原值
                noise = torch.randn_like(values)
                mean, std = self.sde.marginal_prob(values, t)
                perturbed_values = torch.where(
                    mask == 0,
                    mean + std * noise,
                    values  # observed位置保持原值
                )
                
                # Classifier-Free Guidance (CFG) 验证
                # 与训练策略一致，以0.1的概率使用无条件训练
                do_uncond = torch.rand(1) < 0.1
                if do_uncond:
                    # 无条件验证：保留behavior ID，将分位点条件置零
                    uncond_cond = cond.clone()
                    uncond_cond[:, 1:] = 0  # 只mask分位点条件，保留behavior ID
                    pred_noise = self.model(perturbed_values, time_stamps, mask, uncond_cond)
                else:
                    # 条件验证：使用真实条件
                    pred_noise = self.model(perturbed_values, time_stamps, mask, cond)
                
                # 只计算missing位置的loss，并累加
                loss = F.mse_loss(pred_noise[mask == 0], noise[mask == 0])
                total_loss += loss.item()
            
            # 计算平均损失
            avg_loss = total_loss / num_time_steps
        
        self.model.train()
        return torch.tensor(avg_loss, device=self.device)
    
    def check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """改进的早停检查
        
        Args:
            val_loss: 当前验证损失
            epoch: 当前训练轮数
        
        Returns:
            bool: 是否应该停止训练
        """
        self.epoch_count += 1
        
        # 确保至少训练min_epochs个epoch
        if epoch < self.min_epochs:
            print(f"⚠️ 训练轮数 {epoch+1}/{self.min_epochs}，未达到早停最小轮数要求")
            return False
        
        # 相对改进幅度计算
        if self.best_val_loss < float("inf"):
            improvement = (self.best_val_loss - val_loss) / self.best_val_loss
        else:
            improvement = float("inf")
        
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            print(f"✅ 验证损失改善: {improvement*100:.2f}%")
        else:
            self.epochs_without_improvement += 1
            print(f"⚠️ 验证损失未改善，连续 {self.epochs_without_improvement} 个epoch")
        
        # 如果连续patience个epoch没有改善，则停止
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            print(f"🛑 早停触发: {self.patience}个epoch无改善")
        
        return self.should_stop
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-4
    ) -> None:
        """改进的训练函数
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,      # 学习率衰减因子
            patience=3,      # 3个epoch无改善则降低学习率
            min_lr=1e-6,     # 最小学习率
            verbose=True     # 打印学习率变化
        )
        
        for epoch in range(num_epochs):
            # 训练
            total_train_loss = 0.0
            self.model.train()
            for batch in train_loader:
                loss = self.train_step(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 验证
            total_val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    loss = self.validation_step(batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # 打印损失
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # 早停检查
            if self.check_early_stopping(avg_val_loss, epoch):
                print(f"早停在epoch {epoch+1}触发")
                break
            
            # 学习率调度
            scheduler.step(avg_val_loss)


class CSDISampler:
    """CSDI模型采样器
    
    负责从CSDI模型中生成样本
    """
    
    def __init__(
        self,
        model: CSDIModel,
        sde: SimpleSDE,
        device: str = "cpu"
    ):
        """初始化CSDI采样器
        
        Args:
            model: 训练好的CSDI模型
            sde: SDE对象
            device: 采样设备
        """
        self.model = model.to(device)
        self.sde = sde
        self.device = device
    
    def sample(
        self,
        cond: torch.Tensor,
        num_inference_steps: int = 100,  # Increased steps for smoother denoising
        guidance_scale: float = 1.5  # CFG guidance scale - further reduced for uniform space
    ) -> torch.Tensor:
        """从条件向量生成新样本，支持Classifier-Free Guidance (CFG)
        
        Args:
            cond: [B, cond_dim] 条件向量 - 15维：1行为ID + 14分位点
            num_inference_steps: 推理步数
            guidance_scale: CFG引导强度，范围[0, ∞)，0表示无引导
        
        Returns:
            [B, 4, 100] 生成的样本
        """
        cond = cond.to(self.device)
        B = cond.shape[0]
        K = 4
        L = 100
        L_total = L  # 只保留原始序列长度
        
        # 直接使用完整条件向量，不再简化
        simplified_cond = cond  # [B, cond_dim]
        
        # 统一时间范围为[0, 1]
        # time_stamps: [L] → [B, L]，与模型forward期望的维度匹配
        time_stamps = torch.linspace(0.0, 1.0, L_total, device=self.device)
        time_stamps = time_stamps.unsqueeze(0).repeat(B, 1)
        
        # 初始化：从纯噪声开始
        x = torch.randn(B, K, L_total, device=self.device)
        
        # 初始化mask：全0，无任何观测点，完全依赖条件向量引导
        mask = torch.zeros((B, K, L_total), device=self.device)
        
        # 生成时间步序列，从1.0到0.0递减
        time_steps = torch.linspace(1.0, 0.0, num_inference_steps+1, device=self.device)
        
        # 逆SDE步骤
        self.model.eval()
        with torch.no_grad():
            for i in range(num_inference_steps):
                # 计算当前和下一个时间步
                t = time_steps[i].repeat(B)
                next_t = time_steps[i+1].repeat(B)
                
                # Classifier-Free Guidance (CFG)
                if guidance_scale > 1.0:
                    # 1. 生成无条件和条件噪声
                    # 无条件样本：使用特殊标记-100.0表示无条件
                    uncond_cond = torch.full_like(simplified_cond, -100.0)  # 使用-100.0作为无条件标记
                    uncond_noise = self.model(x, time_stamps, mask, uncond_cond)
                    cond_noise = self.model(x, time_stamps, mask, simplified_cond)
                    
                    # 2. CFG插值：ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
                    pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                else:
                    # 无CFG或weak CFG：只使用条件样本
                    pred_noise = self.model(x, time_stamps, mask, simplified_cond)
                
                # 更新x，使用确定性去噪步骤
                x = self.sde.denoise_step(x, pred_noise, t, next_t, deterministic=True)
                # 彻底移除clamp操作，允许模型自由生成，恢复尾部分布
        
        # 生成最终样本
        generated = x[:, :, :L]
        
        return generated


def train_csdi(
    train_windows: List[Dict[str, Any]],
    val_windows: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Tuple[CSDIModel, SimpleSDE]:
    """训练CSDI模型
    
    Args:
        train_windows: 训练窗口列表
        val_windows: 验证窗口列表
        config: 训练配置
    
    Returns:
        Tuple[CSDIModel, SimpleSDE]: 训练好的模型和SDE对象
    """
    # 配置 - 调整训练策略，确保模型充分训练
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 150)  # 增加到150轮，确保达到min_epochs要求
    learning_rate = config.get("learning_rate", 1e-5)  # 保持较低学习率，确保稳定训练
    hidden_dim = config.get("hidden_dim", 128)  # 恢复到128，减少模型参数量，加快训练
    num_layers = config.get("num_layers", 4)  # 恢复到4层，减少计算量
    num_heads = config.get("num_heads", 8)
    dropout = config.get("dropout", 0.1)  # 恢复到0.1，减少训练稳定性影响
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建数据加载器
    train_loader = create_data_loader(train_windows, batch_size, shuffle=True)
    val_loader = create_data_loader(val_windows, batch_size, shuffle=False)
    
    # 初始化模型和SDE
    # 动态获取条件向量维度
    sample_cond = train_windows[0]["cond"]
    actual_cond_dim = len(sample_cond)
    
    # 行为ID最多按照8个算（0-7），直接设置num_behavior_ids为8
    # 无需根据训练数据动态计算，因为用户明确表示实际行为ID不可能超出7
    num_behavior_ids = 8
    
    model = CSDIModel(
        input_dim=4,
        cond_dim=actual_cond_dim,  # 使用实际的条件向量维度
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_behavior_ids=num_behavior_ids  # 固定为8个行为ID（0-7）
    )
    
    sde = SimpleSDE()
    
    # 初始化训练器
    trainer = CSDITrainer(model, sde, device=device)
    
    # 训练模型
    trainer.train(train_loader, val_loader, num_epochs, learning_rate)
    
    return model, sde


def generate_samples(
    model: CSDIModel,
    sde: SimpleSDE,
    cond: torch.Tensor,
    num_inference_steps: int = 50,
    device: str = "cpu",
    guidance_scale: float = 2.0  # CFG引导强度，降低到2.0适合uniform空间
) -> torch.Tensor:
    """生成样本
    
    Args:
        model: 训练好的模型
        sde: SDE对象
        cond: [B, cond_dim] 条件向量
        num_inference_steps: 推理步数
        device: 采样设备
        guidance_scale: CFG引导强度，范围[0, ∞)，默认3.0
    
    Returns:
        [B, 4, 100] 生成的样本
    """
    print(f"[Debug] generate_samples - guidance_scale: {guidance_scale}")
    print(f"[Debug] generate_samples - cond mean: {cond.mean().item():.4f}, cond shape: {cond.shape}")
    
    sampler = CSDISampler(model, sde, device=device)
    return sampler.sample(cond, num_inference_steps, guidance_scale)

def parse_args():
    """解析命令行参数
    
    Args:
        无
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="CSDI模型训练与测试")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="./output/csdi", help="输出目录")
    parser.add_argument("--num-epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--hidden-dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="推理步数")
    parser.add_argument("--generate-samples", type=int, default=10, help="生成样本数量")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--data-dir", type=str, default="./output/datasets", help="已生成的数据集目录")
    parser.add_argument("--skip-preprocessing", action="store_true", help="跳过预处理，直接使用已生成的数据集")
    return parser.parse_args()

def load_windows_from_jsonl(file_path: str) -> list:
    """从JSONL文件加载窗口数据
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        list: 窗口数据列表
    """
    windows = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                window_meta = json.loads(line)
                windows.append(window_meta)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to decode line in {file_path}: {e}")
                continue
    
    # # 修复条件向量合法性
    # fixed_count = 0
    # for i, window_meta in enumerate(windows):
    #     cond = window_meta["cond"]
    #     need_fix = False
    #     
    #     # 确保p5 <= p95
    #     if cond[CondIndex.UP_P5] > cond[CondIndex.UP_P95]:
    #         cond[CondIndex.UP_P5] = cond[CondIndex.UP_P95]
    #         need_fix = True
    #     if cond[CondIndex.DN_P5] > cond[CondIndex.DN_P95]:
    #         cond[CondIndex.DN_P5] = cond[CondIndex.DN_P95]
    #         need_fix = True
    #     
    #     # 确保std >= 0
    #     if cond[CondIndex.UP_STD] < 0:
    #         cond[CondIndex.UP_STD] = 0
    #         need_fix = True
    #     if cond[CondIndex.DN_STD] < 0:
    #         cond[CondIndex.DN_STD] = 0
    #         need_fix = True
    #     
    #     if need_fix:
    #         windows[i]["cond"] = cond
    #         fixed_count += 1
    # 
    # if fixed_count > 0:
    #     print(f"[Data Fix] 修复了 {fixed_count} 个条件向量的合法性问题")
    
    # 检查数据范围
    if windows:
        # 检查条件向量的范围
        cond_values = np.array([w["cond"] for w in windows])
        print(f"[Data Check] 条件向量范围 - min: {cond_values.min():.4f}, max: {cond_values.max():.4f}, mean: {cond_values.mean():.4f}, std: {cond_values.std():.4f}")
        
        # 打印关键统计量的范围
        up_p1_values = cond_values[:, CondIndex.UP_P1]
        up_p99_values = cond_values[:, CondIndex.UP_P99]
        up_p50_values = cond_values[:, CondIndex.UP_P50]
        dn_p1_values = cond_values[:, CondIndex.DN_P1]
        dn_p99_values = cond_values[:, CondIndex.DN_P99]
        dn_p50_values = cond_values[:, CondIndex.DN_P50]
        
        print(f"[Data Check] up_p1 范围 - min: {up_p1_values.min():.4f}, max: {up_p1_values.max():.4f}")
        print(f"[Data Check] up_p50 范围 - min: {up_p50_values.min():.4f}, max: {up_p50_values.max():.4f}")
        print(f"[Data Check] up_p99 范围 - min: {up_p99_values.min():.4f}, max: {up_p99_values.max():.4f}")
        print(f"[Data Check] dn_p1 范围 - min: {dn_p1_values.min():.4f}, max: {dn_p1_values.max():.4f}")
        print(f"[Data Check] dn_p50 范围 - min: {dn_p50_values.min():.4f}, max: {dn_p50_values.max():.4f}")
        print(f"[Data Check] dn_p99 范围 - min: {dn_p99_values.min():.4f}, max: {dn_p99_values.max():.4f}")
    
    return windows

def main():
    """主函数
    
    执行CSDI模型的训练和测试
    """
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== 开始CSDI模型训练与测试 ===")
    
    if args.skip_preprocessing:
        # 从JSONL文件加载数据
        print("1. 从JSONL文件加载数据...")
        data_dir = Path(args.data_dir)
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "val.jsonl"
        
        if not train_file.exists() or not val_file.exists():
            print(f"Error: JSONL files not found in {data_dir}")
            print("Please run preprocessing first or check the data directory.")
            sys.exit(1)
        
        train_windows = load_windows_from_jsonl(train_file)
        val_windows = load_windows_from_jsonl(val_file)
        
        print(f"   加载训练样本数量: {len(train_windows)}")
        print(f"   加载验证样本数量: {len(val_windows)}")
        
        # 检查窗口数据格式，确保window是numpy数组
        # 处理从JSONL加载的窗口数据，转换为(4, 100)数组格式
        for i, window_meta in enumerate(train_windows):
            if isinstance(window_meta["window"], dict) and "window" in window_meta["window"]:
                # 处理嵌套结构
                nested_window = window_meta["window"]["window"]
                if isinstance(nested_window, list):
                    # 转换为numpy数组，假设嵌套结构是(100, 5)的列表，只取后4列
                    window_array = np.array(nested_window)[:, 1:].T  # [4, 100]
                    train_windows[i]["window"] = window_array
        
        for i, window_meta in enumerate(val_windows):
            if isinstance(window_meta["window"], dict) and "window" in window_meta["window"]:
                # 处理嵌套结构
                nested_window = window_meta["window"]["window"]
                if isinstance(nested_window, list):
                    # 转换为numpy数组，假设嵌套结构是(100, 5)的列表，只取后4列
                    window_array = np.array(nested_window)[:, 1:].T  # [4, 100]
                    val_windows[i]["window"] = window_array
    else:
        # 1. 预处理数据
        print("1. 预处理数据...")
        # 动态导入run_preprocessing函数
        from src.preprocessing.pipeline import PreprocessingPipeline
        import yaml
        
        # 加载配置文件
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建并运行预处理流水线
        pipeline = PreprocessingPipeline(config)
        preprocessing_result = pipeline.run()
        
        # 2. 准备训练和验证数据
        print("2. 准备训练和验证数据...")
        train_windows = preprocessing_result["datasets"]["train"]
        val_windows = preprocessing_result["datasets"]["val"]
    
    # 3. 配置训练参数
    # 设备自动检测
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "device": device
    }
    
    # 4. 训练CSDI模型
    print("3. 训练CSDI模型...")
    model, sde = train_csdi(train_windows, val_windows, config)
    
    # 5. 保存模型
    print("4. 保存模型...")
    model_path = output_dir / "csdi_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, model_path)
    
    # 6. 生成样本进行测试
    print(f"5. 生成{args.generate_samples}个样本进行测试...")
    
    # 按行为ID对训练窗口进行分组，确保覆盖不同的行为ID
    behavior_groups = {}
    for window_meta in train_windows:
        behavior_id = int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
        if behavior_id not in behavior_groups:
            behavior_groups[behavior_id] = []
        behavior_groups[behavior_id].append(window_meta)
    
    print(f"   训练数据按行为ID分组：{[(bid, len(windows)) for bid, windows in behavior_groups.items()]}")
    print(f"   行为ID列表：{list(behavior_groups.keys())}")
    
    # 样本挑选策略：从每个行为ID组中选择一个样本，然后从所有训练窗口中随机选择剩余样本
    selected_windows = []
    
    # 1. 从每个行为ID组中选择一个样本
    import random
    for behavior_id, windows in behavior_groups.items():
        if windows:
            selected_windows.append(random.choice(windows))
    
    # 2. 如果还需要更多样本，从所有训练窗口中随机选择剩余样本
    num_needed = min(args.generate_samples, 3)  # 最多选择3个样本
    if len(selected_windows) < num_needed:
        remaining_needed = num_needed - len(selected_windows)
        # 从所有训练窗口中随机选择剩余样本
        remaining_samples = random.sample(train_windows, remaining_needed)
        selected_windows.extend(remaining_samples)
    
    # 确保不超过指定的数量
    selected_windows = selected_windows[:num_needed]
    
    print(f"   从{len(behavior_groups)}个行为ID组中选择了{len(selected_windows)}个样本")
    
    # 提取完整的条件向量
    full_conds = []
    for window in selected_windows:
        full_cond = window["cond"]
        full_conds.append(full_cond)
    
    test_conds = torch.tensor(
        full_conds, 
        dtype=torch.float32
    )
    print(f"[Debug] test_conds full shape: {test_conds.shape}")
    print(f"[Debug] test_conds first sample: {test_conds[0].tolist()}")
    
    # 生成样本 - 降低CFG引导强度，适合uniform空间
    generated_samples = generate_samples(
        model, sde, test_conds, 
        num_inference_steps=args.num_inference_steps,
        guidance_scale=2.0  # 关键修改：从10.0降低到2.0，适合uniform空间
    )

    print(f"6. 生成样本完成，形状：{generated_samples.shape}")
    # 打印生成样本的范围，检查是否有异常值
    print(f"   生成样本范围 - min: {generated_samples.min().item():.4f}, max: {generated_samples.max().item():.4f}")

    # 对生成的样本进行反归一化处理
    generated_samples_np = generated_samples.cpu().numpy()
    # 加载QuantileTransformer，用于反归一化
    if not args.skip_preprocessing:
        # 如果没有跳过预处理，从preprocessing_result中获取qt_up和qt_dn
        qt_up = preprocessing_result["assets"]["qt_up"]
        qt_dn = preprocessing_result["assets"]["qt_down"]
    else:
        # 否则，尝试从assets目录加载
        assets_dir = Path("./output/assets")
        qt_up_path = assets_dir / "qt_up.pkl"
        qt_dn_path = assets_dir / "qt_down.pkl"
        
        if qt_up_path.exists() and qt_dn_path.exists():
            qt_up = joblib.load(qt_up_path)
            qt_dn = joblib.load(qt_dn_path)
            print(f"   成功从 {assets_dir} 加载QuantileTransformer")
        else:
            qt_up = None
            qt_dn = None
            print(f"Warning: QuantileTransformer not found in {assets_dir}, skipping inverse transformation.")
    
    # 对生成的样本进行反归一化处理
    generated_samples_np = generated_samples.cpu().numpy()
    if qt_up is not None and qt_dn is not None:
        print("\n7. 对生成样本进行反归一化处理...")
        # 复制一份原始样本，用于反归一化
        generated_samples_denorm = generated_samples_np.copy()
        
        # 分别对上行和下行时延进行反归一化
        # 维度：[B, 4, 100]，其中第1维是变量索引：0=上行时延, 1=下行时延, 2=上行丢包率, 3=下行丢包率
        B, K, L = generated_samples_np.shape
        
        # 1. 上行时延反归一化
        # 取消clip，直接进行反归一化
        up_delay_norm = generated_samples_np[:, 0, :].reshape(-1, 1)  # 转换为 [B*L, 1]
        up_delay_denorm = qt_up.inverse_transform(up_delay_norm).reshape(B, L)  # 直接反归一化，不进行clip
        generated_samples_denorm[:, 0, :] = up_delay_denorm  # 更新上行时延
        
        # 2. 下行时延反归一化
        # 取消clip，直接进行反归一化
        down_delay_norm = generated_samples_np[:, 1, :].reshape(-1, 1)  # 转换为 [B*L, 1]
        down_delay_denorm = qt_dn.inverse_transform(down_delay_norm).reshape(B, L)  # 直接反归一化，不进行clip
        generated_samples_denorm[:, 1, :] = down_delay_denorm  # 更新下行时延
        
        print(f"   反归一化后样本范围：{generated_samples_denorm.min():.4f} ~ {generated_samples_denorm.max():.4f}")
        print(f"   上行时延范围：{generated_samples_denorm[:, 0, :].min():.4f} ms ~ {generated_samples_denorm[:, 0, :].max():.4f} ms")
        print(f"   下行时延范围：{generated_samples_denorm[:, 1, :].min():.4f} ms ~ {generated_samples_denorm[:, 1, :].max():.4f} ms")
        # 使用反归一化后的样本
        generated_samples_np = generated_samples_denorm
    
    # 保存生成的样本
    samples_path = output_dir / "generated_samples.npy"
    import numpy as np
    np.save(samples_path, generated_samples_np)
    print(f"8. 反归一化后的样本已保存到：{samples_path}")
    
    # 8. 加载QuantileTransformer，用于反归一化
    if not args.skip_preprocessing:
        # 如果没有跳过预处理，从preprocessing_result中获取qt_up和qt_dn
        qt_up = preprocessing_result["assets"]["qt_up"]
        qt_dn = preprocessing_result["assets"]["qt_down"]
    else:
        # 否则，尝试从assets目录加载
        assets_dir = Path("./output/assets")
        qt_up_path = assets_dir / "qt_up.pkl"
        qt_dn_path = assets_dir / "qt_down.pkl"
        
        if qt_up_path.exists() and qt_dn_path.exists():
            qt_up = joblib.load(qt_up_path)
            qt_dn = joblib.load(qt_dn_path)
            print(f"   成功从 {assets_dir} 加载QuantileTransformer")
        else:
            qt_up = None
            qt_dn = None
            print(f"Warning: QuantileTransformer not found in {assets_dir}, skipping inverse transformation.")
    
    # 9. 打印生成样本的统计特征和与参考样本的对比
    print("\n8. 生成样本统计特征与参考样本对比：")
    generated_samples_np = generated_samples.cpu().numpy()
    
    # 保存参考样本信息，用于可视化
    reference_samples_info = []
    
    for i in range(generated_samples_np.shape[0]):
        # 获取参考样本信息
        ref_window = selected_windows[i]
        ref_cond = ref_window["cond"]
        behavior_id = int(ref_cond[CondIndex.BEHAVIOR_ID])
        
        print(f"\n=== 样本 {i+1} ===")
        print(f"  参考样本行为ID: {behavior_id}")
        
        # 打印参考样本的条件向量信息
        print(f"  参考样本条件向量 - up_p10: {ref_cond[CondIndex.UP_P10]:.4f}, up_p99: {ref_cond[CondIndex.UP_P99]:.4f}, up_p50: {ref_cond[CondIndex.UP_P50]:.4f}")
        print(f"  参考样本条件向量 - dn_p10: {ref_cond[CondIndex.DN_P10]:.4f}, dn_p99: {ref_cond[CondIndex.DN_P99]:.4f}, dn_p50: {ref_cond[CondIndex.DN_P50]:.4f}")
        
        # 提取参考样本的实际时延数据并计算统计特征
        ref_window_df = ref_window["window"]
        if isinstance(ref_window_df, pd.DataFrame):
            # 如果window是DataFrame，直接提取值
            ref_del_up_qt = ref_window_df["delay_up_qt"].values
            ref_del_dn_qt = ref_window_df["delay_down_qt"].values
            
            # 将DataFrame转换为字典列表，便于JSON序列化
            window_list = []
            for _, row in ref_window_df.iterrows():
                window_list.append({
                    "delay_up_qt": row["delay_up_qt"],
                    "delay_down_qt": row["delay_down_qt"]
                })
        else:
            # 处理从JSONL加载的窗口数据
            ref_del_up_qt = np.array([row.get("delay_up_qt", 0.0) for row in ref_window_df])
            ref_del_dn_qt = np.array([row.get("delay_down_qt", 0.0) for row in ref_window_df])
            # 已经是列表格式，直接使用
            window_list = ref_window_df
        
        # 计算参考样本的统计特征
        ref_up_p5 = np.percentile(ref_del_up_qt, 5)
        ref_up_p50 = np.percentile(ref_del_up_qt, 50)
        ref_up_p95 = np.percentile(ref_del_up_qt, 95)
        ref_up_p99 = np.percentile(ref_del_up_qt, 99)
        ref_up_std = np.std(ref_del_up_qt)
        
        ref_down_p5 = np.percentile(ref_del_dn_qt, 5)
        ref_down_p50 = np.percentile(ref_del_dn_qt, 50)
        ref_down_p95 = np.percentile(ref_del_dn_qt, 95)
        ref_down_p99 = np.percentile(ref_del_dn_qt, 99)
        ref_down_std = np.std(ref_del_dn_qt)
        
        # 打印参考样本统计特征（QT归一化空间）
        print("\n  参考样本统计特征（QT归一化空间）：")
        print(f"  上行延迟 - p5: {ref_up_p5:.4f}, p50: {ref_up_p50:.4f}, p95: {ref_up_p95:.4f}, p99: {ref_up_p99:.4f}, std: {ref_up_std:.4f}")
        print(f"  下行延迟 - p5: {ref_down_p5:.4f}, p50: {ref_down_p50:.4f}, p95: {ref_down_p95:.4f}, p99: {ref_down_p99:.4f}, std: {ref_down_std:.4f}")
        
        # 参考样本反归一化（QT空间 → 原始空间）
        if qt_up is not None and qt_dn is not None:
            # 将QT空间的时延数据转换回原始空间
            ref_del_up_original = qt_up.inverse_transform(ref_del_up_qt.reshape(-1, 1)).flatten()
            ref_del_dn_original = qt_dn.inverse_transform(ref_del_dn_qt.reshape(-1, 1)).flatten()
            
            # 计算反归一化后的统计特征
            ref_up_original_p5 = np.percentile(ref_del_up_original, 5)
            ref_up_original_p50 = np.percentile(ref_del_up_original, 50)
            ref_up_original_p95 = np.percentile(ref_del_up_original, 95)
            ref_up_original_p99 = np.percentile(ref_del_up_original, 99)
            ref_up_original_std = np.std(ref_del_up_original)
            
            ref_down_original_p5 = np.percentile(ref_del_dn_original, 5)
            ref_down_original_p50 = np.percentile(ref_del_dn_original, 50)
            ref_down_original_p95 = np.percentile(ref_del_dn_original, 95)
            ref_down_original_p99 = np.percentile(ref_del_dn_original, 99)
            ref_down_original_std = np.std(ref_del_dn_original)
            
            # 打印参考样本反归一化后的统计特征
            print("\n  参考样本统计特征（原始空间）：")
            print(f"  上行延迟 - p5: {ref_up_original_p5:.4f} ms, p50: {ref_up_original_p50:.4f} ms, p95: {ref_up_original_p95:.4f} ms, p99: {ref_up_original_p99:.4f} ms, std: {ref_up_original_std:.4f} ms")
            print(f"  下行延迟 - p5: {ref_down_original_p5:.4f} ms, p50: {ref_down_original_p50:.4f} ms, p95: {ref_down_original_p95:.4f} ms, p99: {ref_down_original_p99:.4f} ms, std: {ref_down_original_std:.4f} ms")
        
        # 提取生成的上行延迟和下行延迟数据
        gen_del_up_qt = generated_samples_np[i, 0, :]  # 上行延迟（QT归一化后）
        gen_del_dn_qt = generated_samples_np[i, 1, :]  # 下行延迟（QT归一化后）
        # 上行丢包率和下行丢包率暂时未使用
        # gen_loss_up = generated_samples_np[i, 2, :]  # 上行丢包率
        # gen_loss_dn = generated_samples_np[i, 3, :]  # 下行丢包率
        
        # 计算统计特征 - 使用QT归一化空间
        gen_up_p5 = np.percentile(gen_del_up_qt, 5)
        gen_up_p50 = np.percentile(gen_del_up_qt, 50)
        gen_up_p95 = np.percentile(gen_del_up_qt, 95)
        gen_up_p99 = np.percentile(gen_del_up_qt, 99)
        gen_up_std = np.std(gen_del_up_qt)
        
        gen_down_p5 = np.percentile(gen_del_dn_qt, 5)
        gen_down_p50 = np.percentile(gen_del_dn_qt, 50)
        gen_down_p95 = np.percentile(gen_del_dn_qt, 95)
        gen_down_p99 = np.percentile(gen_del_dn_qt, 99)
        gen_down_std = np.std(gen_del_dn_qt)
        
        # 打印生成样本统计特征（QT归一化空间）
        print("\n  生成样本统计特征（QT归一化空间）：")
        print(f"  上行延迟 - p5: {gen_up_p5:.4f}, p50: {gen_up_p50:.4f}, p95: {gen_up_p95:.4f}, p99: {gen_up_p99:.4f}, std: {gen_up_std:.4f}")
        print(f"  下行延迟 - p5: {gen_down_p5:.4f}, p50: {gen_down_p50:.4f}, p95: {gen_down_p95:.4f}, p99: {gen_down_p99:.4f}, std: {gen_down_std:.4f}")
        
        # 生成样本反归一化（QT空间 → 原始空间）
        if qt_up is not None and qt_dn is not None:
            # 将QT空间的时延数据转换回原始空间
            gen_del_up_original = qt_up.inverse_transform(gen_del_up_qt.reshape(-1, 1)).flatten()
            gen_del_dn_original = qt_dn.inverse_transform(gen_del_dn_qt.reshape(-1, 1)).flatten()
            
            # 计算反归一化后的统计特征
            gen_up_original_p5 = np.percentile(gen_del_up_original, 5)
            gen_up_original_p50 = np.percentile(gen_del_up_original, 50)
            gen_up_original_p95 = np.percentile(gen_del_up_original, 95)
            gen_up_original_p99 = np.percentile(gen_del_up_original, 99)
            gen_up_original_std = np.std(gen_del_up_original)
            
            gen_down_original_p5 = np.percentile(gen_del_dn_original, 5)
            gen_down_original_p50 = np.percentile(gen_del_dn_original, 50)
            gen_down_original_p95 = np.percentile(gen_del_dn_original, 95)
            gen_down_original_p99 = np.percentile(gen_del_dn_original, 99)
            gen_down_original_std = np.std(gen_del_dn_original)
            
            # 打印生成样本反归一化后的统计特征
            print("\n  生成样本统计特征（原始空间）：")
            print(f"  上行延迟 - p5: {gen_up_original_p5:.4f} ms, p50: {gen_up_original_p50:.4f} ms, p95: {gen_up_original_p95:.4f} ms, p99: {gen_up_original_p99:.4f} ms, std: {gen_up_original_std:.4f} ms")
            print(f"  下行延迟 - p5: {gen_down_original_p5:.4f} ms, p50: {gen_down_original_p50:.4f} ms, p95: {gen_down_original_p95:.4f} ms, p99: {gen_down_original_p99:.4f} ms, std: {gen_down_original_std:.4f} ms")
        
        # 计算与参考样本的误差（QT归一化空间）
        
        # 计算上行延迟统计特征的绝对误差
        up_p99_error = abs(gen_up_p99 - ref_cond[CondIndex.UP_P99])
        up_p50_error = abs(gen_up_p50 - ref_cond[CondIndex.UP_P50])
        
        # 计算下行延迟统计特征的绝对误差
        dn_p99_error = abs(gen_down_p99 - ref_cond[CondIndex.DN_P99])
        dn_p50_error = abs(gen_down_p50 - ref_cond[CondIndex.DN_P50])
        
        print("\n  误差分析（QT归一化空间）：")
        print(f"  上行延迟 - p50误差: {up_p50_error:.4f}, p99误差: {up_p99_error:.4f}")
        print(f"  下行延迟 - p50误差: {dn_p50_error:.4f}, p99误差: {dn_p99_error:.4f}")
        
        # 保存参考样本信息，用于可视化
        # 检查ref_cond的类型，如果已经是列表，直接使用；否则调用tolist()方法
        cond_data = ref_cond.tolist() if hasattr(ref_cond, 'tolist') else ref_cond
        reference_samples_info.append({
            "behavior_id": behavior_id,
            "window": window_list,
            "cond": cond_data
        })
    
    # 生成可视化图片
    print("9. 生成可视化图片...")
    import shutil
    import numpy as np
    # 创建可视化目录
    visualization_dir = output_dir.parent / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    # 复制生成的样本到可视化目录
    generated_samples_path = output_dir / "generated_samples.npy"
    visualization_samples_path = visualization_dir / "generated_samples.npy"
    shutil.copy(generated_samples_path, visualization_samples_path)
    
    # 保存参考样本信息到可视化目录，使用npy格式
    reference_info_path = visualization_dir / "reference_samples_info.npz"
    
    # 提取参考样本的关键数据
    behavior_ids = []
    ref_conds = []
    ref_trajectories = []
    
    for sample_info in reference_samples_info:
        behavior_ids.append(sample_info["behavior_id"])
        ref_conds.append(sample_info["cond"])
        
        # 提取参考轨迹数据
        ref_window = sample_info["window"]
        ref_up_delay = []
        ref_down_delay = []
        
        for row in ref_window:
            ref_up_delay.append(row["delay_up_qt"])
            ref_down_delay.append(row["delay_down_qt"])
        
        ref_trajectories.append([ref_up_delay, ref_down_delay])
    
    # 转换为numpy数组
    behavior_ids = np.array(behavior_ids)
    ref_conds = np.array(ref_conds)
    ref_trajectories = np.array(ref_trajectories)
    
    # 保存为npz文件，支持多个数组
    np.savez(reference_info_path, 
             behavior_ids=behavior_ids, 
             ref_conds=ref_conds, 
             ref_trajectories=ref_trajectories)
    
    # 运行可视化脚本，传递正确的参考样本文件路径
    import subprocess
    subprocess.run(["uv", "run", "python", "scripts/step3_visualize.py", "--reference-file", str(reference_info_path)], check=True)
    print(f"10. 可视化图片已生成到: {visualization_dir}")

if __name__ == "__main__":
    main()


