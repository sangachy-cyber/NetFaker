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
        
        # 分位点条件投影：接收14维分位点特征（去除ID）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim - 1, hidden_dim),  # 条件向量去除ID维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
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
        # 1. 提取behavior ID（第0维）和分位点特征（1-14维）
        behavior_id = cond[:, 0].long()  # [B]
        quantile_cond = cond[:, 1:]  # [B, 14]
        
        # 2. behavior ID嵌入
        behavior_id_emb = self.behavior_id_emb(behavior_id)  # [B, H]
        behavior_id_emb = behavior_id_emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
        
        # 3. 分位点条件投影
        quantile_emb = self.cond_proj(quantile_cond)  # [B, H]
        quantile_emb = quantile_emb.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H]
        
        # 4. 合并条件嵌入
        cond_emb = behavior_id_emb + quantile_emb  # [B, 1, 1, H]
        
        # 5. 增强条件嵌入的影响
        cond_emb = cond_emb * 10.0  # 更强力地增强条件嵌入的影响
        
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
        h = value_emb + combined_emb + cond_emb  # [B, K, L, H]
        
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
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        """初始化SDE
        
        Args:
            beta_min: 最小噪声强度
            beta_max: 最大噪声强度
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
    
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
        
        # 计算score：-pred_noise / std_t
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
            window_values = window[["delay_up", "delay_down", "loss_up", "loss_dn"]].values.T  # [4, 100]
        else:
            # 处理从JSONL加载的窗口数据
            # 从window数组中提取4个变量的值
            window_values = np.zeros((4, self.seq_len))
            
            for i, row in enumerate(window):
                if i >= self.seq_len:
                    break
                window_values[0, i] = row.get("delay_up", 0.0)      # up_delay
                window_values[1, i] = row.get("delay_down", 0.0)    # dn_delay
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
        patience: int = 10,  # 增加耐心值
        min_epochs: int = 20,  # 最小训练轮数
        delta: float = 1e-4  # 最小改进阈值
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
        full_cond = batch["cond"].to(self.device)     # [B, 23]
        
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
            # 无条件训练：保留behavior ID，将分位点条件置零
            uncond_cond = cond.clone()
            uncond_cond[:, 1:] = 0  # 只mask分位点条件，保留behavior ID
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
            print(f"样本1 - values[0, 0, {p50_pos}]: {values[0, 0, p50_pos].item():.4f} (up_delay p50 at t=0.5)")
            print(f"样本1 - values[0, 1, {p50_pos}]: {values[0, 1, p50_pos].item():.4f} (dn_delay p50 at t=0.5)")
            
            print(f"样本1 - mask[0, 0, {p50_pos}]: {mask[0, 0, p50_pos].item()} (up_delay p50 mask at t=0.5)")
            print(f"样本1 - mask[0, 1, {p50_pos}]: {mask[0, 1, p50_pos].item()} (dn_delay p50 mask at t=0.5)")
            
            print(f"样本1 - perturbed_values[0, 0, {p50_pos}]: {perturbed_values[0, 0, p50_pos].item():.4f} (up_delay p50 扰动后 at t=0.5)")
            print(f"样本1 - perturbed_values[0, 1, {p50_pos}]: {perturbed_values[0, 1, p50_pos].item():.4f} (dn_delay p50 扰动后 at t=0.5)")
            
            # 检查是否相等（允许微小浮点误差）
            is_up_p50_unchanged = torch.allclose(perturbed_values[0, 0, p50_pos], values[0, 0, p50_pos], atol=1e-6)
            is_dn_p50_unchanged = torch.allclose(perturbed_values[0, 1, p50_pos], values[0, 1, p50_pos], atol=1e-6)
            
            print(f"样本1 - 虚拟点是否未被加噪: up_p50_t0.5={is_up_p50_unchanged}, dn_p50_t0.5={is_dn_p50_unchanged}")
            
            # 打印 x0_pred 的 debug 信息
            print(f"样本1 - x0_pred[0, 0, {p50_pos}]: {x0_pred[0, 0, p50_pos].item():.4f} (x0_pred up_delay at t=0.5)")
            print(f"样本1 - x0_pred[0, 1, {p50_pos}]: {x0_pred[0, 1, p50_pos].item():.4f} (x0_pred dn_delay at t=0.5)")
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
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0  # CFG guidance scale
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
                    # 无条件样本：保留behavior ID，将分位点条件置零
                    uncond_cond = simplified_cond.clone()
                    uncond_cond[:, 1:] = 0  # 只mask分位点条件，保留behavior ID
                    uncond_noise = self.model(x, time_stamps, mask, uncond_cond)
                    cond_noise = self.model(x, time_stamps, mask, simplified_cond)
                    
                    # 2. CFG插值：ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
                    pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
                else:
                    # 无CFG或weak CFG：只使用条件样本
                    pred_noise = self.model(x, time_stamps, mask, simplified_cond)
                
                # 更新x，使用确定性去噪步骤
                x = self.sde.denoise_step(x, pred_noise, t, next_t, deterministic=True)
        
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
    # 配置 - 调整训练策略，平衡性能和速度
    batch_size = config.get("batch_size", 32)
    num_epochs = config.get("num_epochs", 50)  # 恢复到50轮，平衡训练时间和效果
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
    guidance_scale: float = 3.0  # CFG引导强度，提高到3.0增强条件控制
) -> torch.Tensor:
    """生成样本
    
    Args:
        model: 训练好的模型
        sde: SDE对象
        cond: [B, 23] 条件向量
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
        
    #     # 确保p5 <= p95
    #     if cond[CondIndex.UP_P5] > cond[CondIndex.UP_P95]:
    #         cond[CondIndex.UP_P5] = cond[CondIndex.UP_P95]
    #         need_fix = True
    #     if cond[CondIndex.DN_P5] > cond[CondIndex.DN_P95]:
    #         cond[CondIndex.DN_P5] = cond[CondIndex.DN_P95]
    #         need_fix = True
        
    #     # 确保std >= 0
    #     if cond[CondIndex.UP_STD] < 0:
    #         cond[CondIndex.UP_STD] = 0
    #         need_fix = True
    #     if cond[CondIndex.DN_STD] < 0:
    #         cond[CondIndex.DN_STD] = 0
    #         need_fix = True
        
    #     if need_fix:
    #         windows[i]["cond"] = cond
    #         fixed_count += 1
    
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
    
    # 生成样本
    generated_samples = generate_samples(
        model, sde, test_conds, num_inference_steps=args.num_inference_steps
    )
    
    print(f"6. 生成样本完成，形状：{generated_samples.shape}")
    # 打印生成样本的范围，检查是否有异常值
    print(f"   生成样本范围 - min: {generated_samples.min().item():.4f}, max: {generated_samples.max().item():.4f}")
    
    # 7. 保存生成的样本
    samples_path = output_dir / "generated_samples.npy"
    np.save(samples_path, generated_samples.cpu().numpy())
    print(f"7. 生成的样本已保存到：{samples_path}")
    
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
        
        # 提取生成的上行延迟和下行延迟数据
        gen_del_up = generated_samples_np[i, 0, :]  # 上行延迟（归一化后）
        gen_del_dn = generated_samples_np[i, 1, :]  # 下行延迟（归一化后）
        gen_loss_up = generated_samples_np[i, 2, :]  # 上行丢包率（归一化后）
        gen_loss_dn = generated_samples_np[i, 3, :]  # 下行丢包率（归一化后）
        
        # 计算统计特征 - 使用双重归一化空间（QuantileTransformer + z-score）
        gen_up_p5 = np.percentile(gen_del_up, 5)
        gen_up_p50 = np.percentile(gen_del_up, 50)
        gen_up_p95 = np.percentile(gen_del_up, 95)
        gen_up_p99 = np.percentile(gen_del_up, 99)
        gen_up_std = np.std(gen_del_up)
        
        gen_down_p5 = np.percentile(gen_del_dn, 5)
        gen_down_p50 = np.percentile(gen_del_dn, 50)
        gen_down_p95 = np.percentile(gen_del_dn, 95)
        gen_down_p99 = np.percentile(gen_del_dn, 99)
        gen_down_std = np.std(gen_del_dn)
        
        # 打印生成样本统计特征（双重归一化空间）
        print("\n  生成样本统计特征（双重归一化空间：QuantileTransformer + z-score）：")
        print(f"  上行延迟 - p5: {gen_up_p5:.4f}, p50: {gen_up_p50:.4f}, p95: {gen_up_p95:.4f}, p99: {gen_up_p99:.4f}, std: {gen_up_std:.4f}")
        print(f"  下行延迟 - p5: {gen_down_p5:.4f}, p50: {gen_down_p50:.4f}, p95: {gen_down_p95:.4f}, p99: {gen_down_p99:.4f}, std: {gen_down_std:.4f}")
        
        # 计算丢包率统计
        gen_loss_up_ratio = np.mean(gen_loss_up > 0.5)  # 假设0.5为阈值
        gen_loss_dn_ratio = np.mean(gen_loss_dn > 0.5)
        print(f"  丢包率 - 上行: {gen_loss_up_ratio:.4f}, 下行: {gen_loss_dn_ratio:.4f}")
        
        # 反归一化并打印实际值
        if qt_up and qt_dn:
            # 反归一化延迟值
            gen_del_up_real = qt_up.inverse_transform(gen_del_up.reshape(-1, 1)).flatten()
            gen_del_dn_real = qt_dn.inverse_transform(gen_del_dn.reshape(-1, 1)).flatten()
            
            gen_up_p5_real = np.percentile(gen_del_up_real, 5)
            gen_up_p50_real = np.percentile(gen_del_up_real, 50)
            gen_up_p95_real = np.percentile(gen_del_up_real, 95)
            gen_up_p99_real = np.percentile(gen_del_up_real, 99)
            gen_up_std_real = np.std(gen_del_up_real)
            
            gen_down_p5_real = np.percentile(gen_del_dn_real, 5)
            gen_down_p50_real = np.percentile(gen_del_dn_real, 50)
            gen_down_p95_real = np.percentile(gen_del_dn_real, 95)
            gen_down_p99_real = np.percentile(gen_del_dn_real, 99)
            gen_down_std_real = np.std(gen_del_dn_real)
            
            print("\n  生成样本实际值（反归一化后）：")
            print(f"  上行延迟 - p5: {gen_up_p5_real:.2f}ms, p50: {gen_up_p50_real:.2f}ms, p95: {gen_up_p95_real:.2f}ms, p99: {gen_up_p99_real:.2f}ms, std: {gen_up_std_real:.2f}ms")
            print(f"  下行延迟 - p5: {gen_down_p5_real:.2f}ms, p50: {gen_down_p50_real:.2f}ms, p95: {gen_down_p95_real:.2f}ms, p99: {gen_down_p99_real:.2f}ms, std: {gen_down_std_real:.2f}ms")
        
        # 对比生成样本与参考样本的统计特征（均使用双重归一化空间：QuantileTransformer + z-score）
        print("\n  统计特征对比（双重归一化空间：QuantileTransformer + z-score）：")
        print(f"  上行延迟p10差异: {abs(gen_up_p5 - ref_cond[CondIndex.UP_P1]):.4f}")
        print(f"  上行延迟p50差异: {abs(gen_up_p50 - ref_cond[CondIndex.UP_P50]):.4f}")
        print(f"  上行延迟p95差异: {abs(gen_up_p95 - ref_cond[CondIndex.UP_P90]):.4f}")
        print(f"  上行延迟p99差异: {abs(gen_up_p99 - ref_cond[CondIndex.UP_P99]):.4f}")
        print(f"  下行延迟p10差异: {abs(gen_down_p5 - ref_cond[CondIndex.DN_P1]):.4f}")
        print(f"  下行延迟p50差异: {abs(gen_down_p50 - ref_cond[CondIndex.DN_P50]):.4f}")
        print(f"  下行延迟p95差异: {abs(gen_down_p95 - ref_cond[CondIndex.DN_P90]):.4f}")
        print(f"  下行延迟p99差异: {abs(gen_down_p99 - ref_cond[CondIndex.DN_P99]):.4f}")
    
    # 10. 计算超限比例
    print("\n9. 计算超限比例：")
    if qt_up and len(selected_windows) > 0:
        # 计算真实数据的反归一化p95值
        p95_norm = np.array([window["cond"][2] for window in selected_windows])
        true_p95_raw = qt_up.inverse_transform(p95_norm.reshape(-1, 1)).flatten()
        
        # 计算生成数据的反归一化p95值
        gen_up_p95 = np.percentile(generated_samples_np[:, 0, :], 95, axis=1)
        gen_up_p95_real = qt_up.inverse_transform(gen_up_p95.reshape(-1, 1)).flatten()
        
        # 计算超限比例：生成的上行延迟p95 > 真实p95值 * 1.1
        over_limit_ratio = np.mean(gen_up_p95_real > true_p95_raw * 1.1)
        
        print(f"   生成数据上行延迟p95 > 真实p95 * 1.1的比例：{over_limit_ratio:.2%}")
        if over_limit_ratio > 0.05:
            print("   💡 警告：超限比例 > 5%，说明模型控制上限的能力需要改进。")
        else:
            print("   ✅ 超限比例 <= 5%，模型能够较好地控制上限。")
    
    # 11. 添加可视化功能
    print("\n10. 生成可视化结果：")
    import matplotlib.pyplot as plt
    
    # 设置中文字体支持
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # 创建可视化输出目录
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 绘制每个样本的延迟时间序列图（生成样本vs参考样本）
    plt.figure(figsize=(15, 10))
    for i in range(generated_samples_np.shape[0]):
        # 获取生成的样本数据
        gen_del_up = generated_samples_np[i, 0, :]  # 生成的上行延迟（归一化）
        gen_del_dn = generated_samples_np[i, 1, :]  # 生成的下行延迟（归一化）
        
        # 获取对应的参考样本数据
        window_meta = selected_windows[i]
        ref_window = window_meta["window"]
        state_id = int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
        
        # 处理参考样本数据
        if isinstance(ref_window, pd.DataFrame):
            # 如果window是DataFrame，优先使用原始延迟数据
            if "delay_up_origin" in ref_window.columns and "delay_down_origin" in ref_window.columns:
                # 使用原始未归一化的延迟数据
                ref_del_up = ref_window["delay_up_origin"].values  # 参考上行延迟（原始）
                ref_del_dn = ref_window["delay_down_origin"].values  # 参考下行延迟（原始）
            else:
                # 备用方案：使用归一化后的数据进行反归一化
                ref_del_up_norm = ref_window["del_up"].values  # 参考上行延迟（归一化）
                ref_del_dn_norm = ref_window["del_dn"].values  # 参考下行延迟（归一化）
                if qt_up and qt_dn:
                    ref_del_up = qt_up.inverse_transform(ref_del_up_norm.reshape(-1, 1)).flatten()
                    ref_del_dn = qt_dn.inverse_transform(ref_del_dn_norm.reshape(-1, 1)).flatten()
                else:
                    ref_del_up = ref_del_up_norm
                    ref_del_dn = ref_del_dn_norm
        else:
            # 处理从JSONL加载的窗口数据
            # 优先使用原始延迟数据，如果存在的话
            if ref_window and isinstance(ref_window[0], dict):
                if "delay_up_origin" in ref_window[0] and "delay_down_origin" in ref_window[0]:
                    # 使用原始未归一化的延迟数据
                    ref_del_up = np.array([row.get("delay_up_origin", 0.0) for row in ref_window[:100]])  # 参考上行延迟（原始）
                    ref_del_dn = np.array([row.get("delay_down_origin", 0.0) for row in ref_window[:100]])  # 参考下行延迟（原始）
                else:
                    # 备用方案：使用归一化后的数据进行反归一化
                    ref_del_up_norm = np.array([row.get("delay_up", 0.0) for row in ref_window[:100]])  # 参考上行延迟（归一化）
                    ref_del_dn_norm = np.array([row.get("delay_down", 0.0) for row in ref_window[:100]])  # 参考下行延迟（归一化）
                    if qt_up and qt_dn:
                        ref_del_up = qt_up.inverse_transform(ref_del_up_norm.reshape(-1, 1)).flatten()
                        ref_del_dn = qt_dn.inverse_transform(ref_del_dn_norm.reshape(-1, 1)).flatten()
                    else:
                        ref_del_up = ref_del_up_norm
                        ref_del_dn = ref_del_dn_norm
            else:
                # 处理异常情况
                ref_del_up = np.zeros_like(gen_del_up)
                ref_del_dn = np.zeros_like(gen_del_dn)
        
        # 对生成样本进行反归一化
        if qt_up and qt_dn:
            gen_del_up = qt_up.inverse_transform(gen_del_up.reshape(-1, 1)).flatten()
            gen_del_dn = qt_dn.inverse_transform(gen_del_dn.reshape(-1, 1)).flatten()
            ylabel = "延迟 (ms)"
        else:
            ylabel = "归一化延迟"
        
        plt.subplot(generated_samples_np.shape[0], 1, i+1)
        time_steps = np.arange(len(gen_del_up))
        
        # 绘制参考样本（原始数据）
        plt.plot(time_steps, ref_del_up, label="参考上行延迟（原始）", color="blue", linestyle="--")
        plt.plot(time_steps, ref_del_dn, label="参考下行延迟（原始）", color="red", linestyle="--")
        
        # 绘制生成样本（原始数据）
        plt.plot(time_steps, gen_del_up, label="生成上行延迟（生成）", color="blue", linestyle="-")
        plt.plot(time_steps, gen_del_dn, label="生成下行延迟（生成）", color="red", linestyle="-")
        
        plt.title(f"样本 {i+1} (state_id: {state_id}) 延迟时间序列（生成vs参考）")
        plt.xlabel("时间步")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
    
    # 保存时间序列图
    ts_path = viz_dir / "delay_time_series.png"
    plt.tight_layout()
    plt.savefig(ts_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   延迟时间序列图已保存到：{ts_path}")
    
    # 2. 绘制每个样本的丢包率时间序列图（生成样本vs参考样本）
    plt.figure(figsize=(15, 10))
    for i in range(generated_samples_np.shape[0]):
        gen_loss_up = generated_samples_np[i, 2, :]  # 生成的上行丢包率
        gen_loss_dn = generated_samples_np[i, 3, :]  # 生成的下行丢包率
        
        # 获取样本对应的state id和参考样本数据
        window_meta = selected_windows[i]
        ref_window = window_meta["window"]
        state_id = int(window_meta["cond"][11])
        
        # 处理参考样本数据
        if isinstance(ref_window, pd.DataFrame):
            # 如果window是DataFrame，优先使用原始丢包率数据
            if "loss_up_origin" in ref_window.columns and "loss_down_origin" in ref_window.columns:
                # 使用原始未归一化的丢包率数据
                ref_loss_up = ref_window["loss_up_origin"].values  # 参考上行丢包率（原始）
                ref_loss_dn = ref_window["loss_down_origin"].values  # 参考下行丢包率（原始）
            else:
                # 备用方案：使用已处理的丢包率数据
                ref_loss_up = ref_window["loss_up"].values  # 参考上行丢包率
                ref_loss_dn = ref_window["loss_dn"].values  # 参考下行丢包率
        else:
            # 处理从JSONL加载的窗口数据
            # 优先使用原始丢包率数据，如果存在的话
            if ref_window and isinstance(ref_window[0], dict):
                if "loss_up_origin" in ref_window[0] and "loss_down_origin" in ref_window[0]:
                    # 使用原始未归一化的丢包率数据
                    ref_loss_up = np.array([row.get("loss_up_origin", 0.0) for row in ref_window[:100]])  # 参考上行丢包率（原始）
                    ref_loss_dn = np.array([row.get("loss_down_origin", 0.0) for row in ref_window[:100]])  # 参考下行丢包率（原始）
                else:
                    # 备用方案：使用已处理的丢包率数据
                    ref_loss_up = np.array([row.get("loss_up", 0.0) for row in ref_window[:100]])  # 参考上行丢包率
                    ref_loss_dn = np.array([row.get("loss_dn", 0.0) for row in ref_window[:100]])  # 参考下行丢包率
            else:
                # 处理异常情况
                ref_loss_up = np.zeros_like(gen_loss_up)
                ref_loss_dn = np.zeros_like(gen_loss_dn)
        
        plt.subplot(generated_samples_np.shape[0], 1, i+1)
        time_steps = np.arange(len(gen_loss_up))
        
        # 绘制参考样本（原始丢包率，范围0-1）
        plt.plot(time_steps, ref_loss_up, label="参考上行丢包率（原始）", color="green", linestyle="--")
        plt.plot(time_steps, ref_loss_dn, label="参考下行丢包率（原始）", color="orange", linestyle="--")
        
        # 绘制生成样本（生成的丢包率，范围0-1）
        plt.plot(time_steps, gen_loss_up, label="生成上行丢包率（生成）", color="green", linestyle="-")
        plt.plot(time_steps, gen_loss_dn, label="生成下行丢包率（生成）", color="orange", linestyle="-")
        
        plt.title(f"样本 {i+1} (state_id: {state_id}) 丢包率时间序列（生成vs参考）")
        plt.xlabel("时间步")
        plt.ylabel("丢包率")
        plt.grid(True)
        plt.legend()
    
    # 保存丢包率时间序列图
    loss_ts_path = viz_dir / "loss_time_series.png"
    plt.tight_layout()
    plt.savefig(loss_ts_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   丢包率时间序列图已保存到：{loss_ts_path}")
    
    # 3. 绘制延迟分布直方图
    plt.figure(figsize=(15, 8))
    
    # 准备数据
    all_gen_del_up = generated_samples_np[:, 0, :].flatten()
    all_gen_del_dn = generated_samples_np[:, 1, :].flatten()
    all_gen_loss_up = generated_samples_np[:, 2, :].flatten()
    all_gen_loss_dn = generated_samples_np[:, 3, :].flatten()
    
    # 如果有QuantileTransformer，使用反归一化后的数据
    if qt_up and qt_dn:
        all_gen_del_up = qt_up.inverse_transform(all_gen_del_up.reshape(-1, 1)).flatten()
        all_gen_del_dn = qt_dn.inverse_transform(all_gen_del_dn.reshape(-1, 1)).flatten()
        delay_xlabel = "延迟 (ms)"
    else:
        delay_xlabel = "归一化延迟"
    
    # 上行延迟分布
    plt.subplot(2, 2, 1)
    plt.hist(all_gen_del_up, bins=50, alpha=0.7, color="blue", label="生成上行延迟")
    plt.title("上行延迟分布")
    plt.xlabel(delay_xlabel)
    plt.ylabel("频率")
    plt.grid(True)
    plt.legend()
    
    # 下行延迟分布
    plt.subplot(2, 2, 2)
    plt.hist(all_gen_del_dn, bins=50, alpha=0.7, color="red", label="生成下行延迟")
    plt.title("下行延迟分布")
    plt.xlabel(delay_xlabel)
    plt.ylabel("频率")
    plt.grid(True)
    plt.legend()
    
    # 上行丢包率分布
    plt.subplot(2, 2, 3)
    plt.hist(all_gen_loss_up, bins=50, alpha=0.7, color="green", label="生成上行丢包率")
    plt.title("上行丢包率分布")
    plt.xlabel("归一化丢包率")
    plt.ylabel("频率")
    plt.grid(True)
    plt.legend()
    
    # 下行丢包率分布
    plt.subplot(2, 2, 4)
    plt.hist(all_gen_loss_dn, bins=50, alpha=0.7, color="orange", label="生成下行丢包率")
    plt.title("下行丢包率分布")
    plt.xlabel("归一化丢包率")
    plt.ylabel("频率")
    plt.grid(True)
    plt.legend()
    
    # 保存直方图
    hist_path = viz_dir / "delay_loss_histograms.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   延迟和丢包率直方图已保存到：{hist_path}")
    
    # 12. 保存配置
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n11. 训练配置已保存到：{config_path}")
    
    # 13. 执行快速验证
    quick_verify(model, sde, val_windows, device=device)
    
    print("\n=== CSDI模型训练与测试完成 ===")


def quick_verify(model: CSDIModel, sde: SimpleSDE, val_windows: List[Dict[str, Any]], device: str = "cpu", train_windows: List[Dict[str, Any]] = None):
    """快速验证生成样本的p99值是否接近期望的条件值
    
    Args:
        model: 训练好的模型
        sde: SDE对象
        val_windows: 验证窗口列表
        device: 设备
    """
    import random
    print("\n=== 快速验证生成样本质量 ===")
    
    # 测试3个样本
    for i in range(3):
        if i >= len(val_windows):
            break
            
        # 获取完整条件向量
        full_cond = val_windows[i]["cond"]
        cond_tensor = torch.tensor([full_cond], dtype=torch.float32)
        
        # 生成样本
        sampled = generate_samples(model, sde, cond_tensor, num_inference_steps=50, device=device)
        
        # 计算生成样本的p99值
        gen_p99 = np.percentile(sampled[0, 0, :].cpu().numpy(), 99)
        expected_p99 = full_cond[CondIndex.UP_P99]  # 条件中的上行延迟p99
        
        # 打印结果
        print(f"样本 {i+1}:")
        print(f"  期望 p99: {expected_p99:.4f}")
        print(f"  实际 p99: {gen_p99:.4f}")
        print(f"  差距: {abs(gen_p99 - expected_p99):.4f}")
        # 避免除以零，添加1e-8作为保护
        if abs(expected_p99) < 1e-8:
            print("  相对差距: N/A (期望p99接近零)")
        else:
            print(f"  相对差距: {abs(gen_p99 - expected_p99) / abs(expected_p99) * 100:.2f}%")
        print()
    
    # 添加固定条件测试：测试低p99条件下的生成结果
    print("=== 固定条件测试（低p99） ===")
    if len(val_windows) > 0:
        # 获取一个条件向量并修改其p99值为较低值
        base_full_cond = val_windows[0]["cond"].copy()
        
        # 测试不同的低p99值
        test_p99_values = [0.1, 0.5, 1.0]
        
        for test_p99 in test_p99_values:
            # 设置低p99条件
            modified_full_cond = base_full_cond.copy()
            modified_full_cond[CondIndex.UP_P99] = test_p99  # 设置上行延迟p99为低数值
            fixed_cond_tensor = torch.tensor([modified_full_cond], dtype=torch.float32)
            
            # 生成样本
            sampled = generate_samples(model, sde, fixed_cond_tensor, num_inference_steps=50, device=device)
            
            # 计算生成样本的p99值
            gen_p99 = np.percentile(sampled[0, 0, :].cpu().numpy(), 99)
            
            # 打印结果
            print(f"固定条件测试（p99={test_p99:.1f}）:")
            print(f"  期望 p99: {test_p99:.4f}")
            print(f"  实际 p99: {gen_p99:.4f}")
            print(f"  差距: {abs(gen_p99 - test_p99):.4f}")
            print(f"  相对差距: {abs(gen_p99 - test_p99) / (abs(test_p99) + 1e-5) * 100:.2f}%")
            print()
    
    # 相关性分析：测试多个条件下的生成结果
    print("\n=== 条件相关性测试 ===")
    if train_windows and len(train_windows) > 5:
        # 选择5个不同p95条件的样本
        selected_windows = random.sample(train_windows, min(5, len(train_windows)))
        expected_p95_list = []
        generated_p95_list = []
        
        for window_meta in selected_windows:
            cond = window_meta["cond"]
            cond_tensor = torch.tensor([cond], dtype=torch.float32)
            
            # 生成样本
            sampled = generate_samples(model, sde, cond_tensor, num_inference_steps=50, device=device)
            
            # 计算生成样本的p95值
            gen_p95 = np.percentile(sampled[0, 0, :].cpu().numpy(), 95)
            expected_p95 = cond[2]  # 条件中的上行延迟p95
            
            expected_p95_list.append(expected_p95)
            generated_p95_list.append(gen_p95)
        
        # 计算相关性
        if len(expected_p95_list) >= 2:
            correlation = np.corrcoef(expected_p95_list, generated_p95_list)[0, 1]
            print(f"条件p95与生成p95的相关性: {correlation:.4f}")
            print(f"条件p95范围: [{min(expected_p95_list):.4f}, {max(expected_p95_list):.4f}]")
            print(f"生成p95范围: [{min(generated_p95_list):.4f}, {max(generated_p95_list):.4f}]")
            print()
            
            # 简单判断
            if correlation > 0.5:
                print("✅ 条件与生成结果存在较强正相关")
            elif correlation > 0.1:
                print("⚠️ 条件与生成结果存在弱正相关")
            else:
                print("❌ 条件与生成结果几乎无相关")
        print()


if __name__ == "__main__":
    main()
