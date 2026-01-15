#!/usr/bin/env python3

"""CSDI模型训练与测试脚本
整合到现有项目流程中
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from pathlib import Path
from einops import rearrange
import joblib

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子，确保结果可复现
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None


class CondIndex:
    """条件向量各维度索引常量

    定义条件向量中各统计特征和行为ID的索引位置
    条件向量共15维：1个behavior_id + 14个分位点特征
    """

    # 行为ID
    BEHAVIOR_ID = 0

    # 上行分位点（7个: p1, p10, p25, p50, p75, p90, p99）
    UP_P1, UP_P10, UP_P25, UP_P50, UP_P75, UP_P90, UP_P99 = 1, 2, 3, 4, 5, 6, 7

    # 下行分位点（7个: p1, p10, p25, p50, p75, p90, p99）
    DN_P1, DN_P10, DN_P25, DN_P50, DN_P75, DN_P90, DN_P99 = 8, 9, 10, 11, 12, 13, 14

    # 别名映射，便于调试和理解
    ALIAS_MAP = {
        BEHAVIOR_ID: "behavior_id",
        UP_P1: "up_p1",
        UP_P10: "up_p10",
        UP_P25: "up_p25",
        UP_P50: "up_p50",
        UP_P75: "up_p75",
        UP_P90: "up_p90",
        UP_P99: "up_p99",
        DN_P1: "dn_p1",
        DN_P10: "dn_p10",
        DN_P25: "dn_p25",
        DN_P50: "dn_p50",
        DN_P75: "dn_p75",
        DN_P90: "dn_p90",
        DN_P99: "dn_p99",
    }


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
        hidden_dim: int = 64,  # 隐藏层维度
        num_layers: int = 2,  # Transformer层数
        num_heads: int = 4,  # 注意力头数
        dropout: float = 0.1,
        num_behavior_ids: int = 8,  # 行为ID数量，默认为8
    ):
        """初始化CSDI模型

        Args:
            input_dim: 输入通道数（4个变量：up_delay, dn_delay, up_loss, dn_loss）
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout率
            num_behavior_ids: 行为ID数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_behavior_ids = num_behavior_ids

        # 嵌入层
        self.value_emb = nn.Linear(1, hidden_dim)  # 值嵌入
        self.time_emb = TimeEmbedding(hidden_dim)  # 时间嵌入
        self.feature_emb = nn.Embedding(input_dim, hidden_dim)  # 特征嵌入，区分4个变量
        self.behavior_emb = nn.Embedding(
            num_behavior_ids, hidden_dim
        )  # 行为ID嵌入，区分不同行为模式

        # 创建双路径注意力架构：时间注意力层 + 通道注意力层
        # 1. 时间注意力层：在每个通道内部捕捉时间动态
        self.time_attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 2. 通道注意力层：在每个时间步上捕捉通道间依赖
        self.channel_attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 输出头
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        time_stamps: torch.Tensor,
        mask: torch.Tensor,
        behavior_id: torch.Tensor = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            values: [B, K, L] 输入值，K=4变量，L=序列长度
            time_stamps: [B, L] 时间戳
            mask: [B, K, L] 掩码（1=已观测，0=待生成）
            behavior_id: [B] 行为ID，可选

        Returns:
            [B, K, L] 预测的噪声
        """
        B, K, L = values.shape

        # 展平为 [B, K, L, 1]
        x = values.unsqueeze(-1)

        # 嵌入 - 使用加法融合
        value_emb = self.value_emb(x)  # [B, K, L, H] - 值嵌入
        time_emb = self.time_emb(time_stamps).unsqueeze(1)  # [B, 1, L, H] - 时间嵌入

        # 创建feature_ids，使用values的设备
        feature_ids = torch.arange(K, device=values.device).view(1, K, 1)  # [1, K, 1]
        feature_emb = self.feature_emb(feature_ids)  # [1, K, 1, H] - 特征嵌入

        # 扩展time_emb和feature_emb到[B, K, L, H]
        time_emb_expanded = time_emb.expand(B, K, L, -1)  # [B, K, L, H]
        feature_emb_expanded = feature_emb.expand(B, K, L, -1)  # [B, K, L, H]

        # 嵌入融合：使用加法
        h = value_emb + time_emb_expanded + feature_emb_expanded  # [B, K, L, H]

        # 融合行为ID嵌入
        if behavior_id is not None:
            # 使用广播机制，避免显式扩展
            # [B] -> [B, H] -> [B, 1, 1, H]，自动广播到[B, K, L, H]
            beh_emb = self.behavior_emb(behavior_id).unsqueeze(1).unsqueeze(2)
            h = h + beh_emb

        # 双路径注意力计算：时间注意力 + 通道注意力交替进行
        for i in range(len(self.time_attn_layers)):
            # 1. 时间注意力：在每个通道内部捕捉时间动态
            # 将每个通道作为独立序列，序列长度为L
            h_time = rearrange(h, "b k l h -> (b k) l h")
            h_time = self.time_attn_layers[i](h_time)
            # 重排回原始形状
            h = rearrange(h_time, "(b k) l h -> b k l h", b=B, k=K)

            # 2. 通道注意力：在每个时间步上捕捉通道间依赖
            # 将每个时间步的4个通道作为序列，序列长度为K=4
            h_channel = rearrange(h, "b k l h -> (b l) k h")
            h_channel = self.channel_attn_layers[i](h_channel)
            # 重排回原始形状
            h = rearrange(h_channel, "(b l) k h -> b k l h", b=B, l=L)

        # 输出预测
        # 将h重排为 [B*K, L, H] 以便通过输出头
        h = rearrange(h, "b k l h -> (b k) l h")
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

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算边际分布

        Args:
            x0: [B, K, L] 初始值
            t: [B] 时间步

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 边际分布的均值和标准差
        """
        # 数值稳定计算：添加clamp防止log_mean_coeff过于极端
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        # 添加数值稳定性约束
        log_mean_coeff = torch.clamp(
            log_mean_coeff, min=-20.0, max=20.0
        )  # 防止指数溢出

        mean = torch.exp(log_mean_coeff).unsqueeze(-1).unsqueeze(-1) * x0
        # 添加小epsilon防止sqrt(0)问题
        std = (
            torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff).clamp(min=0.0) + 1e-8)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        return mean, std

    def denoise_step(
        self,
        x: torch.Tensor,
        pred_noise: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
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
        log_mean_coeff_t = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        log_mean_coeff_t = torch.clamp(
            log_mean_coeff_t, min=-20.0, max=20.0
        )  # 防止指数溢出

        alpha_t = (
            torch.exp(2.0 * log_mean_coeff_t).unsqueeze(-1).unsqueeze(-1)
        )  # [B, 1, 1]
        # 添加数值稳定性约束，确保alpha_t在合理范围内
        alpha_t = torch.clamp(alpha_t, min=1e-8, max=1.0 - 1e-8)
        # 添加小epsilon防止除以零
        std_t = torch.sqrt(1.0 - alpha_t + 1e-8)  # 当前时间步的标准差

        # 计算score：-pred_noise / std_t（根据SDE理论，score是负的噪声除以标准差）
        score = -pred_noise / (std_t + 1e-8)  # 添加epsilon防止除以零

        # 计算β(t) = β_min + t*(β_max - β_min)
        beta_t = (
            (self.beta_min + t * (self.beta_max - self.beta_min))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # [B, 1, 1]
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
            x_next += (
                diffusion_coeff * noise * torch.sqrt(torch.abs(dt))
            )  # 使用欧拉-马尔可夫方法添加噪声

        return x_next


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
        cond = window_meta[
            "cond"
        ]  # [15] 条件向量（1个行为ID + 7个up分位点 + 7个dn分位点）

        # 处理窗口数据
        window = window_meta["window"]

        # 提取4个变量的值 [4, 100]
        window_values = np.zeros((4, self.seq_len))

        try:
            if isinstance(window, pd.DataFrame):
                # 如果window是DataFrame，直接提取值
                window_values = window[
                    ["delay_up_rs", "delay_down_rs", "loss_up", "loss_dn"]
                ].values.T  # [4, 100]
            elif isinstance(window, dict):
                # 处理嵌套结构
                nested_window = window.get("window", [])
                if isinstance(nested_window, list):
                    # 从嵌套列表中提取值
                    for i, row in enumerate(nested_window):
                        if i >= self.seq_len:
                            break
                        window_values[0, i] = row.get("delay_up_rs", 0.0)  # up_delay_rs
                        window_values[1, i] = row.get(
                            "delay_down_rs", 0.0
                        )  # dn_delay_rs
                        window_values[2, i] = row.get("loss_up", 0.0)  # up_loss
                        window_values[3, i] = row.get("loss_dn", 0.0)  # dn_loss
                else:
                    # 处理嵌套结构中的其他类型
                    print(
                        f"Warning: Unexpected nested window type: {type(nested_window)}"
                    )
            elif isinstance(window, list):
                # 处理从JSONL加载的列表格式窗口数据
                for i, row in enumerate(window):
                    if i >= self.seq_len:
                        break
                    window_values[0, i] = row.get("delay_up_rs", 0.0)  # up_delay_rs
                    window_values[1, i] = row.get("delay_down_rs", 0.0)  # dn_delay_rs
                    window_values[2, i] = row.get("loss_up", 0.0)  # up_loss
                    window_values[3, i] = row.get("loss_dn", 0.0)  # dn_loss
            else:
                # 处理其他类型
                print(f"Warning: Unexpected window type: {type(window)}")
        except Exception as e:
            print(f"Error processing window data: {e}")
            # 保持默认的全零值

        # 验证数据形状，确保是[4, seq_len]
        if window_values.shape != (4, self.seq_len):
            print(
                f"Warning: Window values shape mismatch: {window_values.shape} expected {(4, self.seq_len)}"
            )
            # 重新初始化并填充可用数据
            temp_values = np.zeros((4, self.seq_len))
            if len(window_values.shape) == 2:
                rows = min(window_values.shape[0], 4)
                cols = min(window_values.shape[1], self.seq_len)
                temp_values[:rows, :cols] = window_values[:rows, :cols]
            window_values = temp_values

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

        # 填充原始窗口数据
        values[:, : self.seq_len] = window_values  # [4, 100] - 真实完整序列

        # --- 动态关键点锚点策略 ---
        # 在原始序列的关键位置设置锚点，其余位置为待生成
        mask = np.zeros((4, self.L_total), dtype=np.float32)  # 初始化为全0
        input_values = np.zeros_like(values)  # 初始化为全0

        # 对每个通道动态选择锚点位置
        for k in range(4):
            # 获取当前通道的数据
            channel_data = values[k, :]

            # 计算最小值位置
            min_pos = np.argmin(channel_data)

            # 计算最大值位置
            max_pos = np.argmax(channel_data)

            # 计算中位数位置 - 找到最接近中位数的值的索引
            median_val = np.median(channel_data)
            median_pos = np.abs(channel_data - median_val).argmin()

            # 收集锚点位置，去重
            anchor_positions = list(set([min_pos, median_pos, max_pos]))

            # 如果去重后少于3个，补充中间位置
            if len(anchor_positions) < 3:
                anchor_positions.append(self.L_total // 2)
                anchor_positions = list(set(anchor_positions))

            # 确保锚点位置数量为3
            anchor_positions = anchor_positions[:3]

            # 设置锚点
            input_values[k, anchor_positions] = values[
                k, anchor_positions
            ]  # 锚点位置使用真实值
            mask[k, anchor_positions] = 1.0  # 锚点位置标记为已观测

        # 时间戳 [0.0, ..., 1.0]，统一时间范围为[0, 1]
        time_stamps = np.linspace(0.0, 1.0, self.L_total)

        # 提取行为ID
        behavior_id = cond[0]  # 条件向量的第0位是行为ID

        # 返回numpy数组，在collate_fn中统一转换为torch张量，提高性能
        return {
            "values": values,  # 真实完整序列 [4, 100] - 用于计算损失
            "input_values": input_values,  # 带锚点的输入序列 [4, 100]
            "mask": mask,  # mask指示哪些点是已观测(1)或待生成(0) [4, 100]
            "time_stamps": time_stamps,  # 时间戳 [100]
            "behavior_id": behavior_id,  # 行为ID
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """批次处理函数

        Args:
            batch: 样本列表

        Returns:
            Dict[str, torch.Tensor]: 批次数据
        """
        # 统一将numpy数组转换为torch张量并堆叠，提高性能
        return {
            "values": torch.tensor(
                np.stack([sample["values"] for sample in batch]), dtype=torch.float32
            ),  # 真实完整序列
            "input_values": torch.tensor(
                np.stack([sample["input_values"] for sample in batch]),
                dtype=torch.float32,
            ),  # 带锚点的输入序列
            "mask": torch.tensor(
                np.stack([sample["mask"] for sample in batch]), dtype=torch.float32
            ),  # 掩码
            "time_stamps": torch.tensor(
                np.stack([sample["time_stamps"] for sample in batch]),
                dtype=torch.float32,
            ),  # 时间戳
            "behavior_id": torch.tensor(
                np.array([sample["behavior_id"] for sample in batch]), dtype=torch.long
            ),  # 行为ID
        }


def create_data_loader(
    windows: List[Dict[str, Any]], batch_size: int = 32, shuffle: bool = True
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
        collate_fn=NetworkTraceCSDIDataset.collate_fn,
    )


def load_windows_from_jsonl(file_path: str) -> list:
    """从JSONL文件加载窗口数据

    Args:
        file_path: JSONL文件路径

    Returns:
        list: 窗口数据列表
    """
    windows = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                window_meta = json.loads(line)
                windows.append(window_meta)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to decode line in {file_path}: {e}")
                continue

    # 检查数据范围
    if windows:
        # 检查条件向量的范围
        cond_values = np.array([w["cond"] for w in windows])
        print(
            f"[Data Check] 条件向量范围 - min: {cond_values.min():.4f}, max: {cond_values.max():.4f}, mean: {cond_values.mean():.4f}, std: {cond_values.std():.4f}"
        )

        # 打印关键统计量的范围
        up_p1_values = cond_values[:, CondIndex.UP_P1]
        up_p99_values = cond_values[:, CondIndex.UP_P99]
        up_p50_values = cond_values[:, CondIndex.UP_P50]
        dn_p1_values = cond_values[:, CondIndex.DN_P1]
        dn_p99_values = cond_values[:, CondIndex.DN_P99]
        dn_p50_values = cond_values[:, CondIndex.DN_P50]

        print(
            f"[Data Check] up_p1 范围 - min: {up_p1_values.min():.4f}, max: {up_p1_values.max():.4f}"
        )
        print(
            f"[Data Check] up_p50 范围 - min: {up_p50_values.min():.4f}, max: {up_p50_values.max():.4f}"
        )
        print(
            f"[Data Check] up_p99 范围 - min: {up_p99_values.min():.4f}, max: {up_p99_values.max():.4f}"
        )
        print(
            f"[Data Check] dn_p1 范围 - min: {dn_p1_values.min():.4f}, max: {dn_p1_values.max():.4f}"
        )
        print(
            f"[Data Check] dn_p50 范围 - min: {dn_p50_values.min():.4f}, max: {dn_p50_values.max():.4f}"
        )
        print(
            f"[Data Check] dn_p99 范围 - min: {dn_p99_values.min():.4f}, max: {dn_p99_values.max():.4f}"
        )

    return windows


def compare_data_distributions(train_windows, val_windows):
    """比较训练集和验证集的数据分布

    Args:
        train_windows: 训练集窗口数据
        val_windows: 验证集窗口数据
    """
    print("\n=== 比较训练集和验证集数据分布 ===")

    # 提取条件向量
    train_conds = np.array([w["cond"] for w in train_windows])
    val_conds = np.array([w["cond"] for w in val_windows])

    # 比较条件向量的统计特征
    print(f"训练集条件向量形状: {train_conds.shape}")
    print(f"验证集条件向量形状: {val_conds.shape}")

    # 计算关键统计量
    train_mean = train_conds.mean()
    val_mean = val_conds.mean()
    train_std = train_conds.std()
    val_std = val_conds.std()

    print(
        f"\n条件向量均值: 训练集={train_mean:.4f}, 验证集={val_mean:.4f}, 差异={abs(train_mean - val_mean):.4f}"
    )
    print(
        f"条件向量标准差: 训练集={train_std:.4f}, 验证集={val_std:.4f}, 差异={abs(train_std - val_std):.4f}"
    )

    # 比较关键分位点的分布
    key_indices = [
        (CondIndex.UP_P10, "up_p10"),
        (CondIndex.UP_P50, "up_p50"),
        (CondIndex.UP_P99, "up_p99"),
        (CondIndex.DN_P10, "dn_p10"),
        (CondIndex.DN_P50, "dn_p50"),
        (CondIndex.DN_P99, "dn_p99"),
    ]

    for index, name in key_indices:
        train_vals = train_conds[:, index]
        val_vals = val_conds[:, index]

        train_mean = train_vals.mean()
        val_mean = val_vals.mean()
        train_std = train_vals.std()
        val_std = val_vals.std()

        print(
            f"\n{name}: 训练集均值={train_mean:.4f}, 验证集均值={val_mean:.4f}, 差异={abs(train_mean - val_mean):.4f}"
        )
        print(
            f"      训练集标准差={train_std:.4f}, 验证集标准差={val_std:.4f}, 差异={abs(train_std - val_std):.4f}"
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
        patience: int = 200,  # 大幅增加patience，让模型充分训练
        min_epochs: int = 500,  # 增加最小训练轮数
        delta: float = 1e-4,  # 最小改进阈值，保持不变
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

    def calculate_quantile_loss(self, x0_pred, cond):
        """计算分位点损失，只计算延迟分位点损失

        Args:
            x0_pred: [B, 4, 100] 重建的样本
            cond: [B, cond_dim] 条件向量

        Returns:
            torch.Tensor: 分位点损失
        """
        # 计算重建样本的分位点 (p10, p50, p90, p99) - 一次性计算，提高性能
        # 定义要计算的分位点
        quantiles = torch.tensor([0.1, 0.5, 0.9, 0.99], device=x0_pred.device)

        # 上行延迟 (feature 0)
        gen_up_quantiles = torch.quantile(x0_pred[:, 0, :], quantiles, dim=-1)  # [4, B]
        gen_p10_up, gen_p50_up, gen_p90_up, gen_p99_up = (
            gen_up_quantiles[0],
            gen_up_quantiles[1],
            gen_up_quantiles[2],
            gen_up_quantiles[3],
        )

        # 下行延迟 (feature 1)
        gen_dn_quantiles = torch.quantile(x0_pred[:, 1, :], quantiles, dim=-1)  # [4, B]
        gen_p10_dn, gen_p50_dn, gen_p90_dn, gen_p99_dn = (
            gen_dn_quantiles[0],
            gen_dn_quantiles[1],
            gen_dn_quantiles[2],
            gen_dn_quantiles[3],
        )

        # 从cond中提取目标分位点
        target_p10_up = cond[:, CondIndex.UP_P10]  # 目标上行p10
        target_p50_up = cond[:, CondIndex.UP_P50]  # 目标上行p50
        target_p90_up = cond[:, CondIndex.UP_P90]  # 目标上行p90
        target_p99_up = cond[:, CondIndex.UP_P99]  # 目标上行p99

        target_p10_dn = cond[:, CondIndex.DN_P10]  # 目标下行p10
        target_p50_dn = cond[:, CondIndex.DN_P50]  # 目标下行p50
        target_p90_dn = cond[:, CondIndex.DN_P90]  # 目标下行p90
        target_p99_dn = cond[:, CondIndex.DN_P99]  # 目标下行p99

        # 计算分位点损失 (MSE loss) - 只计算延迟损失
        quantile_loss = (
            # 上行时延损失
            F.mse_loss(gen_p10_up, target_p10_up)
            + F.mse_loss(gen_p50_up, target_p50_up)
            + F.mse_loss(gen_p90_up, target_p90_up)
            + F.mse_loss(gen_p99_up, target_p99_up)
            +
            # 下行时延损失
            F.mse_loss(gen_p10_dn, target_p10_dn)
            + F.mse_loss(gen_p50_dn, target_p50_dn)
            + F.mse_loss(gen_p90_dn, target_p90_dn)
            + F.mse_loss(gen_p99_dn, target_p99_dn)
        )

        return quantile_loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行一个训练步骤

        Args:
            batch: 批次数据

        Returns:
            torch.Tensor: 损失值
        """
        # 获取数据
        input_values = batch["input_values"].to(
            self.device
        )  # [B, 4, 100] - 带锚点的输入序列
        mask = batch["mask"].to(self.device)  # [B, 4, 100] - 掩码（1=已观测，0=待生成）
        time_stamps = batch["time_stamps"].to(self.device)  # [B, 100] - 时间戳
        target_values = batch["values"].to(
            self.device
        )  # [B, 4, 100] - 真实完整序列（用于计算损失）
        behavior_id = batch["behavior_id"].to(self.device)  # [B] - 行为ID

        B = input_values.shape[0]

        # 采样随机时间步
        t = torch.rand(B, device=self.device) * (1 - 1e-4) + 1e-4

        # 只对缺失位置（mask==0）加噪，观测位置（mask==1）保持原值
        noise = torch.randn_like(target_values)
        mean, std = self.sde.marginal_prob(target_values, t)
        perturbed_values = torch.where(
            mask == 0,
            mean + std * noise,
            target_values,  # 观测位置保持原值
        )

        # 训练时随机dropout behavior_id，防止过度拟合 (10%概率)
        use_behavior_id = behavior_id
        if self.model.training and torch.rand(1).item() < 0.1:
            use_behavior_id = None

        # 模型预测噪声
        pred_noise = self.model(perturbed_values, time_stamps, mask, use_behavior_id)

        # 标准MSE损失：只对缺失位置计算噪声损失
        noise_loss = F.mse_loss(pred_noise[mask == 0], noise[mask == 0])

        # 添加debug信息 - 降低频率到0.1%，减少日志输出
        if torch.rand(1) < 0.001:
            print(f"[Debug] Train Step - Noise Loss: {noise_loss.item():.4f}")

        return noise_loss

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
            input_values = batch["input_values"].to(
                self.device
            )  # [B, 4, 100] - 带锚点的输入序列
            mask = batch["mask"].to(
                self.device
            )  # [B, 4, 100] - 掩码（1=已观测，0=待生成）
            time_stamps = batch["time_stamps"].to(self.device)  # [B, 100] - 时间戳
            target_values = batch["values"].to(
                self.device
            )  # [B, 4, 100] - 真实完整序列（用于计算损失）
            behavior_id = batch["behavior_id"].to(self.device)  # [B] - 行为ID

            B = input_values.shape[0]

            # 使用3个固定时间步代替5个，减少计算负担
            # 选择具有代表性的时间步，覆盖从早期到晚期的扩散过程
            fixed_times = torch.tensor([0.2, 0.5, 0.8], device=self.device)

            total_noise_loss = 0.0
            num_time_steps = len(fixed_times)

            for t_val in fixed_times:
                # 使用固定时间步，对整个批次使用相同的时间步
                t = t_val.repeat(B)

                # 只对缺失位置（mask==0）加噪，观测位置（mask==1）保持原值
                noise = torch.randn_like(target_values)
                mean, std = self.sde.marginal_prob(target_values, t)
                perturbed_values = torch.where(
                    mask == 0,
                    mean + std * noise,
                    target_values,  # 观测位置保持原值
                )

                # 模型预测噪声 - 验证时始终传递behavior_id
                pred_noise = self.model(
                    perturbed_values, time_stamps, mask, behavior_id
                )

                # 只对缺失位置计算噪声损失
                noise_loss = F.mse_loss(pred_noise[mask == 0], noise[mask == 0])
                total_noise_loss += noise_loss.item()

            # 计算平均损失
            avg_noise_loss = total_noise_loss / num_time_steps

        self.model.train()
        return torch.tensor(avg_noise_loss, device=self.device)

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
            print(f"⚠️ 训练轮数 {epoch + 1}/{self.min_epochs}，未达到早停最小轮数要求")
            return False

        # 相对改进幅度计算
        if self.best_val_loss < float("inf"):
            improvement = (self.best_val_loss - val_loss) / self.best_val_loss
        else:
            improvement = float("inf")

        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            print(f"✅ 验证损失改善: {improvement * 100:.2f}%")
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
        learning_rate: float = 1e-4,
    ) -> None:
        """改进的训练函数

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
        """
        # 确保num_epochs至少等于min_epochs
        if num_epochs < self.min_epochs:
            print(f"⚠️ 配置的num_epochs ({num_epochs}) < min_epochs ({self.min_epochs})")
            print(f"✅ 自动将num_epochs调整为min_epochs ({self.min_epochs})")
            num_epochs = self.min_epochs

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # 学习率衰减因子
            patience=3,  # 3个epoch无改善则降低学习率
            min_lr=1e-6,  # 最小学习率
            verbose=True,  # 打印学习率变化
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
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

            # 保存检查点
            import os

            checkpoint_dir = "./checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 每10个epoch或验证损失改善时保存检查点
            if (epoch + 1) % 10 == 0 or avg_val_loss < self.best_val_loss:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"csdi_epoch_{epoch + 1}_loss_{avg_val_loss:.4f}.pth",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "val_loss": avg_val_loss,
                        "train_loss": avg_train_loss,
                    },
                    checkpoint_path,
                )
                print(f"💾 检查点已保存: {checkpoint_path}")

            # 早停检查
            if self.check_early_stopping(avg_val_loss, epoch):
                print(f"早停在epoch {epoch + 1}触发")
                break

            # 学习率调度
            scheduler.step(avg_val_loss)


class CSDISampler:
    """CSDI模型采样器

    负责从CSDI模型中生成样本，只支持基于锚点的生成策略

    生成策略：
    1. 基于范例的动态锚点：从输入范例中提取局部极大/极小值作为锚点，保留原始样本的具体形状和突发模式
    2. 支持可选的行为ID指导，用于控制生成样本的行为模式
    """

    def __init__(self, model: CSDIModel, sde: SimpleSDE, device: str = "cpu"):
        """初始化CSDI采样器

        Args:
            model: 训练好的CSDI模型
            sde: SDE对象
            device: 采样设备
        """
        self.model = model.to(device)
        self.sde = sde
        self.device = device

    def _select_dynamic_anchors(self, exemplar, behavior_id, L_total):
        """
        为每个通道独立选择锚点。
        - 延迟通道 (0,1): 锚定 p10, p50, p99 + peaks/valleys.
        - 丢包通道 (2,3): 锚定所有非零点 + 起终点.
        """
        import numpy as np
        from scipy.signal import find_peaks
        
        B, K, L = exemplar.shape
        device = exemplar.device
        MAX_ANCHORS = 35

        all_anchor_vals = torch.full((B, K, MAX_ANCHORS), 0.0, device=device)
        all_anchor_pos = torch.full(
            (B, MAX_ANCHORS), -1, dtype=torch.long, device=device
        )
        all_masks = torch.zeros((B, K, MAX_ANCHORS), dtype=torch.bool, device=device)

        for b in range(B):
            # --- 处理延迟通道 (0: up, 1: down) ---
            for k in [0, 1]:
                series = exemplar[b, k, :]  # [L]
                
                # 1. 强制锚定关键分位数
                q10_val = torch.quantile(series, 0.1)
                q50_val = torch.quantile(series, 0.5)
                q99_val = torch.quantile(series, 0.99)
                
                def find_closest_idx(s, val):
                    return torch.argmin(torch.abs(s - val)).item()
                    
                quantile_positions = [
                    find_closest_idx(series, q10_val),
                    find_closest_idx(series, q50_val),
                    find_closest_idx(series, q99_val)
                ]
                quantile_values = [q10_val.item(), q50_val.item(), q99_val.item()]
                
                # 2. 补充：使用 find_peaks 找局部极值 (降低 prominence 阈值)
                series_np = series.cpu().numpy()
                std_val = np.std(series_np)
                if std_val > 1e-5:  # 避免除零
                    prominence = max(std_val * 0.1, 0.1)  # 设置一个最小绝对阈值
                    peaks, _ = find_peaks(series_np, prominence=prominence)
                    valleys, _ = find_peaks(-series_np, prominence=prominence)
                    peak_valley_positions = np.concatenate((peaks, valleys))
                else:
                    peak_valley_positions = np.array([])
                
                # 3. 合并所有候选点，并去重
                all_candidate_pos = np.unique(np.concatenate((
                    [0],                    # 起点
                    quantile_positions,     # 关键分位数
                    peak_valley_positions,  # 局部极值
                    [L - 1]                 # 终点
                )))
                
                # 4. 确保不超出 MAX_ANCHORS
                selected_pos = all_candidate_pos[:MAX_ANCHORS]
                n_anchors = len(selected_pos)
                
                # 5. 填充结果
                all_anchor_pos[b, :n_anchors] = torch.from_numpy(selected_pos).long().to(device)
                all_anchor_vals[b, k, :n_anchors] = series[selected_pos]
                all_masks[b, k, :n_anchors] = True

            # --- 处理丢包率通道 (2: up_loss, 3: dn_loss) ---
            for k in [2, 3]:
                series = exemplar[b, k, :]
                # 找出所有非零的丢包点
                nonzero_indices = torch.nonzero(series, as_tuple=True)[0].cpu().numpy()
                all_candidate_pos = np.unique(np.concatenate((
                    [0],
                    nonzero_indices,
                    [L - 1]
                )))
                selected_pos = all_candidate_pos[:MAX_ANCHORS]
                n_anchors = len(selected_pos)
                
                all_anchor_pos[b, :n_anchors] = torch.from_numpy(selected_pos).long().to(device)
                all_anchor_vals[b, k, :n_anchors] = series[selected_pos]
                all_masks[b, k, :n_anchors] = True

        return all_anchor_vals, all_anchor_pos, all_masks

    def sample_from_exemplar(
        self,
        exemplar: torch.Tensor,
        num_inference_steps: int = 100,  # Increased steps for smoother denoising
    ) -> torch.Tensor:
        """基于范例生成新样本

        Args:
            exemplar: [B, 4, 100] 范例样本
            num_inference_steps: 推理步数

        Returns:
            [B, 4, 100] 生成的样本
        """
        B, K, L = exemplar.shape
        L_total = L

        # 选择动态锚点，不使用behavior_id
        anchor_values, anchor_positions, mask = self._select_dynamic_anchors(
            exemplar, None, L_total
        )

        # 调用采样方法，不传递behavior_id，传递形状信息
        return self._sample_generic(
            B,
            K,
            L,
            anchor_values,
            anchor_positions,
            mask,
            behavior_id=None,
            num_inference_steps=num_inference_steps,
        )

    def sample_from_exemplar_with_id(
        self,
        exemplar: torch.Tensor,
        behavior_id: torch.Tensor or int,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """基于范例和行为ID生成新样本

        Args:
            exemplar: [B, 4, 100] 范例样本
            behavior_id: [B] 或 int 行为ID
            num_inference_steps: 推理步数

        Returns:
            [B, 4, 100] 生成的样本
        """
        B, K, L = exemplar.shape
        L_total = L

        # 选择动态锚点，使用behavior_id定制策略
        anchor_values, anchor_positions, mask = self._select_dynamic_anchors(
            exemplar, behavior_id, L_total
        )

        # 创建behavior_id张量
        if isinstance(behavior_id, int):
            # 如果是单个ID，为每个样本创建相同的ID
            bid_tensor = torch.full(
                (B,), behavior_id, device=exemplar.device, dtype=torch.long
            )
        else:
            # 否则直接使用传入的张量
            bid_tensor = behavior_id.to(exemplar.device)

        # 调用采样方法，传递behavior_id，传递形状信息
        return self._sample_generic(
            B,
            K,
            L,
            anchor_values,
            anchor_positions,
            mask,
            behavior_id=bid_tensor,
            num_inference_steps=num_inference_steps,
        )

    def _sample_generic(
        self,
        B: int,
        K: int,
        L: int,
        anchor_values: torch.Tensor,
        anchor_positions: torch.Tensor,
        mask: torch.Tensor,
        behavior_id: torch.Tensor = None,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """通用采样方法，基于锚点生成样本

        Args:
            B: 批次大小
            K: 特征数量
            L: 序列长度
            anchor_values: [B, K, max_num_points] 张量，填充后的锚点值
            anchor_positions: [B, max_num_points] 张量，填充后的位置索引
            mask: [B, K, max_num_points] 布尔张量，指示哪些是有效锚点
            behavior_id: [B] 行为ID，可选
            num_inference_steps: 推理步数

        Returns:
            [B, 4, 100] 生成的样本
        """
        # 确保输入和掩码在同一设备
        anchor_values = anchor_values.to(self.device)
        anchor_positions = anchor_positions.to(self.device)
        mask = mask.to(self.device)
        if behavior_id is not None:
            behavior_id = behavior_id.to(self.device)

        L_total = L  # 只保留原始序列长度

        # 统一时间范围为[0, 1]
        # time_stamps: [L] → [B, L]，与模型forward期望的维度匹配
        time_stamps = torch.linspace(0.0, 1.0, L_total, device=self.device)
        time_stamps = time_stamps.unsqueeze(0).repeat(B, 1)

        # 创建输入序列和掩码（兼容原来的模型forward接口）
        input_seq = torch.zeros(B, K, L, device=self.device)

        # 创建兼容的掩码，将所有锚点位置标记为已观测
        compat_mask = torch.zeros(B, K, L, device=self.device)

        # 遍历每个样本，将所有锚点位置标记为已观测
        for b in range(B):
            # 获取当前样本的有效锚点位置
            valid_pos = anchor_positions[b][anchor_positions[b] != -1]
            if len(valid_pos) > 0:
                # 只对延迟维度（K=0: up, K=1: down）应用锚点标记
                for k in range(2):  # 只处理前两个通道
                    compat_mask[b, k, valid_pos] = 1.0

        # 初始化输入序列：将锚点位置的值设为锚点值
        for b in range(B):
            valid_pos = anchor_positions[b][anchor_positions[b] != -1]
            if len(valid_pos) > 0:
                # 只对延迟维度（K=0: up, K=1: down）应用锚点
                for k in range(2):
                    input_seq[b, k, valid_pos] = anchor_values[b, k, : len(valid_pos)]

        # 初始化：使用带锚点的输入序列作为初始值
        x_init = input_seq

        # 在初始状态上添加高斯噪声，避免完全确定的初始值
        # 降低噪声强度，适合RobustScaler缩放后的数据
        x = x_init + torch.randn_like(x_init) * 0.2

        # 生成时间步序列，从1.0到0.0递减
        time_steps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=self.device
        )

        # 逆SDE步骤
        self.model.eval()
        with torch.no_grad():
            for i in range(num_inference_steps):
                # 计算当前和下一个时间步
                t = time_steps[i].repeat(B)
                next_t = time_steps[i + 1].repeat(B)

                # 模型预测噪声，传入行为ID
                pred_noise = self.model(x, time_stamps, compat_mask, behavior_id)

                # 更新x，使用确定性去噪步骤
                x = self.sde.denoise_step(x, pred_noise, t, next_t, deterministic=True)

                # --- 关键：在每一步采样后，强制对齐锚点 ---
                # anchor_pos: [B, MAX_ANCHORS]
                # anchor_vals: [B, 4, MAX_ANCHORS]
                # x: [B, 4, L]

                for b in range(x.shape[0]):
                    valid_pos = anchor_positions[b][
                        anchor_positions[b] != -1
                    ]  # 过滤掉填充的-1
                    if len(valid_pos) > 0:
                        # 只对延迟维度（K=0: up, K=1: down）应用锚点
                        # anchor_values[b, :2, :len(valid_pos)] 是 [2, num_valid]
                        x[b, :2, valid_pos] = anchor_values[b, :2, : len(valid_pos)]

                # 添加范围裁剪，使用更宽的范围以适应RobustScaler缩放后的数据
                x = torch.clamp(x, -5.0, 5.0)

        # 生成最终样本
        generated = x[:, :, :L]

        return generated


def train_csdi(
    train_windows: List[Dict[str, Any]],
    val_windows: List[Dict[str, Any]],
    config: Dict[str, Any],
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
    num_epochs = config.get(
        "num_epochs", 1000
    )  # 大幅增加训练轮数到1000，确保模型充分训练
    min_epochs = config.get("min_epochs", 500)  # 早停最小轮数，可通过配置控制
    learning_rate = config.get("learning_rate", 5e-4)  # 调整学习率，从1e-5提高到5e-4
    hidden_dim = config.get("hidden_dim", 256)  # 增大模型容量，从128调整为256
    num_layers = config.get("num_layers", 6)  # 增加Transformer层数，从4调整为6
    num_heads = config.get("num_heads", 8)
    dropout = config.get("dropout", 0.1)  # 保持0.1，减少训练稳定性影响
    # 支持CUDA、MPS和CPU设备
    default_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = config.get("device", default_device)

    # 创建数据加载器
    train_loader = create_data_loader(train_windows, batch_size, shuffle=True)
    val_loader = create_data_loader(val_windows, batch_size, shuffle=False)

    # 初始化模型和SDE
    model = CSDIModel(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )

    # 从配置中获取SDE参数
    beta_min = config.get("beta_min", 0.1)
    beta_max = config.get("beta_max", 15.0)
    sde = SimpleSDE(beta_min=beta_min, beta_max=beta_max)

    # 初始化训练器，传递min_epochs参数
    trainer = CSDITrainer(model, sde, device=device, min_epochs=min_epochs)

    # 训练模型
    trainer.train(train_loader, val_loader, num_epochs, learning_rate)

    return model, sde


# 移除generate_samples函数，因为我们不再使用基于条件向量的生成方式
# 改为只使用基于范例的生成方式


def generate_from_exemplar(
    model: CSDIModel,
    sde: SimpleSDE,
    exemplar: torch.Tensor,
    num_inference_steps: int = 100,
    device: str = "cpu",
    behavior_id: torch.Tensor or int = None,
) -> torch.Tensor:
    """基于范例生成新样本，使用动态锚点策略

    Args:
        model: 训练好的模型
        sde: SDE对象
        exemplar: [B, 4, 100] 范例样本，基于该样本生成新样本
        num_inference_steps: 推理步数，步数越多生成效果越好但速度越慢
        device: 采样设备
        behavior_id: [B] 或 int 行为ID，可选，用于指导生成过程

    Returns:
        [B, 4, 100] 生成的样本，保留了原始样本的形状特征
    """
    sampler = CSDISampler(model, sde, device=device)

    # 检查输入参数
    if exemplar is None:
        raise ValueError("必须提供exemplar样本")

    # 打印调试信息
    print(
        f"[Debug] generate_from_exemplar - exemplar mean: {exemplar.mean().item():.4f}, exemplar shape: {exemplar.shape}, behavior_id: {behavior_id}"
    )

    # 使用基于范例的生成方法
    if behavior_id is not None:
        # 使用带有行为ID的采样方法
        return sampler.sample_from_exemplar_with_id(
            exemplar, behavior_id, num_inference_steps
        )
    else:
        # 使用普通采样方法
        return sampler.sample_from_exemplar(exemplar, num_inference_steps)


def parse_args():
    """解析命令行参数

    Args:
        无

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="CSDI模型训练与测试")
    # 配置文件和输出目录
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/csdi", help="输出目录"
    )

    # 训练参数
    parser.add_argument("--num-epochs", type=int, default=1000, help="训练轮数")
    parser.add_argument("--min-epochs", type=int, default=500, help="早停最小轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="学习率")

    # 模型参数
    parser.add_argument("--hidden-dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num-layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--num-heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")

    # SDE参数
    parser.add_argument("--beta-min", type=float, default=0.1, help="最小噪声强度")
    parser.add_argument("--beta-max", type=float, default=15.0, help="最大噪声强度")

    # 设备和推理参数
    # 数据参数
    parser.add_argument('--data-file', type=str, default='output/datasets/train.jsonl', help='训练数据文件路径')
    
    # 设备和推理参数
    # 支持CUDA、MPS和CPU设备
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="训练设备",
    )
    parser.add_argument("--num-inference-steps", type=int, default=100, help="推理步数")

    return parser.parse_args()


def main():
    """主函数

    处理命令行参数，执行完整的训练和生成流程：
    1. 加载配置文件
    2. 加载和处理数据
    3. 划分训练集和验证集
    4. 比较数据分布
    5. 训练模型
    6. 生成样本
    7. 反归一化样本
    8. 保存生成的样本
    9. 打印统计特征对比
    10. 运行可视化脚本
    """
    args = parse_args()

    # 打印配置信息
    print(f"配置信息: {args}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 加载配置文件
        print("1. 加载配置文件...")
        config = {}
        if args.config and os.path.exists(args.config):
            try:
                with open(args.config, "r") as f:
                    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
            except Exception as e:
                print(f"警告：加载配置文件失败，将使用默认配置: {e}")
                config = {}

        # 合并命令行参数到配置中
        config.update(vars(args))

        # 2. 加载和处理数据
        print("2. 加载和处理数据...")
        data_file = args.data_file
        if not os.path.exists(data_file):
            print(f"错误：数据文件 {data_file} 不存在")
            print("提示：请先运行 step1_preprocess.py 脚本生成预处理数据")
            print("或使用 --data-file 参数指定正确的数据文件路径")
            return
        
        windows = load_windows_from_jsonl(data_file)
        print(f"   加载了 {len(windows)} 个窗口数据")

        # 3. 划分训练集和验证集
        print("3. 划分训练集和验证集...")
        # 使用80%作为训练集，20%作为验证集
        split_idx = int(len(windows) * 0.8)
        train_windows = windows[:split_idx]
        val_windows = windows[split_idx:]
        print(f"   训练集大小: {len(train_windows)}, 验证集大小: {len(val_windows)}")

        # 4. 比较数据分布
        print("4. 比较训练集和验证集数据分布...")
        compare_data_distributions(train_windows, val_windows)

        # 5. 训练模型
        print("5. 训练CSDI模型...")
        model, sde = train_csdi(train_windows, val_windows, config)
        print("   模型训练完成")

        # 6. 生成样本
        print("6. 生成样本...")
        # 按行为ID分组，每个行为ID挑选一个参考样本
        # 优先从验证集挑选，如果验证集中的行为ID不完整，则从训练集补充
        behavior_id_groups = {}
        
        # 1. 先从验证集中挑选样本
        for window_meta in val_windows:
            behavior_id = int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
            if behavior_id not in behavior_id_groups:
                behavior_id_groups[behavior_id] = []
            behavior_id_groups[behavior_id].append(window_meta)
        
        # 2. 如果验证集中的行为ID不完整，从训练集中补充
        if len(behavior_id_groups) < 8:  # 假设总共有8个行为ID
            missing_ids = set(range(8)) - set(behavior_id_groups.keys())
            print(f"   验证集中缺少行为ID: {missing_ids}，从训练集中补充")
            
            for window_meta in train_windows:
                behavior_id = int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
                if behavior_id in missing_ids and behavior_id not in behavior_id_groups:
                    behavior_id_groups[behavior_id] = [window_meta]
                    missing_ids.remove(behavior_id)
                    if not missing_ids:
                        break
        
        # 3. 如果还是缺少行为ID，从所有数据中随机挑选
        if len(behavior_id_groups) < 8:
            missing_ids = set(range(8)) - set(behavior_id_groups.keys())
            print(f"   仍缺少行为ID: {missing_ids}，从所有数据中随机挑选")
            
            all_windows = train_windows + val_windows
            random.shuffle(all_windows)
            
            for window_meta in all_windows:
                behavior_id = int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
                if behavior_id in missing_ids and behavior_id not in behavior_id_groups:
                    behavior_id_groups[behavior_id] = [window_meta]
                    missing_ids.remove(behavior_id)
                    if not missing_ids:
                        break
        
        # 4. 每个行为ID挑选一个参考样本
        selected_windows = []
        for behavior_id, windows in behavior_id_groups.items():
            # 挑选第一个样本
            selected_windows.append(windows[0])
        
        # 5. 如果没有足够的参考样本，使用默认值
        if not selected_windows:
            print("   警告：未找到参考样本，使用默认值")
            # 从训练集中随机挑选3个样本
            selected_windows = random.sample(train_windows, min(3, len(train_windows)))
        
        num_samples = len(selected_windows)
        print(f"   为 {len(behavior_id_groups)} 个行为ID挑选了 {num_samples} 个参考样本")
        
        # 准备范例样本
        exemplars = []
        for window_meta in selected_windows:
            window = window_meta["window"]
            # 提取窗口数据
            if isinstance(window, pd.DataFrame):
                window_values = window[["delay_up_rs", "delay_down_rs", "loss_up", "loss_dn"]].values.T
            elif isinstance(window, dict):
                nested_window = window.get("window", [])
                window_values = np.zeros((4, 100))
                for i, row in enumerate(nested_window[:100]):
                    window_values[0, i] = row.get("delay_up_rs", 0.0)
                    window_values[1, i] = row.get("delay_down_rs", 0.0)
                    window_values[2, i] = row.get("loss_up", 0.0)
                    window_values[3, i] = row.get("loss_dn", 0.0)
            else:  # list
                window_values = np.zeros((4, 100))
                for i, row in enumerate(window[:100]):
                    window_values[0, i] = row.get("delay_up_rs", 0.0)
                    window_values[1, i] = row.get("delay_down_rs", 0.0)
                    window_values[2, i] = row.get("loss_up", 0.0)
                    window_values[3, i] = row.get("loss_dn", 0.0)
            exemplars.append(window_values)
        
        # 转换为张量
        exemplars_tensor = torch.tensor(np.stack(exemplars), dtype=torch.float32)

        # 生成样本
        generated_samples = generate_from_exemplar(
            model,
            sde,
            exemplar=exemplars_tensor,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        print(f"   生成了 {generated_samples.shape[0]} 个样本")

        # 7. 反归一化样本
        print("7. 反归一化样本...")
        # 加载归一化器，从output/assets/rs_dict.pkl文件加载
        print("   尝试加载归一化器...")
        rs_dict = None
        
        # 尝试从多个位置加载rs_dict.pkl文件
        possible_paths = [
            "output/assets",
            "assets",
            "data"
        ]
        
        for path in possible_paths:
            rs_dict_path = os.path.join(path, "rs_dict.pkl")
            if os.path.exists(rs_dict_path):
                rs_dict = joblib.load(rs_dict_path)
                print(f"   成功从 {path} 目录加载rs_dict")
                break
        
        # 如果没有找到归一化器，抛出异常退出
        if rs_dict is None:
            print(f"错误：无法从以下位置找到rs_dict.pkl文件：")
            for path in possible_paths:
                rs_dict_path = os.path.join(path, "rs_dict.pkl")
                print(f"   - {os.path.abspath(rs_dict_path)}")
            print("请确保已经运行了预处理脚本生成归一化器。")
            print("预处理脚本使用方法：uv run python scripts/step1_preprocess.py")
            sys.exit(1)
        
        print(f"   成功加载了 {len(rs_dict)} 个归一化器")
        
        generated_samples_np = generated_samples.cpu().numpy()
        generated_samples_denorm = np.copy(generated_samples_np)
        
        # 打印调试信息
        print(f"   加载了 {len(rs_dict)} 个归一化器")
        print(f"   生成样本数量：{generated_samples_np.shape[0]}")
        print(f"   参考样本数量：{len(selected_windows)}")
        
        # 准备参考样本反归一化后的结果
        ref_samples_denorm = []
        
        # 反归一化每个生成样本
        B, K, L = generated_samples_np.shape
        for i in range(B):
            # 获取当前生成样本对应的参考样本的行为ID
            if i < len(selected_windows):
                cond = selected_windows[i]["cond"]
                behavior_id = int(cond[CondIndex.BEHAVIOR_ID])
            else:
                # 如果生成样本数量超过参考样本，使用默认行为ID
                behavior_id = 0
            
            # 处理INVALID行为
            if behavior_id >= 8:
                behavior_id = 0
            
            # 获取对应行为ID的归一化器
            if f"up_{behavior_id}" in rs_dict and f"down_{behavior_id}" in rs_dict:
                current_rs_up = rs_dict[f"up_{behavior_id}"]
                current_rs_down = rs_dict[f"down_{behavior_id}"]
                
                # 上行时延反归一化
                up_delay_norm = generated_samples_np[i, 0, :].reshape(-1, 1)
                up_delay_denorm = current_rs_up.inverse_transform(up_delay_norm).flatten()
                generated_samples_denorm[i, 0, :] = up_delay_denorm
                
                # 下行时延反归一化
                down_delay_norm = generated_samples_np[i, 1, :].reshape(-1, 1)
                down_delay_denorm = current_rs_down.inverse_transform(down_delay_norm).flatten()
                generated_samples_denorm[i, 1, :] = down_delay_denorm
        
        # 对参考样本进行反归一化
        for i, window_meta in enumerate(selected_windows):
            cond = window_meta["cond"]
            behavior_id = int(cond[CondIndex.BEHAVIOR_ID])
            
            # 处理INVALID行为
            if behavior_id >= 8:
                behavior_id = 0
            
            # 获取对应行为ID的归一化器
            if f"up_{behavior_id}" in rs_dict and f"down_{behavior_id}" in rs_dict:
                current_rs_up = rs_dict[f"up_{behavior_id}"]
                current_rs_down = rs_dict[f"down_{behavior_id}"]
                
                # 对参考样本进行反归一化
                window = window_meta["window"]
                if isinstance(window, pd.DataFrame):
                    # 如果是DataFrame格式
                    ref_up_delay = window["delay_up_rs"].values.reshape(-1, 1)
                    ref_down_delay = window["delay_down_rs"].values.reshape(-1, 1)
                elif isinstance(window, list):
                    # 如果是列表格式
                    ref_up_delay = np.array([row.get("delay_up_rs", 0.0) for row in window[:100]]).reshape(-1, 1)
                    ref_down_delay = np.array([row.get("delay_down_rs", 0.0) for row in window[:100]]).reshape(-1, 1)
                else:  # 处理dict格式
                    nested_window = window.get("window", [])
                    ref_up_delay = np.array([row.get("delay_up_rs", 0.0) for row in nested_window[:100]]).reshape(-1, 1)
                    ref_down_delay = np.array([row.get("delay_down_rs", 0.0) for row in nested_window[:100]]).reshape(-1, 1)
                
                # 参考样本反归一化
                ref_up_denorm = current_rs_up.inverse_transform(ref_up_delay).flatten()
                ref_down_denorm = current_rs_down.inverse_transform(ref_down_delay).flatten()
                
                ref_samples_denorm.append({
                    "up": ref_up_denorm,
                    "down": ref_down_denorm,
                    "behavior_id": behavior_id
                })
            else:
                # 如果没有对应的归一化器，添加默认值
                ref_samples_denorm.append({
                    "up": np.zeros(100),
                    "down": np.zeros(100),
                    "behavior_id": behavior_id
                })
        
        # 打印反归一化结果
        print(f"   反归一化后样本范围：{generated_samples_denorm.min():.4f} ~ {generated_samples_denorm.max():.4f}")
        print(f"   上行时延范围：{generated_samples_denorm[:, 0, :].min():.4f} ms ~ {generated_samples_denorm[:, 0, :].max():.4f} ms")
        print(f"   下行时延范围：{generated_samples_denorm[:, 1, :].min():.4f} ms ~ {generated_samples_denorm[:, 1, :].max():.4f} ms")

        # 8. 保存生成的样本
        print("8. 保存生成的样本...")
        samples_path = output_dir / "generated_samples.npy"
        np.save(samples_path, generated_samples_denorm)
        print(f"   生成的样本已保存到：{samples_path}")

        # 保存参考样本信息，使用JSONL格式
        reference_info_path = output_dir / "reference_samples.jsonl"
        with open(reference_info_path, "w") as f:
            for window_meta in selected_windows:
                # 确保包含必要字段
                ref_data = {
                    "window": window_meta["window"],
                    "cond": window_meta["cond"],
                    "keep": True,
                    "behavior_label": int(window_meta["cond"][CondIndex.BEHAVIOR_ID])
                }
                f.write(json.dumps(ref_data, ensure_ascii=False) + "\n")
        print(f"   参考样本信息已保存到：{reference_info_path}")

        # 9. 打印统计特征对比
        print("9. 生成样本统计特征与参考样本对比...")
        for i in range(generated_samples_denorm.shape[0]):
            print(f"\n=== 样本 {i + 1} ===")
            ref_window = selected_windows[i]
            ref_cond = ref_window["cond"]
            behavior_id = int(ref_cond[CondIndex.BEHAVIOR_ID])

            print(f"  参考样本行为ID: {behavior_id}")
            print(
                f"  参考样本条件向量 - up_p10: {ref_cond[CondIndex.UP_P10]:.4f}, up_p99: {ref_cond[CondIndex.UP_P99]:.4f}, up_p50: {ref_cond[CondIndex.UP_P50]:.4f}"
            )
            print(
                f"  参考样本条件向量 - dn_p10: {ref_cond[CondIndex.DN_P10]:.4f}, dn_p99: {ref_cond[CondIndex.DN_P99]:.4f}, dn_p50: {ref_cond[CondIndex.DN_P50]:.4f}"
            )

            # 获取参考样本反归一化后的结果
            ref_denorm = ref_samples_denorm[i]
            ref_up = ref_denorm["up"]
            ref_down = ref_denorm["down"]
            
            # 计算参考样本反归一化后的统计特征
            ref_up_p10 = np.percentile(ref_up, 10)
            ref_up_p50 = np.percentile(ref_up, 50)
            ref_up_p99 = np.percentile(ref_up, 99)
            
            ref_down_p10 = np.percentile(ref_down, 10)
            ref_down_p50 = np.percentile(ref_down, 50)
            ref_down_p99 = np.percentile(ref_down, 99)

            # 计算生成样本的统计特征
            gen_up = generated_samples_denorm[i, 0, :]
            gen_down = generated_samples_denorm[i, 1, :]

            gen_up_p10 = np.percentile(gen_up, 10)
            gen_up_p50 = np.percentile(gen_up, 50)
            gen_up_p99 = np.percentile(gen_up, 99)

            gen_down_p10 = np.percentile(gen_down, 10)
            gen_down_p50 = np.percentile(gen_down, 50)
            gen_down_p99 = np.percentile(gen_down, 99)

            # 打印参考样本反归一化后的统计特征
            print("\n  参考样本反归一化后统计特征：")
            print(
                f"  上行时延 - p10: {ref_up_p10:.4f}, p50: {ref_up_p50:.4f}, p99: {ref_up_p99:.4f}"
            )
            print(
                f"  下行时延 - p10: {ref_down_p10:.4f}, p50: {ref_down_p50:.4f}, p99: {ref_down_p99:.4f}"
            )

            # 打印生成样本统计特征
            print("\n  生成样本统计特征：")
            print(
                f"  上行时延 - p10: {gen_up_p10:.4f}, p50: {gen_up_p50:.4f}, p99: {gen_up_p99:.4f}"
            )
            print(
                f"  下行时延 - p10: {gen_down_p10:.4f}, p50: {gen_down_p50:.4f}, p99: {gen_down_p99:.4f}"
            )
            
            # 打印统计特征差异
            print("\n  统计特征差异：")
            print(
                f"  上行时延 - p10差异: {gen_up_p10 - ref_up_p10:.4f}, p50差异: {gen_up_p50 - ref_up_p50:.4f}, p99差异: {gen_up_p99 - ref_up_p99:.4f}"
            )
            print(
                f"  下行时延 - p10差异: {gen_down_p10 - ref_down_p10:.4f}, p50差异: {gen_down_p50 - ref_down_p50:.4f}, p99差异: {gen_down_p99 - ref_down_p99:.4f}"
            )

        # 10. 运行可视化脚本
        print("10. 运行可视化脚本...")
        visualization_dir = output_dir / "visualization"
        visualization_dir.mkdir(exist_ok=True)

        # 运行可视化脚本，传递正确的参考样本文件路径
        import subprocess

        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/step3_visualize.py",
                "--reference-file",
                str(reference_info_path),
                "--generated-file",
                str(samples_path),
            ],
            check=True,
        )
        print(f"   可视化图片已生成到: {visualization_dir}")

        print("\nCSDI脚本执行完成")
        print(f"所有输出已保存到: {output_dir}")

    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
