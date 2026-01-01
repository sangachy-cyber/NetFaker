#!/usr/bin/env python3

"""整合版条件扩散模型训练与评估
包含模型定义、训练、采样、后处理和可视化的完整流程
"""

# 添加当前目录到Python路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import joblib
import warnings
from pathlib import Path



import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet1DModel
from torch import nn

# 兼容不同版本的diffusers
try:
    from diffusers.models.unet_1d import UNet1DOutput
except ImportError:
    try:
        from diffusers.models.unets.unet_1d import UNet1DOutput
    except ImportError:
        from dataclasses import dataclass
        from diffusers.utils import BaseOutput

        @dataclass
        class UNet1DOutput(BaseOutput):
            sample: torch.FloatTensor

warnings.filterwarnings("ignore")


# 模型定义部分
class PaddedConditionalUNet1D(nn.Module):
    """带条件的1D UNet，兼容旧版 diffusers（无需 time_embedding_dim 或 timestep_cond）。
    条件通过additive conditioning（相加）注入，实现CSDI的核心思想，改善幅度控制。
    """

    def __init__(
        self,
        target_length=100,
        padded_length=104,  # 修正为104，符合文档要求
        state_embedding_dim=8,
        num_network_states=10,  # 根据实际状态ID范围设置（如0~9 → 10）
        block_out_channels=(32, 64),  # 可调整，但需与 down/up blocks 匹配
    ):
        super().__init__()
        self.target_length = target_length
        self.padded_length = padded_length

        if padded_length < target_length:
            raise ValueError("padded_length 必须 >= target_length")

        # 定义离散网络状态嵌入层
        self.state_embed = nn.Embedding(
            num_embeddings=num_network_states,
            embedding_dim=state_embedding_dim,
        )

        # 条件维度：22（连续）+ state_embedding_dim（离散）
        cont_dim = 22
        self.cond_dim = cont_dim + state_embedding_dim

        # ⚠️ 旧版 diffusers 不支持 time_embedding_dim！
        unet_config = {
            "sample_size": self.padded_length,  # 确保使用正确的padded_length
            "in_channels": 4,                  # 仅数据通道，条件通过additive conditioning注入
            "out_channels": 4,                  # 输出通道数与输入数据通道数一致
            "down_block_types": ("DownBlock1D", "DownBlock1D"),
            "up_block_types": ("UpBlock1D", "UpBlock1D"),
            "block_out_channels": block_out_channels,  # 时间嵌入维度 = block_out_channels[0]
            "layers_per_block": 1,
            "mid_block_type": "UNetMidBlock1D",
            # ❌ 移除 'time_embedding_dim' —— 旧版不支持！
        }
        self.unet = UNet1DModel(**unet_config)
        
        # 添加条件投影层，实现CSDI的核心思想：additive conditioning
        # 投影条件向量到U-Net的输出通道维度（4）
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 输出4个通道，与U-Net输出匹配
        )

    def forward(self, sample: torch.Tensor, timestep=None, times=None, cond: torch.Tensor = None):
        # 兼容 Rectified Flow 的 times 参数
        if timestep is None and times is not None:
            timestep = times
        B, C, L = sample.shape
        if self.target_length != L:
            raise ValueError(f"输入长度必须为 {self.target_length}，得到了 {L}")

        # Padding
        if self.padded_length > L:
            pad_len = self.padded_length - L
            padded_sample = torch.cat([
                sample,
                torch.zeros(B, C, pad_len, device=sample.device, dtype=sample.dtype),
            ], dim=-1)
        else:
            padded_sample = sample

        # 调用 UNet 主体，仅处理数据通道
        unet_output = self.unet(padded_sample, timestep, return_dict=True)
        h = unet_output.sample
        
        # 实现 CSDI 的核心思想：additive conditioning
        if cond is not None:
            # 分割条件
            global_part = torch.cat([cond[:, :11], cond[:, 12:13]], dim=-1)  # (B, 12)
            # 忽略局部特征，使用零向量替代
            local_part = torch.zeros_like(cond[:, 13:23])  # (B, 10)，使用零向量
            cont = torch.cat([global_part, local_part], dim=-1)  # (B, 22)

            # 处理离散状态
            state_id_raw = cond[:, 11]
            state_id = state_id_raw.long().clamp(0, self.state_embed.num_embeddings - 1)
            if not torch.allclose(state_id_raw, state_id.float(), atol=1e-5):
                warnings.warn(f"非整数 network_state_id: {state_id_raw[:3]}")
            state_emb = self.state_embed(state_id)  # (B, E)

            # 合并条件向量
            full_cond = torch.cat([cont, state_emb], dim=-1)  # (B, 22+E)
            
            # 条件投影并广播到序列长度
            cond_proj = self.cond_proj(full_cond).unsqueeze(-1)  # [B, C, 1]
            h = h + cond_proj  # 广播到 [B, C, T]

        # 裁剪到目标长度
        output = h[:, :4, :self.target_length]
        return output


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
    
    def train_step(self, batch: dict) -> torch.Tensor:
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
    
    def validation_step(self, batch: dict) -> torch.Tensor:
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
    
    def sample(self, cond: torch.Tensor, num_sample_steps: int = 50, guidance_scale: float = 1.0) -> torch.Tensor:
        """从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_sample_steps: 推理步数，DPM-Solver-2推荐使用50步或更多
            guidance_scale: 条件增强比例
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        # 条件增强：放大p95分位数
        if guidance_scale != 1.0:
            cond_aug = cond.clone()
            # 仅对非稳定状态（行为ID不为0）进行条件增强
            # 对于稳定状态（行为ID为0），不进行条件增强，保持原始条件
            behavior_ids = cond[:, 11].long()
            for i in range(cond_aug.shape[0]):
                if behavior_ids[i] != 0:
                    cond_aug[i, 2] *= guidance_scale  # p95_del_up
                    cond_aug[i, 6] *= guidance_scale  # p95_del_dn
            cond = cond_aug
            
        # 确保条件向量在正确的设备上
        cond = cond.to(self.device)
        
        # 初始化噪声样本
        batch_size = cond.shape[0]
        x = torch.randn(batch_size, 4, 100, device=self.device)
        
        # DPM-Solver-2 采样方法实现
        # 时间步调度：从0到1线性分布
        times = torch.linspace(0.0, 1.0, num_sample_steps + 1, device=self.device)
        
        # DPM-Solver-2 使用二阶Runge-Kutta方法，结合最优步长
        for i in range(num_sample_steps):
            t = times[i]
            t_next = times[i+1]
            dt = t_next - t
            
            # 第一步：计算k1
            with torch.no_grad():
                k1 = self.forward(x, timestep=torch.full((batch_size,), t, device=self.device), cond=cond)
            
            # 第二步：计算k2
            x_mid = x + k1 * dt * 0.5
            with torch.no_grad():
                k2 = self.forward(x_mid, timestep=torch.full((batch_size,), t + dt * 0.5, device=self.device), cond=cond)
            
            # 第三步：计算k3
            x_mid = x + k2 * dt * 0.5
            with torch.no_grad():
                k3 = self.forward(x_mid, timestep=torch.full((batch_size,), t + dt * 0.5, device=self.device), cond=cond)
            
            # 第四步：计算k4
            x_next_pred = x + k3 * dt
            with torch.no_grad():
                k4 = self.forward(x_next_pred, timestep=torch.full((batch_size,), t_next, device=self.device), cond=cond)
            
            # 经典RK4更新
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        
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


class PostProcessor:
    """扩散模型后处理器.

    负责将生成的归一化样本转换为实际的网络轨迹数据.
    """

    def __init__(self, qt_up, qt_dn, threshold: float = 0.5) -> None:
        """初始化后处理器.

        Args:
            qt_up: 上行延迟的QuantileTransformer
            qt_dn: 下行延迟的QuantileTransformer
            threshold: 二值化阈值

        """
        self.qt_up = qt_up
        self.qt_dn = qt_dn
        self.threshold = threshold

    def postprocess(self, sample: torch.Tensor, base_time: float = 0.0, cond: torch.Tensor = None) -> np.ndarray:
        """后处理生成的样本.

        Args:
            sample: 生成的样本 (B, 4, 100)
            base_time: 基础时间戳
            cond: 条件向量 (B, 23)，当前未使用

        Returns:
            处理后的轨迹数据 (B, 100, 5)

        """
        # 转换为numpy数组
        sample_np = sample.detach().cpu().numpy()  # (B, 4, 100)

        batch_size = sample_np.shape[0]
        seq_len = sample_np.shape[2]  # 动态获取序列长度
        result = np.zeros((batch_size, seq_len, 5))

        for i in range(batch_size):
            # 提取各通道数据
            del_up_norm = sample_np[i, 0, :]      # (100,)
            del_dn_norm = sample_np[i, 1, :]      # (100,)
            loss_up_norm = sample_np[i, 2, :]     # (100,)
            loss_dn_norm = sample_np[i, 3, :]     # (100,)

            # 反变换延迟数据
            del_up_ms = self.qt_up.inverse_transform(del_up_norm.reshape(-1, 1)).flatten()
            del_dn_ms = self.qt_dn.inverse_transform(del_dn_norm.reshape(-1, 1)).flatten()

            # 二值化丢包数据，仅根据模型预测结果生成，不添加任何人工丢包
            loss_up_bin = (loss_up_norm > self.threshold).astype(float)
            loss_dn_bin = (loss_dn_norm > self.threshold).astype(float)

            # 生成时间戳
            timestamps = base_time + np.arange(seq_len) * 0.1

            # 组合结果
            result[i, :, 0] = timestamps
            result[i, :, 1] = del_up_ms
            result[i, :, 2] = del_dn_ms
            result[i, :, 3] = loss_up_bin
            result[i, :, 4] = loss_dn_bin

        return result


# 数据集和数据加载器
class NetworkTraceDataset(Dataset):
    """网络轨迹数据集"""

    def __init__(self, file_path):
        self.windows = []
        self.conditions = []

        # 读取数据
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("keep", True):
                    # 提取特征，忽略时间戳列
                    # 数据格式: [timestamp, delay_up, delay_down, loss_up, loss_dn]
                    features = []
                    for row in data["window"]:
                        features.append([
                            row["delay_up"],
                            row["delay_down"],
                            row["loss_up"],
                            row["loss_dn"],
                        ])

                    # 确保窗口数据长度为100（模型期望的长度）
                    if len(features) > 100:
                        features = features[:100]
                    elif len(features) < 100:
                        # 用最后一行填充
                        while len(features) < 100:
                            features.append(features[-1])

                    # 转换为张量
                    features = torch.FloatTensor(features)  # (100, 4)
                    cond = torch.FloatTensor(data["cond"])     # (22,) 或 (23,)

                    # 调整维度以匹配模型输入: (4, 100)
                    features = features.transpose(0, 1)

                    self.windows.append(features)
                    self.conditions.append(cond)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            "window": self.windows[idx],
            "cond": self.conditions[idx],
        }


def collate_fn(batch):
    """批处理函数"""
    windows = torch.stack([item["window"] for item in batch])
    conds = torch.stack([item["cond"] for item in batch])

    return {
        "window": windows,
        "cond": conds,
    }


# 训练函数
def train_model(train_file, val_file=None, epochs=10, batch_size=100, learning_rate=1e-4, output_dir="./output/visualization", backbone_type="tsdiff"):
    """训练模型"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建数据集
    train_dataset = NetworkTraceDataset(train_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 创建验证数据集（如果提供了验证文件）
    val_loader = None
    if val_file and os.path.exists(val_file):
        val_dataset = NetworkTraceDataset(val_file)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 创建 Rectified Flow 模型，使用指定的骨干网络
    model = ConditionalFlowModel(device=device, backbone_type=backbone_type)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 早停参数
    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # 记录训练损失
    train_losses = []
    val_losses = []

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # 执行训练步骤
            loss = model.train_step(batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            train_losses.append(loss.item())

            # 每100个批次打印一次损失
            if num_batches % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # 验证阶段
        if val_loader is not None:
            val_total_loss = 0
            val_num_batches = 0
            for val_batch in val_loader:
                val_loss = model.validation_step(val_batch)
                val_total_loss += val_loss.item()
                val_num_batches += 1
                val_losses.append(val_loss.item())

            val_avg_loss = val_total_loss / val_num_batches
            print(f"Epoch {epoch+1}/{epochs} Validation Loss: {val_avg_loss:.4f}")

            # 检查早停条件
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存模型
    model_save_path = output_path / "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)

    # 保存训练损失
    loss_save_path = output_path / "train_losses.npy"
    np.save(loss_save_path, np.array(train_losses))

    return model


# 采样和后处理函数
def generate_valid_cond_vector(train_file, num_network_states=8, num_consecutive_windows=60):
    """生成有效的条件向量
    
    Args:
        train_file: 训练数据文件路径，必须存在且包含真实条件向量
        num_network_states: 网络状态ID的数量
        num_consecutive_windows: 连续窗口数量，默认60个窗口（10分钟）
        
    Returns:
        从训练数据中提取的真实条件向量及其对应的窗口数据，以及连续的条件向量列表
        
    Raises:
        FileNotFoundError: 如果训练文件不存在
        ValueError: 如果训练文件中没有真实条件向量
        KeyError: 如果训练文件中的数据没有'cond'字段
    """
    # 检查训练文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    
    # 从训练数据中提取所有真实条件向量和对应的窗口数据
    all_real_data = []
    with open(train_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("keep", True):
                # 直接从数据中获取条件向量和对应的窗口数据
                if "cond" in data and "window" in data:
                    all_real_data.append((data["cond"], data["window"]))
                else:
                    # 如果没有cond字段或window字段，抛出KeyError
                    raise KeyError("'cond'")
    
    # 只打印一次获取信息
    print(f"条件向量: 从训练数据中获取初始条件向量，随机选择连续 {num_consecutive_windows} 个窗口")
    
    # 检查是否提取到真实条件向量
    if not all_real_data:
        raise ValueError(f"训练文件中没有真实条件向量: {train_file}")
    
    # 如果条件向量数量不足，使用所有可用的条件向量
    if len(all_real_data) < num_consecutive_windows:
        print(f"警告：训练文件中条件向量数量不足 {num_consecutive_windows} 个，仅使用 {len(all_real_data)} 个")
        start_idx = 0
        # 使用所有可用的条件向量
        consecutive_data = all_real_data
    else:
        # 随机选择起始位置，确保有足够的连续窗口
        max_start_idx = len(all_real_data) - num_consecutive_windows
        start_idx = np.random.randint(0, max_start_idx + 1)
        # 获取连续的窗口数据
        consecutive_data = all_real_data[start_idx:start_idx + num_consecutive_windows]
    
    # 提取第一个窗口的条件向量作为初始条件向量
    # 保留真实数据中的网络状态ID，不做修改
    real_cond, real_window = consecutive_data[0]
    cond = torch.FloatTensor(real_cond)
    
    # 提取所有连续的条件向量
    consecutive_conds = [torch.FloatTensor(data[0]) for data in consecutive_data]
    
    return cond, real_window, consecutive_conds


def get_initial_cond_vector(test_file, train_file=None):
    """获取初始条件向量
    
    Args:
        test_file: 测试数据文件路径，优先使用测试集获取参考样本
        train_file: 训练数据文件路径，仅在测试集不可用时作为备选
        
    Returns:
        tuple: (cond_vector, window_data, consecutive_conds)
            cond_vector: 初始条件向量，shape (23,)
            window_data: 参考窗口数据，如果没有参考样本则为None
            consecutive_conds: 连续的条件向量列表
    """
    # 优先使用测试集获取参考样本
    if test_file and os.path.exists(test_file):
        cond, window, consecutive_conds = generate_valid_cond_vector(test_file)
        return cond, window, consecutive_conds
    # 如果测试集不可用，尝试使用训练集作为备选
    elif train_file and os.path.exists(train_file):
        cond, window, consecutive_conds = generate_valid_cond_vector(train_file)
        return cond, window, consecutive_conds
    else:
        # 如果没有参考样本，创建一个默认的条件向量
        # 注意：这里使用torch.zeros(23)作为默认值，实际应用中可以从保存的条件向量中随机选择
        cond = torch.zeros(23)
        cond[11] = 0.0  # network_state_id
        return cond, None, [cond]


def sample_and_postprocess(model, assets_dir, num_samples=10, num_windows_per_sample=1, output_dir="./output/visualization", device="cpu", test_file=None, train_file=None, guidance_scale=1.0):
    """采样并后处理生成的数据
    
    Args:
        model: 训练好的模型
        assets_dir: 资产文件目录
        num_samples: 生成的样本数量
        num_windows_per_sample: 每个样本包含的窗口数量
        output_dir: 输出目录
        device: 设备
        test_file: 测试数据文件路径，用于获取参考样本
        train_file: 训练数据文件路径，仅在测试集不可用时作为备选
        guidance_scale: 条件增强比例
        
    Returns:
        tuple: (生成的样本数据, 样本的行为ID列表, 参考窗口数据列表)
    """
    # 使用 Rectified Flow 采样器
    sampler = FlowSampler(model, device)

    qt_up = joblib.load(Path(assets_dir) / "qt_up.pkl")
    qt_dn = joblib.load(Path(assets_dir) / "qt_down.pkl")

    postprocessor = PostProcessor(qt_up, qt_dn)

    # 生成样本
    print("Generating samples...")
    all_generated_samples = []
    all_conds = []  # 收集所有使用的条件向量，用于计算超限比例
    sample_behavior_ids = []  # 收集每个生成样本对应的原始行为ID
    reference_windows = []  # 收集参考窗口数据
    
    # 读取测试数据中的所有窗口作为参考样本
    test_windows = []
    if test_file and os.path.exists(test_file):
        with open(test_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("keep", True) and "window" in data and "cond" in data:
                    test_windows.append((data["window"], data["cond"]))
    
    print(f"\n使用测试集中的 {len(test_windows)} 个窗口作为参考样本")
    
    # 按行为ID对测试窗口进行分组
    behavior_groups = {}
    for ref_window, ref_cond in test_windows:
        behavior_id = int(ref_cond[11])
        if behavior_id not in behavior_groups:
            behavior_groups[behavior_id] = []
        behavior_groups[behavior_id].append((ref_window, ref_cond))
    
    print(f"\n测试数据按行为ID分组：{[(bid, len(windows)) for bid, windows in behavior_groups.items()]}")
    
    # 从每个行为ID组中随机选择样本，确保覆盖不同的行为ID
    selected_windows = []
    
    # 首先从每个非空行为ID组中至少选择一个样本
    for behavior_id, windows in behavior_groups.items():
        if windows:
            selected_windows.append(windows[0])
    
    # 如果还需要更多样本，从所有测试窗口中随机选择
    if len(selected_windows) < num_samples:
        remaining_needed = num_samples - len(selected_windows)
        # 从所有测试窗口中随机选择剩余样本
        import random
        random_windows = random.sample(test_windows, remaining_needed)
        selected_windows.extend(random_windows)
    
    # 确保不超过指定的样本数量
    selected_windows = selected_windows[:num_samples]
    
    print(f"\n从 {len(selected_windows)} 个窗口中生成样本，包含行为ID: {[int(cond[11]) for _, cond in selected_windows]}")
    
    # 遍历选中的窗口生成样本
    for window_idx, (ref_window, ref_cond) in enumerate(selected_windows):
        print(f"\n处理参考窗口 {window_idx+1}/{len(selected_windows)}...")
        
        # 提取原始行为ID（条件向量的第12个元素，索引11）
        behavior_id = int(ref_cond[11])
        
        # 打印参考样本的条件向量信息
        print(f"  参考样本行为ID: {behavior_id}")
        
        # 反归一化条件向量中的延迟相关特征
        up_p95_norm = ref_cond[2]
        down_p95_norm = ref_cond[6]
        
        # 使用QuantileTransformer反归一化延迟值
        up_p95 = qt_up.inverse_transform(np.array([[up_p95_norm]]))[0][0]
        down_p95 = qt_dn.inverse_transform(np.array([[down_p95_norm]]))[0][0]
        
        print(f"  参考样本条件向量 - up_p95: {up_p95:.2f}ms, down_p95: {down_p95:.2f}ms")
        print(f"  参考样本条件向量 - up_p50: {ref_cond[1]:.4f}, down_p50: {ref_cond[5]:.4f}（归一化值）")
        print(f"  参考样本条件向量 - up_std: {ref_cond[3]:.4f}, down_std: {ref_cond[7]:.4f}（归一化值）")
        
        # 将参考窗口的条件向量转换为张量
        cond = torch.FloatTensor(ref_cond)
        cond_tensor = cond.unsqueeze(0).to(device)
        
        # 打印生成样本使用条件向量信息
        # 反归一化生成样本使用的条件向量值
        gen_up_p95 = qt_up.inverse_transform(np.array([[cond[2].item()]]))[0][0]
        gen_down_p95 = qt_dn.inverse_transform(np.array([[cond[6].item()]]))[0][0]
        
        print(f"  生成样本使用条件向量 - up_p95: {gen_up_p95:.2f}ms, down_p95: {gen_down_p95:.2f}ms")
        print(f"  生成样本使用条件向量 - up_p50: {cond[1]:.4f}, down_p50: {cond[5]:.4f}（归一化值）")
        print(f"  生成样本使用条件向量 - up_std: {cond[3]:.4f}, down_std: {cond[7]:.4f}（归一化值）")
        print(f"  生成样本使用条件向量 - up_p95: {cond[2]:.4f}, down_p95: {cond[6]:.4f}（归一化值）")
        
        # 生成样本
        generated_sample = sampler.sample(cond_tensor, num_inference_steps=100, guidance_scale=guidance_scale)  # 使用100步推理，确保轨迹完整
        processed_sample = postprocessor.postprocess(
            generated_sample, base_time=0.0,
            cond=cond_tensor  # 将条件向量传入，用于soft clamp
        )
        
        # 计算生成样本的统计特征
        # processed_sample 形状: (1, 100, 5)
        # 特征顺序: 时间戳, 上行延迟, 下行延迟, 上行丢包率, 下行丢包率
        
        # 提取生成的上行延迟和下行延迟数据
        gen_del_up = processed_sample[0, :, 1]  # (100,)，上行延迟（反归一化后，ms）
        gen_del_dn = processed_sample[0, :, 2]  # (100,)，下行延迟（反归一化后，ms）
        
        # 计算统计特征
        gen_up_p50 = np.percentile(gen_del_up, 50)
        gen_up_p95 = np.percentile(gen_del_up, 95)
        gen_up_std = np.std(gen_del_up)
        gen_up_min = np.min(gen_del_up)
        gen_up_max = np.max(gen_del_up)
        
        gen_down_p50 = np.percentile(gen_del_dn, 50)
        gen_down_p95 = np.percentile(gen_del_dn, 95)
        gen_down_std = np.std(gen_del_dn)
        gen_down_min = np.min(gen_del_dn)
        gen_down_max = np.max(gen_del_dn)
        
        # 打印生成样本统计特征
        print(f"  生成样本统计特征 - up_p50: {gen_up_p50:.2f}ms, up_p95: {gen_up_p95:.2f}ms, up_std: {gen_up_std:.2f}ms")
        print(f"  生成样本统计特征 - down_p50: {gen_down_p50:.2f}ms, down_p95: {gen_down_p95:.2f}ms, down_std: {gen_down_std:.2f}ms")
        print(f"  生成样本统计特征 - up_min: {gen_up_min:.2f}ms, up_max: {gen_up_max:.2f}ms")
        print(f"  生成样本统计特征 - down_min: {gen_down_min:.2f}ms, down_max: {gen_down_max:.2f}ms")
        
        # 添加到结果列表
        all_generated_samples.append(processed_sample)
        all_conds.append(cond.cpu().numpy())  # 保存条件向量
        sample_behavior_ids.append(behavior_id)  # 保存行为ID
        reference_windows.append(ref_window)  # 保存参考窗口
    
    # 合并所有生成的样本
    final_samples = np.concatenate(all_generated_samples, axis=0)
    print(f"\n生成的样本总形状: {final_samples.shape}")
    print(f"生成的样本对应的行为ID: {sample_behavior_ids}")

    # 计算超限比例
    print("\n计算超限比例...")
    
    # 收集所有使用的条件向量，用于计算超限比例
    # 从训练数据文件中获取真实条件向量和窗口数据，用于计算超限比例
    real_conds = []
    real_windows = []
    if train_file and os.path.exists(train_file):
        with open(train_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("keep", True) and "window" in data and "cond" in data:
                    real_windows.append(data["window"])
                    real_conds.append(data["cond"])
    
    # 从测试数据文件中获取真实条件向量和窗口数据，用于计算超限比例
    if test_file and os.path.exists(test_file):
        with open(test_file) as f:
            for line in f:
                data = json.loads(line)
                if data.get("keep", True) and "window" in data and "cond" in data:
                    real_windows.append(data["window"])
                    real_conds.append(data["cond"])
    
    real_conds = np.array(real_conds)  # (B, 23)
    
    # 注意：final_samples的形状是(num_samples, num_windows_per_sample*window_size, 5)
    # 我们需要将其转换为(B, 100, 5)的形状，其中B=num_samples*num_windows_per_sample
    final_samples_reshaped = final_samples.reshape(-1, 100, 5)  # (B, 100, 5)
    gen_up = final_samples_reshaped[:, :, 1]  # (B, 100)，提取上行延迟（反归一化后，ms）
    
    # 确保real_conds的数量大于等于gen_up的数量
    if len(real_conds) < len(gen_up):
        # 如果真实条件向量数量不足，重复使用
        num_repeats = (len(gen_up) + len(real_conds) - 1) // len(real_conds)
        real_conds = np.tile(real_conds, (num_repeats, 1))[:len(gen_up)]
    else:
        real_conds = real_conds[:len(gen_up)]
    
    # 计算真实数据的反归一化p95值
    # 直接使用条件向量中的p95值，这是预处理时已经计算好的准确值
    # 从条件向量中获取归一化的p95值
    p95_norm = real_conds[:, 2]
    # 反归一化
    true_p95_raw = qt_up.inverse_transform(p95_norm.reshape(-1, 1)).flatten()
    
    # 确保真实p95值数量足够
    if len(true_p95_raw) < len(gen_up):
        # 如果真实p95值数量不足，重复使用
        num_repeats = (len(gen_up) + len(true_p95_raw) - 1) // len(true_p95_raw)
        true_p95_raw = np.tile(true_p95_raw, num_repeats)[:len(gen_up)]
    else:
        true_p95_raw = true_p95_raw[:len(gen_up)]
    
    # 计算超限比例：生成的上行延迟 > 真实p95值 * 1.1
    over_limit_ratio = np.mean(gen_up > true_p95_raw[:, None] * 1.1)
    
    print(f"生成数据（反归一化后）的平均上行延迟：{np.mean(gen_up):.2f} ms")
    print(f"真实数据（反归一化后）的平均p95值：{np.mean(true_p95_raw):.2f} ms")
    print(f"允许的上限：真实p95值 * 1.1 = {np.mean(true_p95_raw * 1.1):.2f} ms")
    print(f"超限比例: {over_limit_ratio:.2%}")
    
    if over_limit_ratio > 0.05:
        print("💡 警告：超限比例 > 5%，说明模型确实'学不会控制上限'，需要干预。")
    else:
        print("✅ 超限比例 <= 5%，模型能够较好地控制上限。")

    # 保存生成的样本
    output_path = Path(output_dir)
    samples_path = output_path / "generated_samples.npy"
    np.save(samples_path, final_samples)
    print(f"Generated samples saved as {samples_path}")
    
    # 保存行为ID和参考窗口数据，用于后续可视化
    np.save(output_path / "sample_behavior_ids.npy", np.array(sample_behavior_ids))
    
    # 保存参考窗口数据
    with open(output_path / "reference_windows.json", "w") as f:
        json.dump(reference_windows, f, indent=2)

    return final_samples, sample_behavior_ids, reference_windows


# 可视化函数
def visualize_training_losses(losses_file, output_dir="./output/visualization"):
    """可视化训练损失曲线"""
    # 加载训练损失
    train_losses = np.load(losses_file)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)

    # 保存图像
    output_path = Path(output_dir)
    loss_curve_path = output_path / "training_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training loss curve saved as {loss_curve_path}")


def visualize_generated_samples(generated_samples_file, real_data_file, output_dir="./output/visualization"):
    """可视化生成的样本与真实数据的对比"""
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 加载生成的样本
    generated_samples = np.load(generated_samples_file)

    # 加载真实数据样本
    real_samples = []
    with open(real_data_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data.get("keep", True):
                window_data = data["window"]
                # 提取特征 - 加载所有时间步
                features = []
                for row in window_data:  # 加载所有时间步
                    features.append([
                        row["delay_up"],
                        row["delay_down"],
                        row["loss_up"],
                        row["loss_dn"],
                    ])
                real_samples.append(features)

    real_samples = np.array(real_samples)

    # 创建输出目录
    output_path = Path(output_dir)

    # 计算丢包率统计信息
    real_loss_up = real_samples[:, :, 2].flatten()
    real_loss_dn = real_samples[:, :, 3].flatten()
    generated_loss_up = generated_samples[:, :, 3].flatten()
    generated_loss_dn = generated_samples[:, :, 4].flatten()
    
    # 统计信息
    loss_metrics = {
        "real_loss_up": {
            "mean": np.mean(real_loss_up),
            "std": np.std(real_loss_up),
            "non_zero_ratio": np.mean(real_loss_up > 0),
            "count": len(real_loss_up)
        },
        "real_loss_dn": {
            "mean": np.mean(real_loss_dn),
            "std": np.std(real_loss_dn),
            "non_zero_ratio": np.mean(real_loss_dn > 0),
            "count": len(real_loss_dn)
        },
        "generated_loss_up": {
            "mean": np.mean(generated_loss_up),
            "std": np.std(generated_loss_up),
            "non_zero_ratio": np.mean(generated_loss_up > 0),
            "count": len(generated_loss_up)
        },
        "generated_loss_dn": {
            "mean": np.mean(generated_loss_dn),
            "std": np.std(generated_loss_dn),
            "non_zero_ratio": np.mean(generated_loss_dn > 0),
            "count": len(generated_loss_dn)
        }
    }
    
    # 打印丢包率分布
    print("\n丢包率分布详情：")
    
    # 真实数据上行丢包率分布
    real_up_unique, real_up_counts = np.unique(real_loss_up, return_counts=True)
    print("真实数据上行丢包率：")
    for value, count in zip(real_up_unique, real_up_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(real_loss_up)*100:.2f}%)")
    
    # 真实数据下行丢包率分布
    real_dn_unique, real_dn_counts = np.unique(real_loss_dn, return_counts=True)
    print("真实数据下行丢包率：")
    for value, count in zip(real_dn_unique, real_dn_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(real_loss_dn)*100:.2f}%)")
    
    # 生成数据上行丢包率分布
    gen_up_unique, gen_up_counts = np.unique(generated_loss_up, return_counts=True)
    print("生成数据上行丢包率：")
    for value, count in zip(gen_up_unique, gen_up_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(generated_loss_up)*100:.2f}%)")
    
    # 生成数据下行丢包率分布
    gen_dn_unique, gen_dn_counts = np.unique(generated_loss_dn, return_counts=True)
    print("生成数据下行丢包率：")
    for value, count in zip(gen_dn_unique, gen_dn_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(generated_loss_dn)*100:.2f}%)")
    
    # 打印统计信息
    print("\n丢包率统计信息：")
    for name, metrics in loss_metrics.items():
        print(f"{name}: 均值={metrics['mean']:.6f}, 标准差={metrics['std']:.6f}, 非零值比例={metrics['non_zero_ratio']:.6f}, 样本数={metrics['count']}")
    
    # 绘制延迟数据的直方图对比
    plt.figure(figsize=(15, 12))

    # 上行延迟
    plt.subplot(3, 2, 1)
    real_del_up = real_samples[:, :, 0].flatten()
    generated_del_up = generated_samples[:, :, 1].flatten()  # 注意索引对应关系
    plt.hist(real_del_up, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_up, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("上行延迟分布")
    plt.xlabel("延迟 (归一化)")
    plt.ylabel("密度")
    plt.legend()

    # 下行延迟
    plt.subplot(3, 2, 2)
    real_del_dn = real_samples[:, :, 1].flatten()
    generated_del_dn = generated_samples[:, :, 2].flatten()
    plt.hist(real_del_dn, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_dn, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("下行延迟分布")
    plt.xlabel("延迟 (归一化)")
    plt.ylabel("密度")
    plt.legend()

    # 上行丢包
    plt.subplot(3, 2, 3)
    plt.hist(real_loss_up, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="真实数据", density=True, align="left")
    plt.hist(generated_loss_up, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="生成数据", density=True, align="left")
    plt.xticks([0, 1], ["0", "1"])
    plt.title("上行丢包率分布")
    plt.xlabel("丢包率")
    plt.ylabel("密度")
    plt.legend()
    
    # 添加上行丢包率统计信息
    plt.text(0.02, 0.95, 
             f"真实数据: 均值={loss_metrics['real_loss_up']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['real_loss_up']['non_zero_ratio']:.6f}\n" +
             f"生成数据: 均值={loss_metrics['generated_loss_up']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['generated_loss_up']['non_zero_ratio']:.6f}",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 下行丢包
    plt.subplot(3, 2, 4)
    plt.hist(real_loss_dn, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="真实数据", density=True, align="left")
    plt.hist(generated_loss_dn, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="生成数据", density=True, align="left")
    plt.xticks([0, 1], ["0", "1"])
    plt.title("下行丢包率分布")
    plt.xlabel("丢包率")
    plt.ylabel("密度")
    plt.legend()
    
    # 添加下行丢包率统计信息和说明
    plt.text(0.02, 0.95, 
             f"真实数据: 均值={loss_metrics['real_loss_dn']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['real_loss_dn']['non_zero_ratio']:.6f}\n" +
             f"生成数据: 均值={loss_metrics['generated_loss_dn']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['generated_loss_dn']['non_zero_ratio']:.6f}",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 丢包率说明
    plt.text(0.02, 0.5,
             "说明：\n" +
             "1. 原始训练数据中丢包率大部分为0\n" +
             "2. 预处理后，丢包率简化为0和1两个值\n" +
             "3. 模型从训练数据中学习生成丢包率",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Q-Q图：上行延迟
    plt.subplot(3, 2, 5)
    stats.probplot(generated_del_up, dist="norm", plot=plt)
    plt.title("Q-Q图：上行延迟")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")

    # Q-Q图：下行延迟
    plt.subplot(3, 2, 6)
    stats.probplot(generated_del_dn, dist="norm", plot=plt)
    plt.title("Q-Q图：下行延迟")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")

    plt.tight_layout()

    # 保存图像
    comparison_path = output_path / "data_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data comparison visualization saved as {comparison_path}")
    
    # 额外保存上行丢包率时间序列图（显示所有时间步）
    if generated_samples.shape[0] > 0:
        plt.figure(figsize=(20, 10))
        for sample_idx in range(min(generated_samples.shape[0], 2)):  # 显示前2个样本
            generated_loss_time = generated_samples[sample_idx, :, 3]
            time_steps = np.arange(len(generated_loss_time))
            
            plt.subplot(min(generated_samples.shape[0], 2), 1, sample_idx+1)
            plt.plot(time_steps, generated_loss_time, label=f"样本 {sample_idx+1} 上行丢包率", linewidth=1)
            plt.title(f"生成数据上行丢包率时间序列 - 样本 {sample_idx+1} (共 {len(time_steps)} 步)")
            plt.xlabel("时间步")
            plt.ylabel("丢包率")
            plt.ylim(-0.1, 1.1)  # 确保0和1都能显示
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 标记非零值位置
            non_zero_indices = np.where(generated_loss_time > 0)[0]
            if len(non_zero_indices) > 0:
                plt.scatter(non_zero_indices, generated_loss_time[non_zero_indices], 
                          color='red', s=50, alpha=0.7, label="非零丢包率")
                plt.legend()
                print(f"样本 {sample_idx+1} 非零丢包率位置: {non_zero_indices}")
                
                # 在非零值位置添加注释
                for idx in non_zero_indices:
                    plt.annotate(f"{idx}", xy=(idx, generated_loss_time[idx]), 
                               xytext=(idx, generated_loss_time[idx]+0.1),
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        time_series_path = output_path / "uplink_loss_time_series.png"
        plt.savefig(time_series_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Uplink loss time series saved as {time_series_path}")
    
    # 保存所有样本的上行丢包率数据到文本文件，方便分析
    with open(output_path / "uplink_loss_data.txt", "w") as f:
        for sample_idx in range(generated_samples.shape[0]):
            f.write(f"样本 {sample_idx+1} 上行丢包率数据:\n")
            loss_data = generated_samples[sample_idx, :, 3]
            for step, loss in enumerate(loss_data):
                f.write(f"{step}: {loss}\n")
            f.write("\n")
        print(f"Uplink loss data saved as {output_path / 'uplink_loss_data.txt'}")


def analyze_test_set_stats(test_file):
    """分析测试集数据统计信息
    
    Args:
        test_file: 测试数据文件路径
    """
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    total_samples = 0
    valid_samples = 0
    network_states = {}
    non_zero_loss_count = 0
    total_loss_count = 0
    window_lengths = []
    
    with open(test_file) as f:
        for i, line in enumerate(f):
            total_samples += 1
            data = json.loads(line)
            
            if data.get("keep", True):
                valid_samples += 1
                
                # 统计网络状态
                if "cond" in data and len(data["cond"]) > 11:
                    # 网络状态ID在条件向量的第11位
                    network_state = int(data["cond"][11])
                    network_states[network_state] = network_states.get(network_state, 0) + 1
                
                # 统计窗口数据
                if "window" in data:
                    window_data = data["window"]
                    window_lengths.append(len(window_data))
                    
                    # 统计丢包率
                    for row in window_data:
                        total_loss_count += 1
                        if row["loss_up"] > 0 or row["loss_dn"] > 0:
                            non_zero_loss_count += 1
    
    # 计算统计结果
    avg_window_length = np.mean(window_lengths) if window_lengths else 0
    std_window_length = np.std(window_lengths) if window_lengths else 0
    non_zero_loss_ratio = non_zero_loss_count / total_loss_count if total_loss_count > 0 else 0
    
    # 打印统计信息
    print(f"测试集总样本数: {total_samples}")
    print(f"有效样本数 (keep=True): {valid_samples}")
    print(f"无效样本数 (keep=False): {total_samples - valid_samples}")
    print(f"有效样本比例: {valid_samples / total_samples:.2%}")
    
    print("\n网络状态分布:")
    for state, count in sorted(network_states.items()):
        print(f"  状态 {state}: {count} 样本 ({count / valid_samples:.2%})")
    
    print("\n窗口长度统计:")
    print(f"  平均窗口长度: {avg_window_length:.2f}")
    print(f"  窗口长度标准差: {std_window_length:.2f}")
    print(f"  最小窗口长度: {min(window_lengths) if window_lengths else 0}")
    print(f"  最大窗口长度: {max(window_lengths) if window_lengths else 0}")
    
    print("\n丢包率统计:")
    print(f"  总丢包数据点: {total_loss_count}")
    print(f"  非零丢包数据点: {non_zero_loss_count}")
    print(f"  非零丢包比例: {non_zero_loss_ratio:.2%}")
    
    print("\n条件向量结构:")
    print("  条件向量维度: 23")
    print("  - 全局特征: 13个 (0-12)")
    print("  - 网络状态ID: 1个 (11)")
    print("  - 局部特征: 10个 (13-22)")


# 主函数
def main():
    parser = argparse.ArgumentParser(description="Integrated training and evaluation pipeline for conditional diffusion model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--num-windows-per-sample", type=int, default=10, help="Number of windows per sample for long sequence generation")
    parser.add_argument("--output-dir", type=str, default="./output/visualization", help="Output directory")
    parser.add_argument("--guidance-scale", type=float, default=1.0, 
                        help="条件增强比例，固定为1.0以保持宽波动，避免峰值过高")
    parser.add_argument("--backbone-type", type=str, default="tsdiff", 
                        choices=["tsdiff", "unet"],
                        help="Backbone network type: 'tsdiff' for TSDiff (MLP) or 'unet' for UNet1D")

    args = parser.parse_args()

    # 获取数据文件路径
    train_file = os.path.join(args.data_dir, "datasets", "train.jsonl")
    val_file = os.path.join(args.data_dir, "datasets", "val.jsonl")
    test_file = os.path.join(args.data_dir, "datasets", "test.jsonl")
    assets_dir = os.path.join(args.data_dir, "assets")

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 训练模型
    print("Starting model training...")
    model = train_model(
        train_file=train_file,
        val_file=val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        backbone_type=args.backbone_type,
    )

    # 可视化训练损失
    print("Visualizing training losses...")
    losses_file = os.path.join(args.output_dir, "train_losses.npy")
    visualize_training_losses(losses_file, args.output_dir)

    # 采样和后处理
    print("Sampling and post-processing...")
    sample_and_postprocess(
        model, assets_dir,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_windows_per_sample=args.num_windows_per_sample,
        output_dir=args.output_dir,
        device=device,
        test_file=test_file,  # 优先使用测试文件路径，用于生成真实的条件向量
        train_file=train_file,  # 仅作为备选
    )

    # 可视化生成的样本
    print("Visualizing generated samples...")
    generated_samples_file = os.path.join(args.output_dir, "generated_samples.npy")
    visualize_generated_samples(
        generated_samples_file, test_file,  # 使用测试文件进行可视化对比
        output_dir=args.output_dir,
    )
    
    # 按行为ID可视化生成的样本
    print("\n按行为ID可视化生成的样本...")
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 加载保存的行为ID和参考窗口数据
    behavior_ids_path = os.path.join(args.output_dir, "sample_behavior_ids.npy")
    ref_windows_path = os.path.join(args.output_dir, "reference_windows.json")
    
    if os.path.exists(behavior_ids_path) and os.path.exists(ref_windows_path):
        behavior_ids = np.load(behavior_ids_path).tolist()
        with open(ref_windows_path, "r") as f:
            ref_windows = json.load(f)
        
        # 创建按行为ID分组的可视化
        output_path = Path(args.output_dir)
        behavior_vis_dir = output_path / "behavior_based_visualization"
        behavior_vis_dir.mkdir(exist_ok=True)
        
        # 行为ID到名称的映射
        behavior_names = {
            0: "稳定状态",
            1: "偶尔波动",
            2: "频繁波动",
            3: "突发高峰",
            4: "持续高延迟",
            5: "频繁丢包",
            6: "突发丢包",
            7: "高丢包率",
            8: "无效状态"
        }
        
        # 按行为ID分组样本
        behavior_groups = {}
        for i, behavior_id in enumerate(behavior_ids):
            if behavior_id not in behavior_groups:
                behavior_groups[behavior_id] = []
            behavior_groups[behavior_id].append(i)
        
        print(f"行为ID分布: {[(bid, len(ids)) for bid, ids in behavior_groups.items()]}")
        
        # 加载生成的样本
        generated_samples = np.load(generated_samples_file)
        
        # 对每个行为ID组进行可视化
        for behavior_id, sample_indices in behavior_groups.items():
            print(f"\n可视化行为ID {behavior_id} ({behavior_names.get(behavior_id, f'未知行为 {behavior_id}')})，样本数: {len(sample_indices)}")
            
            # 为每个样本绘制对比图
            for i, sample_idx in enumerate(sample_indices[:5]):  # 每个行为最多显示5个样本
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                # 使用样本原始的行为ID，而不是分组的行为ID
                original_behavior_id = behavior_ids[sample_idx]
                fig.suptitle(f"行为ID: {original_behavior_id} ({behavior_names.get(original_behavior_id, f'未知行为 {original_behavior_id}')}) - 样本 {i+1}/{len(sample_indices)}", fontsize=16)
                
                # 获取参考窗口和生成样本
                gen_sample = generated_samples[sample_idx]
                ref_window = ref_windows[sample_idx]
                
                # 提取参考窗口的特征 - 使用原始延迟值（毫秒）和原始丢包率
                ref_del_up = [row["delay_up_origin"] for row in ref_window]
                ref_del_dn = [row["delay_down_origin"] for row in ref_window]
                ref_loss_up = [row["loss_up_origin"] for row in ref_window]
                ref_loss_dn = [row["loss_down_origin"] for row in ref_window]
                
                # 提取生成样本的特征
                gen_del_up = gen_sample[:, 1]  # 上行延迟
                gen_del_dn = gen_sample[:, 2]  # 下行延迟
                gen_loss_up = gen_sample[:, 3]  # 上行丢包率
                gen_loss_dn = gen_sample[:, 4]  # 下行丢包率
                
                # 确保数据长度一致
                min_len = min(len(ref_del_up), len(gen_del_up))
                ref_del_up = ref_del_up[:min_len]
                ref_del_dn = ref_del_dn[:min_len]
                ref_loss_up = ref_loss_up[:min_len]
                ref_loss_dn = ref_loss_dn[:min_len]
                gen_del_up = gen_del_up[:min_len]
                gen_del_dn = gen_del_dn[:min_len]
                gen_loss_up = gen_loss_up[:min_len]
                gen_loss_dn = gen_loss_dn[:min_len]
                
                time_steps = np.arange(min_len)
                
                # 上行延迟对比
                axes[0, 0].plot(time_steps, ref_del_up, label="参考窗口", color="blue")
                axes[0, 0].plot(time_steps, gen_del_up, label="生成样本", color="red", alpha=0.7)
                axes[0, 0].set_title("上行延迟对比")
                axes[0, 0].set_xlabel("时间步")
                axes[0, 0].set_ylabel("延迟 (ms)")
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                # 下行延迟对比
                axes[0, 1].plot(time_steps, ref_del_dn, label="参考窗口", color="blue")
                axes[0, 1].plot(time_steps, gen_del_dn, label="生成样本", color="red", alpha=0.7)
                axes[0, 1].set_title("下行延迟对比")
                axes[0, 1].set_xlabel("时间步")
                axes[0, 1].set_ylabel("延迟 (ms)")
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                
                # 上行丢包率对比
                axes[1, 0].plot(time_steps, ref_loss_up, label="参考窗口", color="blue")
                axes[1, 0].plot(time_steps, gen_loss_up, label="生成样本", color="red", alpha=0.7)
                axes[1, 0].set_title("上行丢包率对比")
                axes[1, 0].set_xlabel("时间步")
                axes[1, 0].set_ylabel("丢包率")
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # 下行丢包率对比
                axes[1, 1].plot(time_steps, ref_loss_dn, label="参考窗口", color="blue")
                axes[1, 1].plot(time_steps, gen_loss_dn, label="生成样本", color="red", alpha=0.7)
                axes[1, 1].set_title("下行丢包率对比")
                axes[1, 1].set_xlabel("时间步")
                axes[1, 1].set_ylabel("丢包率")
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                
                plt.tight_layout()
                plt.savefig(behavior_vis_dir / f"behavior_{behavior_id}_sample_{i+1}.png", dpi=300, bbox_inches="tight")
                plt.close()
        
        # 绘制所有行为的统计对比
        plt.figure(figsize=(12, 8))
        
        # 统计每个行为的样本数量
        behavior_counts = [len(ids) for bid, ids in sorted(behavior_groups.items())]
        behavior_labels = [behavior_names.get(bid, f"{bid}") for bid, ids in sorted(behavior_groups.items())]
        
        plt.bar(range(len(behavior_counts)), behavior_counts, tick_label=behavior_labels, color='skyblue')
        plt.title("生成样本行为ID分布")
        plt.xlabel("行为类别")
        plt.ylabel("样本数量")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "behavior_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"\n行为基于的可视化结果已保存到: {behavior_vis_dir}")
    
    # 分析测试集数据统计信息
    print("\n=== 测试集数据统计信息 ===")
    analyze_test_set_stats(test_file)
    
    print("Full pipeline completed!")



if __name__ == "__main__":
    main()