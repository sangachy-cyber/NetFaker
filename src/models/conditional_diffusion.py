#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
条件扩散模型实现
整合了模型定义、训练器、采样器、后处理器等组件
严格按照《条件扩散模型方案.md》实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from typing import Dict, Any, Optional
from diffusers import UNet1DModel
from diffusers.schedulers import DDPMScheduler

# 兼容不同版本的diffusers
try:
    from diffusers.models.unet_1d import UNet1DOutput
except ImportError:
    try:
        from diffusers.models.unets.unet_1d import UNet1DOutput
    except ImportError:
        from diffusers.utils import BaseOutput
        from dataclasses import dataclass
        
        @dataclass
        class UNet1DOutput(BaseOutput):
            sample: torch.FloatTensor




class PaddedConditionalUNet1D(nn.Module):
    """
    带条件的1D UNet，兼容旧版 diffusers（无需 time_embedding_dim 或 timestep_cond）。
    条件通过通道拼接注入，稳定可靠。
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
            embedding_dim=state_embedding_dim
        )

        # 条件维度：22（连续）+ state_embedding_dim（离散）
        cont_dim = 22
        self.cond_dim = cont_dim + state_embedding_dim

        # ⚠️ 旧版 diffusers 不支持 time_embedding_dim！
        unet_config = {
            "sample_size": self.padded_length,  # 确保使用正确的padded_length
            "in_channels": 4 + self.cond_dim,   # 数据(4) + 条件
            "out_channels": 4,                  # 输出通道数与输入数据通道数一致
            "down_block_types": ("DownBlock1D", "DownBlock1D"),
            "up_block_types": ("UpBlock1D", "UpBlock1D"),
            "block_out_channels": block_out_channels,  # 时间嵌入维度 = block_out_channels[0]
            "layers_per_block": 1,
            "mid_block_type": "UNetMidBlock1D",
            # ❌ 移除 'time_embedding_dim' —— 旧版不支持！
        }
        self.unet = UNet1DModel(**unet_config)

    def forward(self, sample: torch.Tensor, timestep, cond: torch.Tensor = None):
        B, C, L = sample.shape
        if L != self.target_length:
            raise ValueError(f"输入长度必须为 {self.target_length}，得到了 {L}")

        # Padding
        if L < self.padded_length:
            pad_len = self.padded_length - L
            padded_sample = torch.cat([
                sample,
                torch.zeros(B, C, pad_len, device=sample.device, dtype=sample.dtype)
            ], dim=-1)
        else:
            padded_sample = sample

        # 注入条件（通道拼接）
        if cond is not None:
            # 分割条件
            global_part = torch.cat([cond[:, :11], cond[:, 12:13]], dim=-1)  # (B, 12)
            local_part = cond[:, 13:23]                                      # (B, 10)
            cont = torch.cat([global_part, local_part], dim=-1)             # (B, 22)

            # 处理离散状态
            state_id_raw = cond[:, 11]
            state_id = state_id_raw.long().clamp(0, self.state_embed.num_embeddings - 1)
            if not torch.allclose(state_id_raw, state_id.float(), atol=1e-5):
                warnings.warn(f"非整数 network_state_id: {state_id_raw[:3]}")
            state_emb = self.state_embed(state_id)  # (B, E)

            # 合并并扩展到序列长度
            full_cond = torch.cat([cont, state_emb], dim=-1)  # (B, 22+E)
            cond_expanded = full_cond.unsqueeze(-1).expand(-1, -1, self.padded_length)

            # 拼接到输入通道
            unet_input = torch.cat([padded_sample, cond_expanded], dim=1)
        else:
            unet_input = padded_sample

        # 调用 UNet（无 timestep_cond，无 time_embedding_dim）
        unet_output = self.unet(unet_input, timestep, return_dict=True)
        output = unet_output.sample[:, :4, :self.target_length]
        return output

def validate_cond(cond: torch.Tensor, num_network_states: int = 8):
    """
    验证条件向量格式
    
    Args:
        cond: 条件向量，形状为(..., 23)
        num_network_states: 网络状态数量，默认8
        
    Raises:
        ValueError: 当条件向量格式不正确时抛出异常
    """
    if cond.dim() != 2 or cond.shape[-1] != 23:
        raise ValueError(f"期望条件向量形状为(B, 23)，得到了{cond.shape}")
    
    state_id = cond[:, 11]
    if not torch.allclose(state_id, state_id.round(), atol=1e-5):
        warnings.warn(f"检测到非整数的network_state_id: {state_id.tolist()}")
    
    if (state_id < 0).any() or (state_id >= num_network_states).any():
        raise ValueError(f"network_state_id超出范围[0, {num_network_states})")


class DiffusionTrainer:
    """
    扩散模型训练器
    负责模型训练和验证
    """
    
    def __init__(self, model, scheduler, device: str = "cpu", patience: int = 5):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            scheduler: 调度器
            device: 训练设备
            patience: 早停耐心值
        """
        self.model = model
        self.scheduler = scheduler
        # 确保调度器配置正确
        if hasattr(self.scheduler, 'config'):
            config = self.scheduler.config
            if getattr(config, 'prediction_type', None) != 'sample':
                warnings.warn(f"调度器prediction_type为{getattr(config, 'prediction_type', None)}，期望为'sample'")
        self.device = device
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.should_stop = False
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        执行一个训练步骤
        
        Args:
            batch: 包含'window'和'cond'的批次数据
            
        Returns:
            损失值
        """
        # 获取数据
        clean_sample = batch['window'].to(self.device)  # (B, 4, 100)
        cond = batch['cond'].to(self.device)            # (B, 23)
        
        # 采样随机时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (clean_sample.shape[0],), 
            device=self.device
        ).long()
        
        # 生成噪声
        noise = torch.randn_like(clean_sample)
        
        # 添加噪声
        noisy_samples = self.scheduler.add_noise(clean_sample, noise, timesteps)
        
        # 模型预测（预测原始数据而不是噪声）
        model_pred = self.model(noisy_samples, timesteps, cond)
        
        # 计算损失 - 目标是原始数据而不是噪声
        loss = F.mse_loss(model_pred, clean_sample)
        
        # 添加调试信息
        if torch.isnan(loss):
            print(f"Clean sample range: {clean_sample.min().item():.4f} ~ {clean_sample.max().item():.4f}")
            print(f"Model pred range: {model_pred.min().item():.4f} ~ {model_pred.max().item():.4f}")
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        执行一个验证步骤
        
        Args:
            batch: 包含'window'和'cond'的批次数据
            
        Returns:
            损失值
        """
        self.model.eval()
        with torch.no_grad():
            # 获取数据
            clean_sample = batch['window'].to(self.device)  # (B, 4, 100)
            cond = batch['cond'].to(self.device)            # (B, 23)
            
            # 采样随机时间步
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps, (clean_sample.shape[0],), 
                device=self.device
            ).long()
            
            # 生成噪声
            noise = torch.randn_like(clean_sample)
            
            # 添加噪声
            noisy_samples = self.scheduler.add_noise(clean_sample, noise, timesteps)
            
            # 模型预测（预测原始数据而不是噪声）
            model_pred = self.model(noisy_samples, timesteps, cond)
            
            # 计算损失 - 目标是原始数据而不是噪声
            loss = F.mse_loss(model_pred, clean_sample)
            
            # 添加调试信息
            if torch.isnan(loss):
                print(f"Val Clean sample range: {clean_sample.min().item():.4f} ~ {clean_sample.max().item():.4f}")
                print(f"Val Model pred range: {model_pred.min().item():.4f} ~ {model_pred.max().item():.4f}")
        
        self.model.train()
        return loss
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 当前验证损失
            
        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            
        return self.should_stop


class DiffusionSampler:
    """
    扩散模型采样器
    负责从训练好的模型中生成新样本
    """
    
    def __init__(self, model, scheduler, device: str = "cpu"):
        """
        初始化采样器
        
        Args:
            model: 训练好的模型
            scheduler: 采样调度器
            device: 采样设备
        """
        self.model = model
        self.scheduler = scheduler
        # 确保调度器配置正确
        if hasattr(self.scheduler, 'config'):
            config = self.scheduler.config
            if getattr(config, 'prediction_type', None) != 'sample':
                warnings.warn(f"调度器prediction_type为{getattr(config, 'prediction_type', None)}，期望为'sample'")
        self.device = device
        
    def sample(self, cond: torch.Tensor, num_inference_steps: int = 1000) -> torch.Tensor:
        """
        从条件向量生成新样本
        
        Args:
            cond: 条件向量 (B, 23)
            num_inference_steps: 推理步数
            
        Returns:
            生成的样本 (B, 4, 100)
        """
        # 确保条件向量在正确的设备上
        cond = cond.to(self.device)
        
        # 初始化噪声样本
        # 使用标准高斯噪声初始化（std=1.0）
        batch_size = cond.shape[0]
        sample = torch.randn(batch_size, 4, 100, device=self.device)
        # 丢包通道保持标准噪声（不缩放）
        
        # 设置调度器推理步数
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 迭代去噪
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # 获取模型预测（预测原始数据而不是噪声）
                model_pred = self.model(sample, t, cond)
                
                # 确保传入 scheduler 的是张量而不是对象
                if hasattr(model_pred, 'sample'):
                    model_pred = model_pred.sample
                
            # 执行去噪步骤
            sample = self.scheduler.step(
                model_pred, 
                t, 
                sample,
                generator=None  # 显式指定
            ).prev_sample
            
            # 调试信息
            if t % 100 == 0:
                print(f"时间步 {t}: 样本范围 [{sample.min().item():.2f}, {sample.max().item():.2f}]")
        
        # 打印最终生成范围
        print(f"最终生成范围: [{sample.min().item():.2f}, {sample.max().item():.2f}]")
        return sample

class PostProcessor:
    """
    扩散模型后处理器
    负责将生成的归一化样本转换为实际的网络轨迹数据
    """
    
    def __init__(self, qt_up, qt_dn, threshold: float = 0.5):
        """
        初始化后处理器
        
        Args:
            qt_up: 上行延迟的QuantileTransformer
            qt_dn: 下行延迟的QuantileTransformer
            threshold: 二值化阈值
        """
        self.qt_up = qt_up
        self.qt_dn = qt_dn
        self.threshold = threshold
        
    def postprocess(self, sample: torch.Tensor, base_time: float = 0.0) -> np.ndarray:
        """
        后处理生成的样本
        
        Args:
            sample: 生成的样本 (B, 4, 100)
            base_time: 基础时间戳
            
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
            
            # 打印归一化值范围
            print(f"生成的上行延迟归一化范围: {del_up_norm.min():.4f}, {del_up_norm.max():.4f}")
            print(f"生成的下行延迟归一化范围: {del_dn_norm.min():.4f}, {del_dn_norm.max():.4f}")
            
            # 反变换延迟数据
            del_up_ms = self.qt_up.inverse_transform(del_up_norm.reshape(-1, 1)).flatten()
            del_dn_ms = self.qt_dn.inverse_transform(del_dn_norm.reshape(-1, 1)).flatten()
            
            # 打印反变换后的范围
            print(f"恢复的延迟(秒)范围: {del_up_ms.min():.4f}, {del_up_ms.max():.4f}")
            
            # 二值化丢包数据
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


# compute_local_stats 函数已移至 src/preprocessing/local_stats.py


