#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
条件扩散模型实现
基于预处理后的网络轨迹数据进行训练
严格按照条件扩散模型方案.md实现
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import warnings

# 尝试导入diffusers，如果失败则使用占位符
try:
    from diffusers import UNet1DModel
    DIFFUSERS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # 如果无法导入diffusers，使用占位符
    print(f"Warning: diffusers not available ({e}), using placeholder")
    DIFFUSERS_AVAILABLE = False
    
    class UNet1DModel:
        def __init__(self, *args, **kwargs):
            pass


class ConditionalUNet1D(UNet1DModel):
    """
    UNet1D with conditioning on:
      - 22 continuous features (Z-scored)
      - 1 discrete network state ID (embedded)
    Injects condition via `timestep_cond` using `addition_embed_type="text"`.
    """

    def __init__(
        self,
        num_network_states: int = 8,
        state_embed_dim: int = 32,
        time_embedding_dim: int = 256,
        **kwargs
    ):
        if DIFFUSERS_AVAILABLE:
            # Enable timestep_cond usage
            kwargs["addition_embed_type"] = "text"
            kwargs["addition_time_embed_dim"] = time_embedding_dim
            super().__init__(**kwargs)

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
        else:
            # 如果diffusers不可用，创建一个简单的替代实现
            super().__init__()
            print("Using simplified implementation due to missing diffusers")

    def forward(self, sample: torch.Tensor, timestep, cond: torch.Tensor = None):
        if DIFFUSERS_AVAILABLE:
            if cond is not None:
                # Split cond: skip index 11 (state_id)
                global_part = torch.cat([cond[:, :11], cond[:, 12:13]], dim=-1)  # (B, 12)
                local_part = cond[:, 13:23]                                      # (B, 10)
                cont = torch.cat([global_part, local_part], dim=-1)             # (B, 22)

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

            return super().forward(sample, timestep, timestep_cond=timestep_cond)
        else:
            # 简单的占位符实现
            class DummyOutput:
                def __init__(self, sample):
                    self.sample = sample
            return DummyOutput(sample)


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
    print(f"Diffusers可用: {DIFFUSERS_AVAILABLE}")
    print('模型结构:')
    print(type(model).__name__)