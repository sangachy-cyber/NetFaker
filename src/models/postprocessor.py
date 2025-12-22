#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
扩散模型后处理器
基于条件扩散模型方案.md实现
"""

import torch
import numpy as np


class PostProcessor:
    """
    扩散模型后处理器
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
        result = np.zeros((batch_size, 100, 5))
        
        for i in range(batch_size):
            # 提取各通道数据
            del_up_norm = sample_np[i, 0, :]      # (100,)
            del_dn_norm = sample_np[i, 1, :]      # (100,)
            loss_up_norm = sample_np[i, 2, :]     # (100,)
            loss_dn_norm = sample_np[i, 3, :]     # (100,)
            
            # 反变换延迟数据
            del_up_ms = self.qt_up.inverse_transform(del_up_norm.reshape(-1, 1)).flatten()
            del_dn_ms = self.qt_dn.inverse_transform(del_dn_norm.reshape(-1, 1)).flatten()
            
            # 二值化丢包数据
            loss_up_bin = (loss_up_norm > self.threshold).astype(float)
            loss_dn_bin = (loss_dn_norm > self.threshold).astype(float)
            
            # 生成时间戳
            timestamps = base_time + np.arange(100) * 0.1
            
            # 组合结果
            result[i, :, 0] = timestamps
            result[i, :, 1] = del_up_ms
            result[i, :, 2] = del_dn_ms
            result[i, :, 3] = loss_up_bin
            result[i, :, 4] = loss_dn_bin
            
        return result