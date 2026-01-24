"""burst时序评估指标。

计算真实序列与合成序列中burst事件的时序分布差异，
用于评估合成数据中burst事件的节奏匹配程度。
"""

from typing import Dict, List

import numpy as np


class BurstTimingMetrics:
    """burst时序评估指标计算器。
    
    计算真实序列与合成序列中burst事件的时序分布差异，
    用于评估合成数据中burst事件的节奏匹配程度。
    """

    def calculate(self, real_loss: np.ndarray, syn_loss: np.ndarray, loss_threshold: float = 5.0) -> Dict:
        """计算burst时序指标。
        
        Args:
            real_loss: 真实丢包率序列
            syn_loss: 合成丢包率序列
            loss_threshold: 丢包率阈值，超过此值视为burst事件，默认5.0%
            
        Returns:
            Dict: 包含burst时序指标的字典
        """
        # 检测burst事件
        real_bursts = self._detect_bursts(real_loss, loss_threshold)
        syn_bursts = self._detect_bursts(syn_loss, loss_threshold)
        
        # 计算burst起始时间间隔的均方误差
        inter_burst_interval_mae = self._calculate_inter_burst_interval_mae(real_bursts, syn_bursts)
        
        return {
            "inter_burst_interval_mae": inter_burst_interval_mae
        }
    
    def _detect_bursts(self, loss: np.ndarray, loss_threshold: float) -> List[int]:
        """检测burst事件。
        
        Args:
            loss: 丢包率序列
            loss_threshold: 丢包率阈值，超过此值视为burst事件
            
        Returns:
            List[int]: burst事件的起始时间点列表
        """
        bursts = []
        in_burst = False
        
        for i, loss_val in enumerate(loss):
            if loss_val > loss_threshold:
                if not in_burst:
                    bursts.append(i)
                    in_burst = True
            else:
                in_burst = False
        
        return bursts
    
    def _calculate_inter_burst_interval_mae(self, real_bursts: List[int], syn_bursts: List[int]) -> float:
        """计算burst起始时间间隔的均方误差。
        
        Args:
            real_bursts: 真实burst起始时间列表
            syn_bursts: 合成burst起始时间列表
            
        Returns:
            float: burst起始时间间隔的均方误差
        """
        if len(real_bursts) < 2 or len(syn_bursts) < 2:
            return 0.0
        
        # 计算真实序列中burst间隔
        real_intervals = []
        for i in range(1, len(real_bursts)):
            real_intervals.append(real_bursts[i] - real_bursts[i - 1])
        
        # 计算合成序列中burst间隔
        syn_intervals = []
        for i in range(1, len(syn_bursts)):
            syn_intervals.append(syn_bursts[i] - syn_bursts[i - 1])
        
        # 调整长度，取最小值
        min_len = min(len(real_intervals), len(syn_intervals))
        real_intervals = real_intervals[:min_len]
        syn_intervals = syn_intervals[:min_len]
        
        # 计算均方误差
        mae = np.mean(np.abs(np.array(real_intervals) - np.array(syn_intervals)))
        
        return mae
