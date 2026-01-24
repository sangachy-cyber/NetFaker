"""丢包突发性指标计算模块。

负责计算单窗口内合成数据与真实数据的丢包突发性差异，
包括burst数量差和平均长度误差。
"""

from typing import Dict, List

import numpy as np


class BurstMetrics:
    """丢包突发性指标计算器。
    
    实现真实与合成数据在丢包突发性上的差异评估，
    包括burst数量差和平均长度误差。
    
    Burst定义：≥2 连续点丢包率 > 5%
    """

    def calculate(self, real_loss: np.ndarray, syn_loss: np.ndarray) -> Dict:
        """计算丢包突发性指标。
        
        Args:
            real_loss: 真实丢包率数据 (%)
            syn_loss: 合成丢包率数据 (%)
            
        Returns:
            Dict: 包含丢包突发性指标的字典
        """
        # 检测burst（连续≥2个点丢包率>5%）
        real_bursts = self._detect_bursts(real_loss)
        syn_bursts = self._detect_bursts(syn_loss)

        # 计算burst数量差
        burst_count_diff = abs(len(real_bursts) - len(syn_bursts))

        # 计算平均长度误差
        if len(real_bursts) == 0 and len(syn_bursts) == 0:
            burst_len_mae = 0.0
        elif len(real_bursts) == 0:
            avg_syn_len = np.mean([burst['length'] for burst in syn_bursts])
            burst_len_mae = avg_syn_len
        elif len(syn_bursts) == 0:
            avg_real_len = np.mean([burst['length'] for burst in real_bursts])
            burst_len_mae = avg_real_len
        else:
            avg_real_len = np.mean([burst['length'] for burst in real_bursts])
            avg_syn_len = np.mean([burst['length'] for burst in syn_bursts])
            burst_len_mae = abs(avg_real_len - avg_syn_len)

        return {
            'burst_count_diff': burst_count_diff,
            'burst_len_mae': burst_len_mae
        }

    def _detect_bursts(self, loss_data: np.ndarray) -> List[Dict]:
        """检测丢包burst。
        
        Args:
            loss_data: 丢包率数据 (%)
            
        Returns:
            List[Dict]: burst列表，每个burst包含start和length
        """
        bursts = []
        in_burst = False
        burst_start = 0

        for i, loss in enumerate(loss_data):
            if loss > 5.0:  # 丢包率>5%视为丢包事件
                if not in_burst:
                    # 开始新的burst
                    in_burst = True
                    burst_start = i
            else:
                if in_burst:
                    # 结束当前burst
                    in_burst = False
                    burst_length = i - burst_start
                    if burst_length >= 2:  # 只保留长度≥2的burst
                        bursts.append({
                            'start': burst_start,
                            'length': burst_length
                        })

        # 处理末尾的burst
        if in_burst:
            burst_length = len(loss_data) - burst_start
            if burst_length >= 2:
                bursts.append({
                    'start': burst_start,
                    'length': burst_length
                })

        return bursts
