"""联合一致性指标计算模块。

负责计算单窗口内合成数据与真实数据的联合一致性差异，
包括在burst期间的平均延迟偏差。
"""

from typing import Dict

import numpy as np

from netfaker.evaluator.metrics.burst import BurstMetrics


class JointMetrics:
    """联合一致性指标计算器。
    
    实现真实与合成数据在联合一致性上的差异评估，
    包括在burst期间的平均延迟偏差。
    """

    def __init__(self):
        """初始化联合指标计算器。
        
        使用BurstMetrics来检测burst事件。
        """
        self.burst_metrics = BurstMetrics()

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray,
                 real_loss: np.ndarray, syn_loss: np.ndarray) -> Dict:
        """计算联合一致性指标。
        
        Args:
            real_delay: 真实延迟数据 (ms)
            syn_delay: 合成延迟数据 (ms)
            real_loss: 真实丢包率数据 (%)
            syn_loss: 合成丢包率数据 (%)
            
        Returns:
            Dict: 包含联合一致性指标的字典
        """
        # 检测真实数据中的burst
        real_bursts = self.burst_metrics._detect_bursts(real_loss)

        if not real_bursts:
            # 如果没有burst，偏差为0
            cond_delay_bias = 0.0
        else:
            # 收集所有burst期间的延迟偏差
            all_bias = []

            for burst in real_bursts:
                # 获取burst期间的索引范围
                start = burst['start']
                end = start + burst['length']

                # 获取burst期间的真实和合成延迟
                real_burst_delay = real_delay[start:end]
                syn_burst_delay = syn_delay[start:end]

                # 计算偏差
                bias = np.abs(syn_burst_delay - real_burst_delay)
                all_bias.extend(bias.tolist())

            # 计算平均偏差
            cond_delay_bias = np.mean(all_bias)

        return {
            'cond_delay_bias': cond_delay_bias
        }
