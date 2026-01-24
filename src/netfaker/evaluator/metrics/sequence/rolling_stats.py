"""滚动统计稳定性评估指标。

计算滚动统计量的漂移，用于评估合成数据的全局稳定性。
"""

from typing import Dict

import numpy as np


class RollingStatsMetrics:
    """滚动统计稳定性评估指标计算器。
    
    计算真实序列与合成序列在滚动时间窗口内的统计量差异，
    用于评估合成数据的全局稳定性。
    """

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray, window_size: int = 100) -> Dict:
        """计算滚动统计指标。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            window_size: 滚动窗口大小，默认100
            
        Returns:
            Dict: 包含滚动统计指标的字典
        """
        # 计算滚动P99值
        real_rolling_p99 = self._calculate_rolling_percentile(real_delay, window_size, 99)
        syn_rolling_p99 = self._calculate_rolling_percentile(syn_delay, window_size, 99)
        
        # 计算P99漂移
        rolling_p99_drift = np.mean(np.abs(real_rolling_p99 - syn_rolling_p99))
        
        return {
            "rolling_p99_drift": rolling_p99_drift
        }
    
    def _calculate_rolling_percentile(self, data: np.ndarray, window_size: int, percentile: float) -> np.ndarray:
        """计算滚动分位数。
        
        Args:
            data: 输入数据序列
            window_size: 滚动窗口大小
            percentile: 分位数，范围[0, 100]
            
        Returns:
            np.ndarray: 滚动分位数值
        """
        rolling_percentiles = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            p = np.percentile(window, percentile)
            rolling_percentiles.append(p)
        
        return np.array(rolling_percentiles)
