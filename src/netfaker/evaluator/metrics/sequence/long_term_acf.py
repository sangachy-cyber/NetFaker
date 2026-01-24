"""长程自相关评估指标。

计算长程自相关函数的均方误差，用于评估跨窗口连续性。
"""

from typing import Dict, List

import numpy as np


class LongTermACFMetrics:
    """长程自相关评估指标计算器。
    
    计算真实序列与合成序列在长滞后范围内的自相关函数差异，
    用于评估合成数据的跨窗口连续性。
    """

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray, lag_range: List[int] = None) -> Dict:
        """计算长程自相关指标。
        
        Args:
            real_delay: 真实延迟序列
            syn_delay: 合成延迟序列
            lag_range: 滞后范围，默认 [1, 100]
            
        Returns:
            Dict: 包含长程自相关均方误差的指标字典
        """
        if lag_range is None:
            lag_range = [1, 100]
        
        # 计算自相关函数
        real_acf = self._calculate_acf(real_delay, lag_range)
        syn_acf = self._calculate_acf(syn_delay, lag_range)
        
        # 计算均方误差
        acf_mae = np.mean(np.abs(real_acf - syn_acf))
        
        return {
            "acf_mae_lag1_100": acf_mae
        }
    
    def _calculate_acf(self, data: np.ndarray, lag_range: List[int]) -> np.ndarray:
        """计算自相关函数。
        
        Args:
            data: 输入数据序列
            lag_range: 滞后范围
            
        Returns:
            np.ndarray: 自相关函数值
        """
        max_lag = lag_range[1]
        acf = []
        
        # 去均值
        data_mean = np.mean(data)
        data_normalized = data - data_mean
        
        # 计算自协方差
        for lag in range(lag_range[0], lag_range[1] + 1):
            # 分子：cov(X_t, X_{t+lag})
            numerator = np.sum(data_normalized[:-lag] * data_normalized[lag:])
            # 分母：var(X_t) * N
            denominator = np.sum(data_normalized ** 2)
            
            if denominator == 0:
                acf_val = 0.0
            else:
                acf_val = numerator / denominator
            
            acf.append(acf_val)
        
        return np.array(acf)
