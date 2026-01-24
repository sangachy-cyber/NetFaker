"""时序依赖指标计算模块。

负责计算单窗口内合成数据与真实数据的时序依赖差异，
包括自相关函数的平均绝对误差。
"""

from typing import Dict

import numpy as np
from statsmodels.tsa.stattools import acf


class TemporalMetrics:
    """时序依赖指标计算器。
    
    实现真实与合成数据在时序依赖上的差异评估，
    包括自相关函数的平均绝对误差。
    """

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray) -> Dict:
        """计算时序依赖指标。
        
        Args:
            real_delay: 真实延迟数据 (ms)
            syn_delay: 合成延迟数据 (ms)
            
        Returns:
            Dict: 包含时序依赖指标的字典
        """
        # 计算自相关函数（lag=1~10）
        real_acf = acf(real_delay, nlags=10, fft=True, missing='drop')[1:]  # 去掉lag=0
        syn_acf = acf(syn_delay, nlags=10, fft=True, missing='drop')[1:]  # 去掉lag=0

        # 计算MAE（平均绝对误差）
        acf_mae_lag1_10 = np.mean(np.abs(real_acf - syn_acf))

        return {
            'acf_mae_lag1_10': acf_mae_lag1_10
        }
