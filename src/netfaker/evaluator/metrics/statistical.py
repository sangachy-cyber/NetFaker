"""统计分布指标计算模块。

负责计算单窗口内合成数据与真实数据的统计分布差异，
包括延迟P90/P99偏差和KS距离。
"""

from typing import Dict

import numpy as np
from scipy.stats import ks_2samp


class StatisticalMetrics:
    """统计分布指标计算器。
    
    实现真实与合成数据在统计分布上的差异评估，
    包括延迟P90/P99偏差和KS距离。
    """

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray) -> Dict:
        """计算统计分布指标。
        
        Args:
            real_delay: 真实延迟数据 (ms)
            syn_delay: 合成延迟数据 (ms)
            
        Returns:
            Dict: 包含统计分布指标的字典
        """
        # 计算P90延迟偏差
        p90_real = np.percentile(real_delay, 90)
        p90_syn = np.percentile(syn_delay, 90)
        p90_delay_err = abs(p90_real - p90_syn)

        # 计算P99延迟偏差
        p99_real = np.percentile(real_delay, 99)
        p99_syn = np.percentile(syn_delay, 99)
        p99_delay_err = abs(p99_real - p99_syn)

        # 计算KS距离
        ks_statistic, _ = ks_2samp(real_delay, syn_delay)

        return {
            'p90_delay_err': p90_delay_err,
            'p99_delay_err': p99_delay_err,
            'ks_statistic': ks_statistic
        }
