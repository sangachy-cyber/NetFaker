"""频域特性指标计算模块。

负责计算单窗口内合成数据与真实数据的频域特性差异，
包括Welch PSD在log域的平均绝对误差。
"""

from typing import Dict

import numpy as np
from scipy.signal import welch


class SpectralMetrics:
    """频域特性指标计算器。
    
    实现真实与合成数据在频域特性上的差异评估，
    包括Welch PSD在log域的平均绝对误差。
    """

    def calculate(self, real_delay: np.ndarray, syn_delay: np.ndarray) -> Dict:
        """计算频域特性指标。
        
        Args:
            real_delay: 真实延迟数据 (ms)
            syn_delay: 合成延迟数据 (ms)
            
        Returns:
            Dict: 包含频域特性指标的字典
        """
        # 计算Welch PSD（功率谱密度）
        f_real, psd_real = welch(real_delay, fs=10.0, nperseg=50, noverlap=25)
        f_syn, psd_syn = welch(syn_delay, fs=10.0, nperseg=50, noverlap=25)

        # 确保频率轴对齐
        if not np.array_equal(f_real, f_syn):
            # 如果频率轴不同，使用插值对齐
            psd_syn = np.interp(f_real, f_syn, psd_syn)

        # 转换到log域，避免极小值问题
        log_psd_real = np.log10(psd_real + 1e-10)  # 加小值避免log(0)
        log_psd_syn = np.log10(psd_syn + 1e-10)

        # 计算log域的MAE
        psd_log_mae = np.mean(np.abs(log_psd_real - log_psd_syn))

        return {
            'psd_log_mae': psd_log_mae
        }
