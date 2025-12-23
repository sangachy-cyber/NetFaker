#!/usr/bin/env python3

"""计算局部统计特征
"""

import numpy as np


def compute_local_stats(
    del_up_qt: np.ndarray,   # shape (5,), values in [0, 1]
    del_dn_qt: np.ndarray,   # shape (5,), values in [0, 1]
    loss_up: np.ndarray,     # shape (5,), **binary 0/1**
    loss_dn: np.ndarray,      # shape (5,), **binary 0/1**
) -> np.ndarray:            # shape (10,)
    """从窗口的最后5个样本计算10个局部统计数据
    延迟输入在QuantileTransformer归一化空间中
    丢包输入必须是二进制的(0/1)

    统计数据顺序（示例）:
      0: del_up 均值
      1: del_up 标准差
      2: del_up 最大值
      3: del_dn 均值
      4: del_dn 标准差
      5: del_dn 最大值
      6: loss_up 比率
      7: loss_dn 比率
      8: 最大连续loss_up
      9: 最大连续loss_dn

    Returns:
        raw_local: 10维向量，Z-score之前的值

    """
    # 计算上行延迟统计量
    del_up_mean = np.mean(del_up_qt)
    del_up_std = np.std(del_up_qt)
    del_up_max = np.max(del_up_qt)

    # 计算下行延迟统计量
    del_dn_mean = np.mean(del_dn_qt)
    del_dn_std = np.std(del_dn_qt)
    del_dn_max = np.max(del_dn_qt)

    # 计算丢包率
    loss_up_rate = np.mean(loss_up)
    loss_dn_rate = np.mean(loss_dn)

    # 计算最大连续丢包数
    def max_consecutive(arr):
        if len(arr) == 0:
            return 0
        max_count = 0
        current_count = 0
        for val in arr:
            if val == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    max_consec_loss_up = max_consecutive(loss_up)
    max_consec_loss_dn = max_consecutive(loss_dn)

    # 组合所有统计量
    raw_local = np.array([
        del_up_mean,
        del_up_std,
        del_up_max,
        del_dn_mean,
        del_dn_std,
        del_dn_max,
        loss_up_rate,
        loss_dn_rate,
        max_consec_loss_up,
        max_consec_loss_dn,
    ])

    return raw_local
