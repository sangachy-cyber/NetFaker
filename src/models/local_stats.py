#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
局部统计量计算
基于条件扩散模型方案.md实现
"""

import numpy as np


def compute_local_stats(
    del_up_qt: np.ndarray,   # shape (5,), values in [0, 1]
    del_dn_qt: np.ndarray,   # shape (5,), values in [0, 1]
    loss_up: np.ndarray,     # shape (5,), **binary 0/1**
    loss_dn: np.ndarray      # shape (5,), **binary 0/1**
) -> np.ndarray:            # shape (10,)
    """
    Compute 10 local statistics from the last 5 samples of a window.
    Delay inputs are in QuantileTransformer-normalized space.
    Loss inputs must be binary (0/1).

    Statistics order (example):
      0: del_up mean
      1: del_up std
      2: del_up max
      3: del_dn mean
      4: del_dn std
      5: del_dn max
      6: loss_up rate
      7: loss_dn rate
      8: max consecutive loss_up
      9: max consecutive loss_dn

    Returns:
        raw_local: 10-dim vector BEFORE Z-scoring.
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
        max_consec_loss_dn
    ])
    
    return raw_local