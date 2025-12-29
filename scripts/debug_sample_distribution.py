#!/usr/bin/env python3
"""
调试样本分布，分析为什么所有样本都被分类为FREQUENT_FLUCTUATION
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_windows_data() -> List[Dict]:
    """加载窗口数据"""
    windows_path = Path("assets/windows_meta.pkl")
    if not windows_path.exists():
        print("窗口数据文件不存在，请先运行预处理")
        return []
    
    return pd.read_pickle(windows_path).to_dict("records")


def analyze_sample_distribution(windows: List[Dict]):
    """分析样本分布"""
    if not windows:
        print("没有窗口数据")
        return
    
    print(f"=== 分析 {len(windows)} 个样本的分布 ===")
    
    # 收集统计数据
    delay_means = []
    delay_stds = []
    loss_means = []
    loss_stds = []
    
    for window in windows[:100]:  # 只分析前100个样本
        window_df = window["window"]
        if window_df.empty:
            continue
        
        # 获取上下行数据
        delay_up = window_df["delay_up"].values
        loss_up = window_df["loss_up"].values
        delay_down = window_df["delay_down"].values
        loss_down = window_df["loss_dn"].values
        
        # 计算统计信息
        for delays, loss_rates in [(delay_up, loss_up), (delay_down, loss_down)]:
            if len(delays) > 0:
                delay_means.append(np.mean(delays))
                delay_stds.append(np.std(delays))
            if len(loss_rates) > 0:
                loss_means.append(np.mean(loss_rates))
                loss_stds.append(np.std(loss_rates))
    
    # 打印统计结果
    print(f"\n时延统计:")
    print(f"  平均时延均值: {np.mean(delay_means):.2f} ms")
    print(f"  时延均值中位数: {np.median(delay_means):.2f} ms")
    print(f"  最大时延均值: {np.max(delay_means):.2f} ms")
    print(f"  最小时延均值: {np.min(delay_means):.2f} ms")
    print(f"  平均时延标准差: {np.mean(delay_stds):.2f} ms")
    print(f"  时延标准差中位数: {np.median(delay_stds):.2f} ms")
    print(f"  最大时延标准差: {np.max(delay_stds):.2f} ms")
    
    print(f"\n丢包率统计:")
    print(f"  平均丢包率均值: {np.mean(loss_means):.4f}")
    print(f"  丢包率均值中位数: {np.median(loss_means):.4f}")
    print(f"  最大丢包率均值: {np.max(loss_means):.4f}")
    print(f"  平均丢包率标准差: {np.mean(loss_stds):.4f}")
    print(f"  丢包率标准差中位数: {np.median(loss_stds):.4f}")
    print(f"  最大丢包率标准差: {np.max(loss_stds):.4f}")
    
    # 分析延迟波动大的样本
    high_delay_std_samples = sum(1 for std in delay_stds if std > 100)  # 标准差大于100ms的样本
    print(f"\n时延标准差 > 100ms 的样本占比: {high_delay_std_samples / len(delay_stds) * 100:.2f}%")


def analyze_individual_sample(window: Dict, index: int):
    """分析单个样本"""
    print(f"\n=== 样本 {index} 详细分析 ===")
    window_df = window["window"]
    
    # 获取上下行数据
    delay_up = window_df["delay_up"].values
    loss_up = window_df["loss_up"].values
    delay_down = window_df["delay_down"].values
    loss_down = window_df["loss_dn"].values
    
    print(f"上行链路:")
    print(f"  平均时延: {np.mean(delay_up):.2f} ms")
    print(f"  时延标准差: {np.std(delay_up):.2f} ms")
    print(f"  最大时延: {np.max(delay_up):.2f} ms")
    print(f"  最小时延: {np.min(delay_up):.2f} ms")
    print(f"  平均丢包率: {np.mean(loss_up):.4f}")
    print(f"  丢包率标准差: {np.std(loss_up):.4f}")
    
    print(f"下行链路:")
    print(f"  平均时延: {np.mean(delay_down):.2f} ms")
    print(f"  时延标准差: {np.std(delay_down):.2f} ms")
    print(f"  最大时延: {np.max(delay_down):.2f} ms")
    print(f"  最小时延: {np.min(delay_down):.2f} ms")
    print(f"  平均丢包率: {np.mean(loss_down):.4f}")
    print(f"  丢包率标准差: {np.std(loss_down):.4f}")


if __name__ == "__main__":
    windows = load_windows_data()
    if windows:
        analyze_sample_distribution(windows)
        # 分析前几个样本
        for i in range(3):
            analyze_individual_sample(windows[i], i+1)
    print("\n=== 完成 ===")
