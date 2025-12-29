#!/usr/bin/env python3
"""
分析被分类为稳定状态的样本，了解为什么高实验数据会被错误分类
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.behavior_discovery.pattern_identifier import PatternIdentifier


def load_window_data():
    """加载窗口数据
    
    Returns:
        list: 窗口数据列表
    """
    datasets_dir = Path("output/datasets")
    window_data = []
    
    # 遍历所有数据集文件
    for split in ["train", "val", "test"]:
        file_path = datasets_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                window_data.append(record)
    
    return window_data


def load_behavior_labels():
    """加载行为标签
    
    Returns:
        np.array: 行为标签数组
    """
    datasets_dir = Path("output/datasets")
    behavior_labels = []
    
    # 遍历所有数据集文件
    for split in ["train", "val", "test"]:
        file_path = datasets_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                # 行为标签存储在cond[11]位置
                behavior_id = int(record["cond"][11])
                behavior_labels.append(behavior_id)
    
    return np.array(behavior_labels)


def analyze_stable_samples(window_data, behavior_labels):
    """分析被分类为稳定状态的样本
    
    Args:
        window_data: 窗口数据列表
        behavior_labels: 行为标签数组
    """
    # 找出所有被分类为稳定状态的样本
    stable_indices = np.where(behavior_labels == 0)[0]
    
    print(f"\n=== 稳定状态样本分析 ===")
    print(f"总样本数: {len(behavior_labels)}")
    print(f"稳定状态样本数: {len(stable_indices)}")
    print(f"稳定状态占比: {len(stable_indices) / len(behavior_labels) * 100:.2f}%")
    
    # 创建PatternIdentifier实例，用于分析检测逻辑
    pattern_identifier = PatternIdentifier()
    
    # 分析前5个稳定状态样本
    for i in range(min(5, len(stable_indices))):
        idx = stable_indices[i]
        sample = window_data[idx]
        window_df = pd.DataFrame(sample["window"])
        
        print(f"\n=== 稳定状态样本 {i+1} ===")
        print(f"样本索引: {idx}")
        print(f"Trace ID: {sample['trace_id']}")
        
        # 提取时延和丢包率数据
        delay_up = window_df["delay_up_origin"].values
        delay_down = window_df["delay_down_origin"].values
        loss_up = window_df["loss_up_origin"].values
        loss_down = window_df["loss_down_origin"].values
        
        # 计算统计信息
        delay_mean_up = np.mean(delay_up)
        delay_std_up = np.std(delay_up)
        loss_mean_up = np.mean(loss_up)
        loss_std_up = np.std(loss_up)
        
        delay_mean_down = np.mean(delay_down)
        delay_std_down = np.std(delay_down)
        loss_mean_down = np.mean(loss_down)
        loss_std_down = np.std(loss_down)
        
        print(f"\n上行链路统计:")
        print(f"  平均时延: {delay_mean_up:.2f} ms")
        print(f"  时延标准差: {delay_std_up:.2f} ms")
        print(f"  平均丢包率: {loss_mean_up:.4f}")
        print(f"  丢包率标准差: {loss_std_up:.4f}")
        print(f"  最大时延: {np.max(delay_up):.2f} ms")
        print(f"  最大丢包率: {np.max(loss_up):.4f}")
        
        print(f"\n下行链路统计:")
        print(f"  平均时延: {delay_mean_down:.2f} ms")
        print(f"  时延标准差: {delay_std_down:.2f} ms")
        print(f"  平均丢包率: {loss_mean_down:.4f}")
        print(f"  丢包率标准差: {loss_std_down:.4f}")
        print(f"  最大时延: {np.max(delay_down):.2f} ms")
        print(f"  最大丢包率: {np.max(loss_down):.4f}")
        
        # 检查稳定状态条件
        thresholds = {
            "delay_mean_low": pattern_identifier.DEFAULT_DELAY_MEAN_LOW,
            "delay_mean_high": pattern_identifier.DEFAULT_DELAY_MEAN_HIGH,
            "delay_std_high": pattern_identifier.DEFAULT_DELAY_STD_HIGH,
            "loss_mean_low": pattern_identifier.DEFAULT_LOSS_MEAN_LOW,
            "loss_mean_high": pattern_identifier.DEFAULT_LOSS_MEAN_HIGH,
            "loss_std_high": pattern_identifier.DEFAULT_LOSS_STD_HIGH,
        }
        
        # 计算稳定状态条件
        print(f"\n稳定状态条件检查 (上行):")
        condition1 = loss_mean_up <= thresholds["loss_mean_low"] * 1.3
        condition2 = delay_std_up < thresholds["delay_std_high"] / 2.5
        condition3 = loss_std_up < thresholds["loss_std_high"] / 2.5
        condition4 = delay_mean_up < thresholds["delay_mean_high"] * 0.7
        condition5 = np.max(delay_up) < pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8
        condition6 = np.max(loss_up) < pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8
        
        print(f"  1. 丢包率 <= {thresholds['loss_mean_low'] * 1.3:.4f}: {condition1}")
        print(f"  2. 时延标准差 < {thresholds['delay_std_high'] / 2.5:.2f} ms: {condition2}")
        print(f"  3. 丢包率标准差 < {thresholds['loss_std_high'] / 2.5:.4f}: {condition3}")
        print(f"  4. 平均时延 < {thresholds['delay_mean_high'] * 0.7:.2f} ms: {condition4}")
        print(f"  5. 最大时延 < {pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8:.2f} ms: {condition5}")
        print(f"  6. 最大丢包率 < {pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8:.4f}: {condition6}")
        print(f"  所有条件满足: {condition1 and condition2 and condition3 and condition4 and condition5 and condition6}")
        
        print(f"\n稳定状态条件检查 (下行):")
        condition1_dn = loss_mean_down <= thresholds["loss_mean_low"] * 1.3
        condition2_dn = delay_std_down < thresholds["delay_std_high"] / 2.5
        condition3_dn = loss_std_down < thresholds["loss_std_high"] / 2.5
        condition4_dn = delay_mean_down < thresholds["delay_mean_high"] * 0.7
        condition5_dn = np.max(delay_down) < pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8
        condition6_dn = np.max(loss_down) < pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8
        
        print(f"  1. 丢包率 <= {thresholds['loss_mean_low'] * 1.3:.4f}: {condition1_dn}")
        print(f"  2. 时延标准差 < {thresholds['delay_std_high'] / 2.5:.2f} ms: {condition2_dn}")
        print(f"  3. 丢包率标准差 < {thresholds['loss_std_high'] / 2.5:.4f}: {condition3_dn}")
        print(f"  4. 平均时延 < {thresholds['delay_mean_high'] * 0.7:.2f} ms: {condition4_dn}")
        print(f"  5. 最大时延 < {pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8:.2f} ms: {condition5_dn}")
        print(f"  6. 最大丢包率 < {pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8:.4f}: {condition6_dn}")
        print(f"  所有条件满足: {condition1_dn and condition2_dn and condition3_dn and condition4_dn and condition5_dn and condition6_dn}")
        
        # 检查是否应该被分类为其他行为类型
        print(f"\n其他行为类型检查:")
        
        # 检查瞬时峰值
        instant_spike_count_up = np.sum((delay_up >= pattern_identifier.INSTANT_SPIKE_DELAY_THRESHOLD) | (loss_up >= pattern_identifier.INSTANT_SPIKE_LOSS_THRESHOLD))
        instant_spike_count_down = np.sum((delay_down >= pattern_identifier.INSTANT_SPIKE_DELAY_THRESHOLD) | (loss_down >= pattern_identifier.INSTANT_SPIKE_LOSS_THRESHOLD))
        print(f"  瞬时峰值计数 (上行): {instant_spike_count_up}")
        print(f"  瞬时峰值计数 (下行): {instant_spike_count_down}")
        
        # 检查高延迟无丢包
        high_delay_no_loss_up = (delay_mean_up > thresholds["delay_mean_high"]) and (loss_mean_up <= thresholds["loss_mean_low"])
        high_delay_no_loss_down = (delay_mean_down > thresholds["delay_mean_high"]) and (loss_mean_down <= thresholds["loss_mean_low"])
        print(f"  高延迟无丢包 (上行): {high_delay_no_loss_up}")
        print(f"  高延迟无丢包 (下行): {high_delay_no_loss_down}")
        
        # 检查强突发
        # 简化的强突发检查
        strong_burst_up = (delay_mean_up > pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8) and (loss_mean_up > pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8)
        strong_burst_down = (delay_mean_down > pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8) and (loss_mean_down > pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8)
        print(f"  接近强突发 (上行): {strong_burst_up}")
        print(f"  接近强突发 (下行): {strong_burst_down}")


def main():
    """主函数"""
    print("=== 分析稳定状态样本 ===")
    
    # 加载窗口数据
    print("\n1. 加载窗口数据...")
    window_data = load_window_data()
    
    # 加载行为标签
    print("2. 加载行为标签...")
    behavior_labels = load_behavior_labels()
    
    # 分析稳定状态样本
    print("3. 分析稳定状态样本...")
    analyze_stable_samples(window_data, behavior_labels)
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
