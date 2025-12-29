#!/usr/bin/env python3
"""
分析之前被错误分类的特定样本，确认修复效果
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


def analyze_specific_samples(window_data, behavior_labels):
    """分析特定样本
    
    Args:
        window_data: 窗口数据列表
        behavior_labels: 行为标签数组
    """
    # 创建PatternIdentifier实例，用于分析检测逻辑
    pattern_identifier = PatternIdentifier()
    
    # 定义BEHAVIOR_LABELS反向映射
    LABEL_TO_NAME = {
        0: "STABLE",
        1: "WEAK_BURST",
        2: "FREQUENT_FLUCTUATION",
        3: "HIGH_DELAY_NO_LOSS",
        4: "HIGH_LOSS_STEADY",
        5: "LOW_DELAY_HIGH_LOSS",
        6: "STRONG_BURST",
        7: "INSTANT_SPIKE",
        8: "INVALID"
    }
    
    # 分析样本0和样本1（之前被错误分类为稳定状态的样本）
    specific_indices = [0, 1]
    
    for idx in specific_indices:
        sample = window_data[idx]
        behavior_id = behavior_labels[idx]
        behavior_name = LABEL_TO_NAME.get(behavior_id, f"未知 ({behavior_id})")
        
        window_df = pd.DataFrame(sample["window"])
        
        print(f"\n=== 样本 {idx} 分析 ===")
        print(f"Trace ID: {sample['trace_id']}")
        print(f"当前分类: {behavior_id} - {behavior_name}")
        
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
        
        # 检查行为检测条件
        thresholds = {
            "delay_mean_low": pattern_identifier.DEFAULT_DELAY_MEAN_LOW,
            "delay_mean_high": pattern_identifier.DEFAULT_DELAY_MEAN_HIGH,
            "delay_std_high": pattern_identifier.DEFAULT_DELAY_STD_HIGH,
            "loss_mean_low": pattern_identifier.DEFAULT_LOSS_MEAN_LOW,
            "loss_mean_high": pattern_identifier.DEFAULT_LOSS_MEAN_HIGH,
            "loss_std_high": pattern_identifier.DEFAULT_LOSS_STD_HIGH,
        }
        
        print(f"\n行为检测条件检查:")
        print(f"  静态阈值:")
        print(f"    delay_mean_low: {thresholds['delay_mean_low']} ms")
        print(f"    delay_mean_high: {thresholds['delay_mean_high']} ms")
        print(f"    delay_std_high: {thresholds['delay_std_high']} ms")
        
        # 检查高延迟条件
        high_delay_up = delay_mean_up > thresholds["delay_mean_high"] * 0.7
        high_delay_down = delay_mean_down > thresholds["delay_mean_high"] * 0.7
        print(f"  高延迟条件 (delay_mean > {thresholds['delay_mean_high'] * 0.7:.2f} ms):")
        print(f"    上行: {high_delay_up}")
        print(f"    下行: {high_delay_down}")
        
        # 检查延迟波动大条件
        delay_fluctuation_up = delay_std_up > thresholds["delay_std_high"] / 2.0
        delay_fluctuation_down = delay_std_down > thresholds["delay_std_high"] / 2.0
        print(f"  延迟波动大条件 (delay_std > {thresholds['delay_std_high'] / 2.0:.2f} ms):")
        print(f"    上行: {delay_fluctuation_up}")
        print(f"    下行: {delay_fluctuation_down}")
        
        # 检查稳定状态条件
        stable_up = (
            loss_mean_up <= thresholds["loss_mean_low"] * 1.3 and 
            delay_mean_up < thresholds["delay_mean_high"] * 0.7 and
            delay_std_up < thresholds["delay_std_high"] / 2.5 and 
            loss_std_up < thresholds["loss_std_high"] / 2.5 and 
            np.max(delay_up) < pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8 and
            np.max(loss_up) < pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8
        )
        
        stable_down = (
            loss_mean_down <= thresholds["loss_mean_low"] * 1.3 and 
            delay_mean_down < thresholds["delay_mean_high"] * 0.7 and
            delay_std_down < thresholds["delay_std_high"] / 2.5 and 
            loss_std_down < thresholds["loss_std_high"] / 2.5 and 
            np.max(delay_down) < pattern_identifier.STRONG_BURST_DELAY_THRESHOLD * 0.8 and
            np.max(loss_down) < pattern_identifier.STRONG_BURST_LOSS_THRESHOLD * 0.8
        )
        
        print(f"  稳定状态条件:")
        print(f"    上行: {stable_up}")
        print(f"    下行: {stable_down}")


def main():
    """主函数"""
    print("=== 分析特定样本 ===")
    
    # 加载窗口数据
    print("\n1. 加载窗口数据...")
    window_data = load_window_data()
    
    # 加载行为标签
    print("2. 加载行为标签...")
    behavior_labels = load_behavior_labels()
    
    # 分析特定样本
    print("3. 分析特定样本...")
    analyze_specific_samples(window_data, behavior_labels)
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
