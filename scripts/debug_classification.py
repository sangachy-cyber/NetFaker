#!/usr/bin/env python3
"""
详细分析样本分类问题，找出状态与描述不一致的原因
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


def analyze_sample(sample, pattern_identifier):
    """分析单个样本
    
    Args:
        sample: 单个样本数据
        pattern_identifier: 行为识别器实例
    
    Returns:
        dict: 分析结果
    """
    # 提取时延和丢包率数据
    window_df = pd.DataFrame(sample["window"])
    
    # 检查数据结构
    if "delay_up_origin" in window_df.columns and "delay_down_origin" in window_df.columns:
        delay_up = window_df["delay_up_origin"].values
        delay_down = window_df["delay_down_origin"].values
        loss_up = window_df.get("loss_up_origin", np.zeros(len(window_df))).values
        loss_down = window_df.get("loss_down_origin", np.zeros(len(window_df))).values
    else:
        print(f"样本数据不包含必要列: {window_df.columns}")
        return None
    
    # 计算统计信息
    delay_mean_up = np.mean(delay_up)
    delay_std_up = np.std(delay_up)
    loss_mean_up = np.mean(loss_up)
    loss_std_up = np.std(loss_up)
    
    delay_mean_down = np.mean(delay_down)
    delay_std_down = np.std(delay_down)
    loss_mean_down = np.mean(loss_down)
    loss_std_down = np.std(loss_down)
    
    # 获取行为标签
    thresholds = {
        "delay_mean_low": pattern_identifier.DEFAULT_DELAY_MEAN_LOW,
        "delay_mean_high": pattern_identifier.DEFAULT_DELAY_MEAN_HIGH,
        "delay_std_high": pattern_identifier.DEFAULT_DELAY_STD_HIGH,
        "loss_mean_low": pattern_identifier.DEFAULT_LOSS_MEAN_LOW,
        "loss_mean_high": pattern_identifier.DEFAULT_LOSS_MEAN_HIGH,
        "loss_std_high": pattern_identifier.DEFAULT_LOSS_STD_HIGH,
    }
    
    behavior_up = pattern_identifier._detect_behavior_with_raw_data(delay_up, loss_up, thresholds)
    behavior_down = pattern_identifier._detect_behavior_with_raw_data(delay_down, loss_down, thresholds)
    
    # 获取行为名称
    BEHAVIOR_NAMES = {
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
    
    return {
        "delay_up": {
            "mean": delay_mean_up,
            "std": delay_std_up,
            "max": np.max(delay_up),
            "min": np.min(delay_up)
        },
        "delay_down": {
            "mean": delay_mean_down,
            "std": delay_std_down,
            "max": np.max(delay_down),
            "min": np.min(delay_down)
        },
        "loss_up": {
            "mean": loss_mean_up,
            "std": loss_std_up,
            "max": np.max(loss_up),
            "min": np.min(loss_up)
        },
        "loss_down": {
            "mean": loss_mean_down,
            "std": loss_std_down,
            "max": np.max(loss_down),
            "min": np.min(loss_down)
        },
        "behavior_up": behavior_up,
        "behavior_down": behavior_down,
        "behavior_up_name": BEHAVIOR_NAMES.get(behavior_up, f"未知({behavior_up})"),
        "behavior_down_name": BEHAVIOR_NAMES.get(behavior_down, f"未知({behavior_down})"),
        "sample": sample
    }


def main():
    """主函数"""
    print("=== 分析样本分类问题 ===")
    
    # 加载窗口数据
    print("\n1. 加载窗口数据...")
    window_data = load_window_data()
    print(f"加载了 {len(window_data)} 个样本")
    
    # 创建行为识别器实例
    pattern_identifier = PatternIdentifier()
    
    # 收集各类行为的样本
    print("\n2. 分析样本...")
    
    # 按行为类型分类样本
    behavior_samples = {
        0: [],  # 稳定
        1: [],  # 弱突发
        2: [],  # 频繁波动
        3: [],  # 高时延无丢包
        4: [],  # 持续高丢包
        5: [],  # 低时延高丢包
        6: [],  # 强突发
        7: [],  # 瞬时峰值
        8: []   # 无效
    }
    
    # 收集所有样本，按行为类型分类
    for sample in window_data:
        if "cond" in sample and len(sample["cond"]) > 11:
            behavior_id = int(sample["cond"][11])
            if behavior_id in behavior_samples:
                behavior_samples[behavior_id].append(sample)
    
    # 打印各类行为样本数量
    behavior_names = {
        0: "稳定状态",
        1: "弱突发",
        2: "频繁波动",
        3: "高时延无丢包",
        4: "持续高丢包",
        5: "低时延高丢包",
        6: "强突发",
        7: "瞬时峰值",
        8: "无效"
    }
    
    print("各类行为样本数量:")
    for behavior_id, samples in behavior_samples.items():
        print(f"  {behavior_names.get(behavior_id, f'未知({behavior_id})')}: {len(samples)} 个样本")
    
    # 详细分析各类行为的前2个样本
    for behavior_id in [0, 1, 2]:  # 重点分析稳定、弱突发、频繁波动
        samples = behavior_samples.get(behavior_id, [])
        if not samples:
            continue
        
        behavior_name = behavior_names.get(behavior_id, f'未知({behavior_id})')
        print(f"\n=== {behavior_name} 样本详细分析 ===")
        
        # 分析前2个样本
        for i, sample in enumerate(samples[:2]):
            print(f"\n--- {behavior_name} 样本 {i+1} ---")
            result = analyze_sample(sample, pattern_identifier)
            if result:
                print(f"上行链路:")
                print(f"  平均时延: {result['delay_up']['mean']:.2f} ms")
                print(f"  时延标准差: {result['delay_up']['std']:.2f} ms")
                print(f"  最大时延: {result['delay_up']['max']:.2f} ms")
                print(f"  最小时延: {result['delay_up']['min']:.2f} ms")
                print(f"  平均丢包率: {result['loss_up']['mean']:.4f}")
                print(f"  丢包率标准差: {result['loss_up']['std']:.4f}")
                print(f"  行为分类: {result['behavior_up_name']}")
                
                print(f"\n下行链路:")
                print(f"  平均时延: {result['delay_down']['mean']:.2f} ms")
                print(f"  时延标准差: {result['delay_down']['std']:.2f} ms")
                print(f"  最大时延: {result['delay_down']['max']:.2f} ms")
                print(f"  最小时延: {result['delay_down']['min']:.2f} ms")
                print(f"  平均丢包率: {result['loss_down']['mean']:.4f}")
                print(f"  丢包率标准差: {result['loss_down']['std']:.4f}")
                print(f"  行为分类: {result['behavior_down_name']}")
                
                # 检查是否存在高时延或高波动
                if result['delay_up']['std'] > 50 or result['delay_down']['std'] > 50:
                    print(f"  ⚠️  注意: 存在高波动")
                if result['delay_up']['max'] > 500 or result['delay_down']['max'] > 500:
                    print(f"  ⚠️  注意: 存在高时延峰值")
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
