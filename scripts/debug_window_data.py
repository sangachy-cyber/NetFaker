#!/usr/bin/env python3
"""
调试窗口数据，查看具体的窗口内容
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_window_data():
    """加载窗口数据"""
    datasets_dir = Path("output/datasets")
    window_data = []
    behavior_labels = []
    
    # 遍历所有数据集文件
    for split in ["train", "val", "test"]:
        file_path = datasets_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                window_data.append(record)
                # 行为标签位于cond[11]位置
                behavior_id = int(record["cond"][11])
                behavior_labels.append(behavior_id)
    
    return window_data, np.array(behavior_labels)


def select_behavior_typical_cases(window_data, behavior_labels, num_cases=3):
    """从每个行为类型中选择典型案例"""
    typical_cases = {}
    unique_behaviors = np.unique(behavior_labels)
    
    for behavior_id in unique_behaviors:
        if behavior_id == 8:  # 跳过INVALID类型
            continue
            
        # 找出当前行为类型的所有窗口
        behavior_indices = np.where(behavior_labels == behavior_id)[0]
        if len(behavior_indices) == 0:
            continue
        
        # 随机选择样本
        np.random.seed(42)  # 设置随机种子，确保结果可复现
        selected_indices = np.random.choice(behavior_indices, 
                                           size=min(num_cases, len(behavior_indices)), 
                                           replace=False)
        
        cases = []
        for idx in selected_indices:
            cases.append({
                "window": window_data[idx],
                "behavior_id": behavior_id,
                "original_idx": idx
            })
        
        typical_cases[behavior_id] = cases
    
    return typical_cases


def find_original_file(trace_id, data_dir=Path("data")):
    """查找原始数据文件"""
    # 遍历数据目录
    for file_path in data_dir.rglob("*.txt"):
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # 检查trace_id
            for i, line in enumerate(lines):
                if trace_id in line:
                    return file_path, i+1  # 返回文件路径和行号（从1开始）
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
    
    return None, None


def main():
    """主函数"""
    print("=== 调试窗口数据 ===")
    
    # 加载窗口数据
    print("\n1. 加载窗口数据...")
    window_data, behavior_labels = load_window_data()
    print(f"加载了 {len(window_data)} 个窗口数据")
    
    # 选择典型案例
    print("\n2. 选择典型案例...")
    typical_cases = select_behavior_typical_cases(window_data, behavior_labels, num_cases=3)
    
    # 查看行为类别5的案例2
    behavior_id = 5
    case_idx = 1  # 索引从0开始，所以案例2对应索引1
    
    if behavior_id in typical_cases and len(typical_cases[behavior_id]) > case_idx:
        case = typical_cases[behavior_id][case_idx]
        print(f"\n3. 查看行为类别 {behavior_id} 的案例 {case_idx+1}:")
        print(f"   原始索引: {case['original_idx']}")
        print(f"   窗口数据类型: {type(case['window'])}")
        
        # 检查窗口数据结构
        window_item = case['window']
        if isinstance(window_item, dict):
            print(f"   窗口数据字典键: {list(window_item.keys())}")
            
            # 查看trace_id和start_time
            if 'trace_id' in window_item:
                print(f"   Trace ID: {window_item['trace_id']}")
            if 'start_time' in window_item:
                print(f"   Start Time: {window_item['start_time']}")
        
        # 提取窗口数据
        if isinstance(window_item, dict) and "window" in window_item:
            window_df = pd.DataFrame(window_item["window"])
        else:
            window_df = pd.DataFrame(window_item)
        
        print(f"   窗口数据形状: {window_df.shape}")
        print(f"   窗口数据列名: {list(window_df.columns)}")
        
        # 查看前几行数据
        print("\n   窗口数据前5行:")
        print(window_df.head())
        
        # 查看数据变化范围
        print("\n   数据变化范围:")
        for col in ['delay_up_origin', 'delay_down_origin', 'loss_up_origin', 'loss_down_origin']:
            if col in window_df.columns:
                min_val = window_df[col].min()
                max_val = window_df[col].max()
                mean_val = window_df[col].mean()
                print(f"   {col}: 最小值={min_val:.3f}, 最大值={max_val:.3f}, 平均值={mean_val:.3f}")
        
        # 检查是否有重复值
        print("\n   数据统计:")
        for col in window_df.columns:
            if window_df[col].dtype in ['int64', 'float64']:
                unique_vals = window_df[col].nunique()
                total_vals = len(window_df[col])
                print(f"   {col}: 唯一值数量 = {unique_vals}, 总数量 = {total_vals}")
                if unique_vals == 1:
                    print(f"     警告: {col} 所有值都相同！值为: {window_df[col].iloc[0]}")
        
        # 查找原始文件
        print("\n4. 查找原始数据文件...")
        if isinstance(window_item, dict) and 'trace_id' in window_item:
            trace_id = window_item['trace_id']
            file_path, line_num = find_original_file(trace_id)
            if file_path:
                print(f"   找到原始文件: {file_path}")
                print(f"   行号: {line_num}")
                
                # 查看原始文件的相关内容
                print("\n   原始文件相关内容:")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    # 查看前后5行
                    start_line = max(0, line_num - 6)
                    end_line = min(len(lines), line_num + 5)
                    
                    for i in range(start_line, end_line):
                        prefix = ">>> " if i == line_num - 1 else "    "
                        print(f"   {prefix}{i+1:4d}: {lines[i].rstrip()}")
                except Exception as e:
                    print(f"   读取原始文件时出错: {e}")
            else:
                print(f"   未找到包含trace_id {trace_id} 的原始文件")
    else:
        print(f"\n未找到行为类别 {behavior_id} 的案例 {case_idx+1}")
        print(f"可用的行为类别: {list(typical_cases.keys())}")
        if behavior_id in typical_cases:
            print(f"行为类别 {behavior_id} 的案例数量: {len(typical_cases[behavior_id])}")


if __name__ == "__main__":
    main()
