#!/usr/bin/env python3
"""
展示行为发现的典型样本，包括不同行为类型的典型案例
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.behavior_visualization import select_behavior_typical_cases, generate_behavior_typical_cases



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



def main():
    """主函数"""
    print("=== 展示行为发现典型样本 ===")
    
    # 1. 加载窗口数据
    print("\n1. 加载窗口数据...")
    window_data = load_window_data()
    
    # 2. 加载行为标签
    print("2. 加载行为标签...")
    behavior_labels = load_behavior_labels()
    
    # 3. 选择典型案例
    print("3. 选择典型案例...")
    # 每个行为类型选择3个典型案例，让用户能更清楚地看到不同行为类型的特征
    typical_cases = select_behavior_typical_cases(window_data, behavior_labels, num_cases=3)
    
    # 4. 生成可视化图表
    print("4. 生成可视化图表...")
    
    # 生成图表
    output_dir = Path("output/behavior_typical_cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "behavior_typical_cases.png"
    
    # 使用可视化模块生成图表
    fig = generate_behavior_typical_cases(typical_cases, output_path=output_path)
    
    # 显示图表
    plt.show()
    plt.close(fig)  # 关闭图表，释放资源
    
    print(f"图表已保存到: {output_path}")
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
