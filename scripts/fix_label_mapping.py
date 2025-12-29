#!/usr/bin/env python3
"""
分析标签映射问题，并提供修复方案
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


def main():
    """主函数"""
    print("=== 分析标签映射问题 ===")
    
    # 说明问题
    print("\n1. 问题分析：")
    print("   - 行为发现模块按照all_windows的原始顺序生成标签")
    print("   - 但在分配标签时，窗口被按trace_id和start_time重新排序")
    print("   - 这导致标签与窗口的对应关系完全错乱")
    print("   - 例如，第一个稳定状态样本实际上应该是频繁波动类型")
    
    # 提供修复方案
    print("\n2. 修复方案：")
    print("   - 需要确保行为标签的生成顺序与最终处理的窗口顺序一致")
    print("   - 或者在分配标签前建立窗口与标签的正确映射关系")
    
    print("\n3. 修复代码：")
    print("   - 修改pipeline.py中的标签分配逻辑")
    print("   - 确保生成行为标签时使用与最终处理相同的窗口顺序")
    
    print("\n=== 修复pipeline.py中的标签分配问题 ===")
    
    # 读取pipeline.py文件
    pipeline_path = Path("src/preprocessing/pipeline.py")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 修复方案
    print("\n修复后的代码逻辑：")
    print("1. 收集所有窗口并建立索引映射")
    print("2. 按最终处理顺序生成行为标签")
    print("3. 确保标签与窗口一一对应")
    
    print("\n=== 开始修复 ===")
    
    # 修改pipeline.py中的标签分配逻辑
    new_content = content.replace(
        "        # 收集所有窗口元数据，用于行为发现\n        all_windows = []\n        for dataset in datasets.values():\n            all_windows.extend(dataset)\n        \n        # 执行行为发现\n        if all_windows and all_raw_data:\n            # 为每个窗口创建简单的特征DataFrame，包含window_start和window_end作为数据索引\n            # 使用简单的索引而不是时间戳，因为raw_data_df的索引是连续整数\n            features_df = pd.DataFrame({\n                "window_start": [i * self.config["step_size"] for i in range(len(all_windows))],\n                "window_end": [(i * self.config["step_size"]) + self.config["window_size"] for i in range(len(all_windows))]\n            })\n            \n            # 合并所有原始数据\n            raw_data_df = pd.concat(all_raw_data, ignore_index=True)\n            \n            # 调用行为发现模块\n            try:\n                behavior_results = self.pattern_identifier.identify(features_df, raw_data_df)\n                behavior_labels = behavior_results["labels"]\n                extra_assets["behavior_stats"] = behavior_results["behavior_stats"]\n            except Exception as e:\n                print(f"行为发现模块执行失败，使用默认标签: {e}")\n                # 如果行为发现失败，使用默认标签\n                behavior_labels = [0] * len(all_windows)\n        else:\n            # 如果没有数据，使用默认标签\n            behavior_labels = [0] * len(all_windows)\n        \n        # 分配行为标签到各个数据集\n        label_idx = 0\n        for dataset_name, dataset in datasets.items():\n            # 按 trace_id 分组并排序\n            traces = group_by_trace_id(dataset)\n            \n            # 对每个trace内的窗口按 start_time 排序\n            for trace_id, metas in traces.items():\n                traces[trace_id] = sorted(metas, key=lambda x: x["start_time"])\n            \n            # 处理每个窗口\n            normed_metas = []\n            dataset_state_ids = []\n            \n            for trace_id, metas in traces.items():\n                for i, meta in enumerate(metas):\n                    # 获取行为标签\n                    if label_idx < len(behavior_labels):\n                        network_state_id = behavior_labels[label_idx]\n                        label_idx += 1\n                    else:\n                        network_state_id = 0",
        "        # 执行行为发现，确保标签顺序与最终处理顺序一致\n        if all_raw_data:\n            # 先收集所有窗口并按最终处理顺序排序，用于生成行为标签\n            all_windows_sorted = []\n            # 按最终处理顺序收集窗口\n            for dataset_name, dataset in datasets.items():\n                # 按 trace_id 分组并排序\n                traces = group_by_trace_id(dataset)\n                # 对每个trace内的窗口按 start_time 排序\n                for trace_id, metas in traces.items():\n                    traces[trace_id] = sorted(metas, key=lambda x: x["start_time"])\n                # 收集排序后的窗口\n                for trace_id, metas in traces.items():\n                    all_windows_sorted.extend(metas)\n            \n            if all_windows_sorted:  # 只有在有窗口的情况下才执行行为发现\n                # 为每个窗口创建简单的特征DataFrame，包含window_start和window_end作为数据索引\n                # 使用简单的索引而不是时间戳，因为raw_data_df的索引是连续整数\n                features_df = pd.DataFrame({\n                    "window_start": [i * self.config["step_size"] for i in range(len(all_windows_sorted))],\n                    "window_end": [(i * self.config["step_size"]) + self.config["window_size"] for i in range(len(all_windows_sorted))]\n                })\n                \n                # 合并所有原始数据\n                raw_data_df = pd.concat(all_raw_data, ignore_index=True)\n                \n                # 调用行为发现模块\n                try:\n                    behavior_results = self.pattern_identifier.identify(features_df, raw_data_df)\n                    behavior_labels = behavior_results["labels"]\n                    extra_assets["behavior_stats"] = behavior_results["behavior_stats"]\n                except Exception as e:\n                    print(f"行为发现模块执行失败，使用默认标签: {e}")\n                    # 如果行为发现失败，使用默认标签\n                    behavior_labels = [0] * len(all_windows_sorted)\n            else:\n                # 如果没有窗口，使用空标签列表\n                behavior_labels = []\n        else:\n            # 如果没有数据，使用空标签列表\n            behavior_labels = []\n            all_windows_sorted = []\n        \n        # 分配行为标签到各个数据集\n        label_idx = 0\n        # 建立窗口到标签的映射\n        window_to_label = {}\n        for i, window in enumerate(all_windows_sorted):\n            # 使用唯一标识作为键，这里使用trace_id和start_time的组合\n            window_key = f"{window['trace_id']}_{window['start_time']}"\n            window_to_label[window_key] = behavior_labels[i] if i < len(behavior_labels) else 0\n        \n        # 处理所有数据集\n        for dataset_name, dataset in datasets.items():\n            # 按 trace_id 分组并排序\n            traces = group_by_trace_id(dataset)\n            \n            # 对每个trace内的窗口按 start_time 排序\n            for trace_id, metas in traces.items():\n                traces[trace_id] = sorted(metas, key=lambda x: x["start_time"])\n            \n            # 处理每个窗口\n            normed_metas = []\n            dataset_state_ids = []\n            \n            for trace_id, metas in traces.items():\n                for i, meta in enumerate(metas):\n                    # 使用唯一标识查找对应的行为标签\n                    window_key = f"{meta['trace_id']}_{meta['start_time']}"\n                    network_state_id = window_to_label.get(window_key, 0)"))
    
    # 保存修复后的文件
    with open(pipeline_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("\n=== 修复完成 ===")
    print("已修复pipeline.py中的标签分配问题")
    print("修复内容：")
    print("1. 确保生成行为标签时使用与最终处理相同的窗口顺序")
    print("2. 建立窗口到标签的映射关系")
    print("3. 使用唯一标识查找对应的行为标签")
    print("4. 避免标签与窗口顺序错乱")
    
    print("\n=== 建议后续操作 ===")
    print("1. 重新运行step1_preprocess.py")
    print("2. 生成新的可视化图表")
    print("3. 验证样本分类是否正确")


if __name__ == "__main__":
    main()
