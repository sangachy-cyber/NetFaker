#!/usr/bin/env python3
"""
使用10D特征进行聚类，生成state_id，并统计train/test/val集的分布
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.preprocessing import (
    stage1_parse_txt, stage2_clean_and_truncate, stage3_split_and_resample,
    stage4_extract_windows, stage5_mark_first_window, stage6_group_by_trace_id,
    stage7_assign_split_by_trace, stage8_fit_and_normalize, 
    stage9_recompute_condition_vectors, stage10_save_artifacts
)


def main():
    """主函数"""
    print("=== 10D特征聚类分析 ===")
    
    # 1. 加载配置
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # 2. 设置输出目录
    out_dir = Path("output/10d_clustering_result")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    # 3. 加载原始数据
    print("\n加载原始数据...")
    txt_files = list(Path(cfg["input_dir"]).glob("*.txt"))
    all_windows_meta = []
    
    for txt_file in txt_files:
        print(f"处理文件: {txt_file.name}")
        df_raw = stage1_parse_txt(txt_file, cfg["raw_interval_sec"])
        df_clean = stage2_clean_and_truncate(df_raw, cfg["max_delay_ms"])
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=cfg["max_gap_sec"],
            min_rows=cfg["min_segment_rows"],
            max_invalid_ratio=cfg.get("max_invalid_ratio", 0.01),
        )
        
        for seg in segments:
            seg_windows = stage4_extract_windows([seg], cfg["window_size"], cfg["step_size"])
            trace_id = txt_file.stem
            window_metas = stage5_mark_first_window(seg_windows, trace_id)
            all_windows_meta.extend(window_metas)
    
    # 4. 按trace_id分组
    traces_dict = stage6_group_by_trace_id(all_windows_meta)
    
    # 5. 划分数据集
    print("\n划分数据集...")
    train, val, test = stage7_assign_split_by_trace(
        traces_dict,
        train_ratio=cfg["split"]["train"],
        val_ratio=cfg["split"]["val"],
        random_state=cfg["split"]["random_state"],
    )
    
    # 6. 归一化数据
    print("\n归一化数据...")
    renamed_datasets, assets = stage8_fit_and_normalize(
        train, val, test, random_state=cfg["quantile_transformer"]["random_state"],
    )
    
    # 7. 使用10D特征进行聚类并生成state_id
    print("\n使用10D特征进行聚类并生成state_id...")
    final_datasets, extra_assets = stage9_recompute_condition_vectors(
        renamed_datasets, assets, {},  # 传递空的network_state_map，让函数自动聚类
    )
    
    # 8. 更新assets字典
    assets.update(extra_assets)
    
    # 9. 保存结果
    print("\n保存结果...")
    stage10_save_artifacts(
        final_datasets, assets, out_dir, dtype=np.float32,
    )
    
    # 10. 打印state_id分布统计
    if "state_id_stats" in assets:
        print("\n=== 网络状态ID分布统计 ===")
        state_id_stats = assets["state_id_stats"]
        total_samples = 0
        
        for dataset_name, stats in state_id_stats.items():
            print(f"\n{dataset_name.upper()}集:")
            dataset_samples = sum(stat["count"] for stat in stats.values())
            total_samples += dataset_samples
            print(f"  样本数: {dataset_samples}")
            print(f"  状态ID分布:")
            for state_id in sorted(stats.keys()):
                stat = stats[state_id]
                print(f"    状态ID {state_id}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
        
        print(f"\n总样本数: {total_samples}")
    
    # 11. 保存state_id分布到单独的文件
    if "state_id_stats" in assets:
        state_id_stats = assets["state_id_stats"]
        with open(out_dir / "state_id_distribution.json", "w", encoding="utf-8") as f:
            json.dump(state_id_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！结果保存到: {out_dir}")


if __name__ == "__main__":
    main()
