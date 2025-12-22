#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import *

def debug_full_pipeline():
    print("=== Debug Full Pipeline ===")
    
    # 读取配置
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    print("Configuration loaded")
    
    # 处理单个文件进行测试
    txt_file = Path("data/20251204_222027_ZGp-playback.txt")
    
    try:
        print(f"\nProcessing file: {txt_file.name}")
        
        # 阶段1: 解析
        df_raw = stage1_parse_txt(txt_file, cfg["raw_interval_sec"])
        print(f"Stage 1 - Raw DataFrame shape: {df_raw.shape}")
        
        # 阶段2: 清洗
        df_clean = stage2_clean_and_truncate(df_raw, cfg["max_delay_ms"])
        print(f"Stage 2 - Cleaned DataFrame shape: {df_clean.shape}")
        
        # 阶段3: 分割和重采样
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=cfg["max_gap_sec"],
            min_rows=cfg["min_segment_rows"],
            max_invalid_ratio=cfg.get("max_invalid_ratio", 0.01)
        )
        print(f"Stage 3 - Number of segments: {len(segments)}")
        if segments:
            print(f"First segment shape: {segments[0].shape}")
        
        # 阶段4: 提取窗口
        windows = []
        for seg in segments:
            seg_windows = stage4_extract_windows([seg], cfg["window_size"], cfg["step_size"])
            windows.extend(seg_windows)
        
        print(f"Stage 4 - Number of windows: {len(windows)}")
        if windows:
            print(f"First window shape: {windows[0].shape}")
        
        # 阶段5: 标记首窗
        trace_id = txt_file.stem
        window_metas = stage5_mark_first_window(windows, trace_id)
        print(f"Stage 5 - Number of window metas: {len(window_metas)}")
        if window_metas:
            first_is_first = window_metas[0]['is_first']
            print(f"First window is_first: {first_is_first}")
            print(f"Last window is_first: {window_metas[-1]['is_first']}")
        
        # 阶段6: 按trace分组
        traces_dict = stage6_group_by_trace_id(window_metas)
        print(f"Stage 6 - Number of traces: {len(traces_dict)}")
        print(f"Trace IDs: {list(traces_dict.keys())}")
        
        # 阶段7: 分配到训练/验证/测试集
        train, val, test = stage7_assign_split_by_trace(
            traces_dict,
            train_ratio=cfg["split"]["train"],
            val_ratio=cfg["split"]["val"],
            random_state=cfg["split"]["random_state"]
        )
        print(f"Stage 7 - Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
        
        # 检查训练集
        print(f"Train metas sample:")
        for i, meta in enumerate(train[:3]):
            print(f"  Meta {i}: is_first={meta['is_first']}, trace_id={meta['trace_id']}")
        
        # 阶段8: 拟合和标准化
        print("\nAttempting Stage 8...")
        renamed_datasets, assets = stage8_fit_and_normalize(
            train, val, test, random_state=cfg["quantile_transformer"]["random_state"]
        )
        print("Stage 8 completed successfully!")
        print(f"Renamed datasets keys: {list(renamed_datasets.keys())}")
        print(f"Train renamed size: {len(renamed_datasets['train'])}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_pipeline()