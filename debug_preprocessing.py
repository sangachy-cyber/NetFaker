#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import *

def debug_stage1():
    print("=== Debug Stage 1 ===")
    txt_path = Path("data/20251204_222027_ZGp-playback.txt")
    
    # 读取配置获取正确的interval_sec
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    df = stage1_parse_txt(txt_path, cfg["raw_interval_sec"])
    print(f"Parsed DataFrame shape: {df.shape}")
    print(df.head())
    return df

def debug_stage2(df):
    print("\n=== Debug Stage 2 ===")
    df_clean = stage2_clean_and_truncate(df, 2000)
    print(f"Cleaned DataFrame shape: {df_clean.shape}")
    print(df_clean.head())
    return df_clean

def debug_stage3(df):
    print("\n=== Debug Stage 3 ===")
    segments = stage3_split_and_resample(
        df,
        max_gap_sec=2.0,
        min_rows=200,
        max_invalid_ratio=0.01
    )
    print(f"Number of segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"Segment {i} shape: {seg.shape}")
        if i >= 2:  # 只显示前几个segment
            break
    return segments

def debug_stage4(segments):
    print("\n=== Debug Stage 4 ===")
    windows = []
    for seg in segments[:1]:  # 只处理第一个segment以节省时间
        seg_windows = stage4_extract_windows([seg], 100, 50)
        windows.extend(seg_windows)
        print(f"Segment produced {len(seg_windows)} windows")
    print(f"Total windows: {len(windows)}")
    if windows:
        print(f"First window shape: {windows[0].shape}")
    return windows

def debug_stage5(windows):
    print("\n=== Debug Stage 5 ===")
    window_metas = stage5_mark_first_window(windows, "test_trace")
    print(f"Number of window metas: {len(window_metas)}")
    if window_metas:
        print(f"First meta is_first: {window_metas[0]['is_first']}")
        print(f"First meta start_time: {window_metas[0]['start_time']}")
    return window_metas

if __name__ == "__main__":
    df = debug_stage1()
    df_clean = debug_stage2(df)
    segments = debug_stage3(df_clean)
    windows = debug_stage4(segments)
    window_metas = debug_stage5(windows)
    print("\nDebug completed.")