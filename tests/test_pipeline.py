#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试预处理管道的各个阶段
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    stage1_parse_txt,
    stage2_clean_and_truncate,
    stage3_split_and_resample,
    stage4_extract_windows,
    stage5_mark_first_window
)
import yaml

def test_stage1():
    """测试阶段1: 解析txt文件"""
    print("测试阶段1: 解析txt文件")
    # 使用一个实际的数据文件进行测试
    data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
    if not data_file.exists():
        print(f"数据文件 {data_file} 不存在")
        return False
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        df = stage1_parse_txt(data_file, config["raw_interval_sec"])
        print(f"成功解析文件，得到 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        print(f"前5行数据:")
        print(df.head())
        return True
    except Exception as e:
        print(f"阶段1测试失败: {e}")
        return False

def test_stage2():
    """测试阶段2: 清洗和截断数据"""
    print("\n测试阶段2: 清洗和截断数据")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
        df_raw = stage1_parse_txt(data_file, config["raw_interval_sec"])
        df_clean = stage2_clean_and_truncate(df_raw, config["max_delay_ms"])
        print(f"清洗后剩余 {len(df_clean)} 行数据")
        return True
    except Exception as e:
        print(f"阶段2测试失败: {e}")
        return False

def test_stage3():
    """测试阶段3: 分割和重采样"""
    print("\n测试阶段3: 分割和重采样")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
        df_raw = stage1_parse_txt(data_file, config["raw_interval_sec"])
        df_clean = stage2_clean_and_truncate(df_raw, config["max_delay_ms"])
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=config["max_gap_sec"],
            min_rows=config["min_segment_rows"],
            max_invalid_ratio=config.get("max_invalid_ratio", 0.01)
        )
        print(f"分割得到 {len(segments)} 个段")
        if segments:
            print(f"第一个段有 {len(segments[0])} 行数据")
        return True
    except Exception as e:
        print(f"阶段3测试失败: {e}")
        return False

def test_stage4():
    """测试阶段4: 提取窗口"""
    print("\n测试阶段4: 提取窗口")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
        df_raw = stage1_parse_txt(data_file, config["raw_interval_sec"])
        df_clean = stage2_clean_and_truncate(df_raw, config["max_delay_ms"])
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=config["max_gap_sec"],
            min_rows=config["min_segment_rows"],
            max_invalid_ratio=config.get("max_invalid_ratio", 0.01)
        )
        windows = stage4_extract_windows(segments, config["window_size"], config["step_size"])
        print(f"提取得到 {len(windows)} 个窗口")
        if windows:
            print(f"第一个窗口有 {len(windows[0])} 行数据")
        return True
    except Exception as e:
        print(f"阶段4测试失败: {e}")
        return False

def test_stage5():
    """测试阶段5: 标记首个窗口"""
    print("\n测试阶段5: 标记首个窗口")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
        df_raw = stage1_parse_txt(data_file, config["raw_interval_sec"])
        df_clean = stage2_clean_and_truncate(df_raw, config["max_delay_ms"])
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=config["max_gap_sec"],
            min_rows=config["min_segment_rows"],
            max_invalid_ratio=config.get("max_invalid_ratio", 0.01)
        )
        windows = stage4_extract_windows(segments, config["window_size"], config["step_size"])
        window_metas = stage5_mark_first_window(windows, data_file.stem)
        print(f"标记得到 {len(window_metas)} 个窗口元数据")
        if window_metas:
            print(f"第一个窗口是否为首窗口: {window_metas[0]['is_first']}")
            print(f"窗口起始时间: {window_metas[0]['start_time']}")
        return True
    except Exception as e:
        print(f"阶段5测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试预处理管道...")
    
    tests = [
        test_stage1,
        test_stage2,
        test_stage3,
        test_stage4,
        test_stage5
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("所有测试都通过了!")
        return 0
    else:
        print("部分测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())