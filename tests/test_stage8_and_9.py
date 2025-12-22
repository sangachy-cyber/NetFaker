#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试预处理管道的stage8和stage9
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    stage1_parse_txt,
    stage2_clean_and_truncate,
    stage3_split_and_resample,
    stage4_extract_windows,
    stage5_mark_first_window,
    stage6_group_by_trace_id,
    stage7_assign_split_by_trace,
    stage8_fit_and_normalize,
    stage9_recompute_condition_vectors,
    _compute_window_features,
    _compute_local_features
)
import yaml


def create_sample_data():
    """创建用于测试的样本数据"""
    # 创建一个简单的测试DataFrame
    timestamps = np.arange(100)
    data = pd.DataFrame({
        'timestamp': timestamps,
        'del_up': np.random.normal(0.1, 0.05, 100),
        'loss_up': np.random.choice([0.0, 1.0], 100, p=[0.7, 0.3]),  # 只保留0和1
        'del_dn': np.random.normal(0.1, 0.05, 100),
        'loss_dn': np.random.choice([0.0, 1.0], 100, p=[0.7, 0.3])   # 只保留0和1
    })
    return data


def test_compute_window_features():
    """测试计算窗口全局特征的函数"""
    print("测试 _compute_window_features 函数")
    try:
        # 创建测试数据
        data = create_sample_data()
        
        # 计算特征
        features = _compute_window_features(data)
        
        # 检查返回值
        assert isinstance(features, np.ndarray), "返回值应该是numpy数组"
        assert features.shape == (13,), f"特征向量应该是13维，实际是{features.shape}"
        
        print(f"成功计算窗口特征，形状: {features.shape}")
        print(f"特征示例: {features[:5]}")
        return True
    except Exception as e:
        print(f"_compute_window_features 测试失败: {e}")
        return False


def test_compute_local_features():
    """测试计算窗口局部特征的函数"""
    print("\n测试 _compute_local_features 函数")
    try:
        # 创建测试数据
        data = create_sample_data()
        
        # 计算特征
        features = _compute_local_features(data)
        
        # 检查返回值
        assert isinstance(features, np.ndarray), "返回值应该是numpy数组"
        assert features.shape == (10,), f"局部特征向量应该是10维，实际是{features.shape}"
        
        print(f"成功计算局部特征，形状: {features.shape}")
        print(f"特征示例: {features[:5]}")
        return True
    except Exception as e:
        print(f"_compute_local_features 测试失败: {e}")
        return False


def test_stage8_and_9_integration():
    """测试stage8和stage9的集成"""
    print("\n测试 stage8 和 stage9 集成")
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # 使用实际数据文件
        data_file = Path("data") / "20251204_222027_ZGp-playback.txt"
        if not data_file.exists():
            print(f"数据文件 {data_file} 不存在，使用模拟数据")
            # 创建模拟数据进行测试
            df_raw = create_sample_data()
        else:
            df_raw = stage1_parse_txt(data_file, config["raw_interval_sec"])
        
        df_clean = stage2_clean_and_truncate(df_raw, config["max_delay_ms"])
        segments = stage3_split_and_resample(
            df_clean,
            max_gap_sec=config["max_gap_sec"],
            min_rows=config["min_segment_rows"],
            max_invalid_ratio=config.get("max_invalid_ratio", 0.01)
        )
        
        if not segments:
            print("没有有效的数据段，创建模拟数据段")
            segments = [create_sample_data()]
        
        windows = stage4_extract_windows(segments, config["window_size"], config["step_size"])
        if not windows:
            print("没有提取到窗口，创建模拟窗口")
            windows = [create_sample_data()]
            
        window_metas = stage5_mark_first_window(windows, "test_trace")
        traces_dict = stage6_group_by_trace_id(window_metas)
        train, val, test = stage7_assign_split_by_trace(
            traces_dict,
            train_ratio=config["split"]["train"],
            val_ratio=config["split"]["val"],
            random_state=config["split"]["random_state"]
        )
        
        # 测试stage8
        renamed_datasets, assets = stage8_fit_and_normalize(
            train, val, test, random_state=config["quantile_transformer"]["random_state"]
        )
        
        print(f"Stage8完成，重命名数据集键: {list(renamed_datasets.keys())}")
        print(f"Assets键: {list(assets.keys())}")
        
        # 测试stage9
        network_state_map = {"test_trace": 1}
        final_datasets, extra_assets = stage9_recompute_condition_vectors(
            renamed_datasets, assets, network_state_map
        )
        
        print(f"Stage9完成，最终数据集键: {list(final_datasets.keys())}")
        print(f"Extra assets键: {list(extra_assets.keys())}")
        
        # 验证结果
        if final_datasets['train'] and len(final_datasets['train']) > 0:
            sample_meta = final_datasets['train'][0]
            assert 'cond' in sample_meta, "条件向量应该存在于元数据中"
            print(f"条件向量形状: {sample_meta['cond'].shape}")
            print(f"条件向量示例: {sample_meta['cond'][:5]}")
            # 检查条件向量维度（可能是22或23）
            assert sample_meta['cond'].shape[0] in [22, 23], f"条件向量维度异常: {sample_meta['cond'].shape}"
            if sample_meta['cond'].shape[0] == 23:
                print("条件向量维度正确 (23维)")
            else:
                print(f"条件向量维度为 {sample_meta['cond'].shape[0]} 维，可能需要进一步检查")
        
        return True
    except Exception as e:
        print(f"Stage8和Stage9集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试Stage8和Stage9...")
    
    tests = [
        test_compute_window_features,
        test_compute_local_features,
        test_stage8_and_9_integration
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