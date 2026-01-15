import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    stage1_parse_txt,
    stage2_clean_and_truncate,
    stage3_split_and_resample,
    stage4_extract_windows,
    stage5_mark_first_window,
    stage6_group_by_trace_id,
    stage7_assign_split_by_trace,
    stage8_fit_and_normalize,
    stage9_recompute_condition_vectors,
    stage10_save_artifacts,
    stage11_generate_report,
    WindowMetaRaw,
    WindowMetaRenamed,
    WindowMetaNormed,
)


def test_stage1_parse_txt():
    """测试解析原始文本文件"""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Start Time: 2025-01-01 00:00:00\n")
        f.write("------------------------------------------------\n")
        f.write("10.0,0.0,100.0,15.0,0.0,100.0\n")
        f.write("20.0,0.0,100.0,25.0,0.0,100.0\n")
        f.write("30.0,0.0,100.0,35.0,0.0,100.0\n")
    
    try:
        # 测试正常情况
        df = stage1_parse_txt(Path(f.name), 0.1)
        assert df.shape == (3, 5)
        assert list(df.columns) == [
            'timestamp', 'delay_up', 'delay_down', 'loss_up', 'loss_down'
        ]
        
        # 测试数据值
        assert np.isclose(df['delay_up'].iloc[0], 10.0)
        assert np.isclose(df['delay_down'].iloc[0], 15.0)
        assert np.isclose(df['loss_up'].iloc[0], 0.0)
        assert np.isclose(df['loss_down'].iloc[0], 0.0)
    finally:
        # 清理临时文件
        os.unlink(f.name)


def test_stage2_clean_and_truncate():
    """测试清洗并截断数据"""
    # 创建测试数据
    df = pd.DataFrame({
        'timestamp': [1, 2, 3, 4, 5],
        'delay_up': [10, 20, 3000, 40, 50],
        'delay_down': [15, 25, 35, 45, 55],
        'loss_up': [0.0, 0.0, 0.0, 0.0, 0.0],
        'loss_down': [0.0, 0.0, 0.0, 0.0, 0.0]
    })

    # 测试截断功能
    df_clean = stage2_clean_and_truncate(df, max_delay_ms=2000)
    assert df_clean.shape == (2, 5)  # 应该截断到第2行，因为第3行delay_up=3000 > 2000
    
    # 测试清洗功能（移除NaN）
    df_with_nan = df.copy()
    df_with_nan.loc[1, 'delay_up'] = np.nan
    df_clean = stage2_clean_and_truncate(df_with_nan, max_delay_ms=2000)
    assert df_clean.shape == (1, 5)  # 应该移除第1行（包含NaN）


def test_stage3_split_and_resample():
    """测试分割并重新采样数据"""
    # 创建测试数据，包含一个间隙
    df = pd.DataFrame({
        'timestamp': [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3],
        'delay_up': [10, 20, 30, 40, 50, 60, 70, 80],
        'delay_down': [15, 25, 35, 45, 55, 65, 75, 85],
        'loss_up': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'loss_down': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    # 测试分割和重采样
    segments = stage3_split_and_resample(
        df, 
        max_gap_sec=0.5,  # 间隙超过0.5秒会被分割
        min_rows=3,
        freq_hz=10
    )
    
    assert len(segments) == 2  # 应该被分割成两段
    assert segments[0].shape[0] >= 3
    assert segments[1].shape[0] >= 3
    
    # 测试重采样后的数据没有NaN
    for segment in segments:
        assert not segment.isnull().values.any()


def test_stage4_extract_windows():
    """测试提取窗口"""
    # 创建测试数据段
    segment = pd.DataFrame({
        'timestamp': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'delay_up': [10, 20, 30, 40, 50, 60],
        'delay_down': [15, 25, 35, 45, 55, 65],
        'loss_up': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'loss_down': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    # 测试提取窗口
    windows = stage4_extract_windows([segment], window=3, step=2)
    assert len(windows) == 2  # 应该提取2个窗口
    assert windows[0].shape == (3, 5)
    assert windows[1].shape == (3, 5)


def test_stage5_mark_first_window():
    """测试标记首个窗口"""
    # 创建测试窗口
    windows = [
        pd.DataFrame({
            'timestamp': [1.0, 1.1, 1.2],
            'delay_up': [10, 20, 30],
            'delay_down': [15, 25, 35],
            'loss_up': [0.0, 0.0, 0.0],
            'loss_down': [0.0, 0.0, 0.0]
        }),
        pd.DataFrame({
            'timestamp': [1.2, 1.3, 1.4],
            'delay_up': [30, 40, 50],
            'delay_down': [35, 45, 55],
            'loss_up': [0.0, 0.0, 0.0],
            'loss_down': [0.0, 0.0, 0.0]
        })
    ]
    
    # 测试标记首个窗口
    meta_windows = stage5_mark_first_window(windows, "test_trace")
    assert len(meta_windows) == 2
    assert meta_windows[0]['is_first'] == True
    assert meta_windows[1]['is_first'] == False
    assert meta_windows[0]['trace_id'] == "test_trace"
    assert meta_windows[1]['trace_id'] == "test_trace"


def test_stage6_group_by_trace_id():
    """测试按 trace_id 分组"""
    # 创建测试元数据
    meta1 = WindowMetaRaw(
        window=pd.DataFrame(),
        is_first=True,
        start_time=1.0,
        trace_id="trace1"
    )
    meta2 = WindowMetaRaw(
        window=pd.DataFrame(),
        is_first=False,
        start_time=1.1,
        trace_id="trace1"
    )
    meta3 = WindowMetaRaw(
        window=pd.DataFrame(),
        is_first=True,
        start_time=1.0,
        trace_id="trace2"
    )
    
    all_metas = [meta1, meta2, meta3]
    
    # 测试分组
    grouped = stage6_group_by_trace_id(all_metas)
    assert len(grouped) == 2
    assert len(grouped["trace1"]) == 2
    assert len(grouped["trace2"]) == 1


def test_stage7_assign_split_by_trace():
    """测试分配训练/验证/测试集"""
    # 创建测试数据
    traces = {
        "trace1": [WindowMetaRaw(window=pd.DataFrame(), is_first=True, start_time=1.0, trace_id="trace1")],
        "trace2": [WindowMetaRaw(window=pd.DataFrame(), is_first=True, start_time=1.0, trace_id="trace2")],
        "trace3": [WindowMetaRaw(window=pd.DataFrame(), is_first=True, start_time=1.0, trace_id="trace3")],
        "trace4": [WindowMetaRaw(window=pd.DataFrame(), is_first=True, start_time=1.0, trace_id="trace4")],
        "trace5": [WindowMetaRaw(window=pd.DataFrame(), is_first=True, start_time=1.0, trace_id="trace5")],
    }
    
    # 测试分配
    train, val, test = stage7_assign_split_by_trace(traces, train_ratio=0.6, val_ratio=0.2, random_state=42)
    
    # 验证分配比例
    assert len(train) == 3  # 60% of 5 = 3
    assert len(val) == 1     # 20% of 5 = 1
    assert len(test) == 1    # 20% of 5 = 1


def test_stage8_fit_and_normalize():
    """测试拟合并标准化数据"""
    # 创建测试数据
    window1 = pd.DataFrame({
        'timestamp': [1.0, 1.1, 1.2],
        'delay_up': [10, 20, 30],
        'delay_down': [15, 25, 35],
        'loss_up': [0.0, 0.0, 0.0],
        'loss_down': [0.0, 0.0, 0.0]
    })
    
    train_metas = [
        WindowMetaRaw(window=window1, is_first=True, start_time=1.0, trace_id="trace1"),
        WindowMetaRaw(window=window1.copy(), is_first=False, start_time=1.1, trace_id="trace1"),
    ]
    
    val_metas = [
        WindowMetaRaw(window=window1.copy(), is_first=True, start_time=1.0, trace_id="trace2"),
    ]
    
    test_metas = [
        WindowMetaRaw(window=window1.copy(), is_first=True, start_time=1.0, trace_id="trace3"),
    ]
    
    # 测试归一化
    renamed_datasets, assets = stage8_fit_and_normalize(train_metas, val_metas, test_metas)
    
    assert 'train' in renamed_datasets
    assert 'val' in renamed_datasets
    assert 'test' in renamed_datasets
    assert 'qt_up' in assets
    assert 'qt_down' in assets
    
    # 测试归一化后的数据
    assert len(renamed_datasets['train']) == 2
    # 检查字典结构，而不是使用isinstance检查TypedDict
    assert isinstance(renamed_datasets['train'][0], dict)
    assert 'window' in renamed_datasets['train'][0]
    assert 'is_first' in renamed_datasets['train'][0]
    assert 'start_time' in renamed_datasets['train'][0]
    assert 'trace_id' in renamed_datasets['train'][0]


@pytest.mark.skip(reason="stage9_recompute_condition_vectors已迁移到PreprocessingPipeline类")
def test_stage9_recompute_condition_vectors():
    """测试重新计算条件向量"""
    # 创建简单测试数据
    window1 = pd.DataFrame({
        'timestamp': [1.0, 1.1, 1.2],
        'delay_up': [10, 20, 30],
        'delay_down': [15, 25, 35],
        'loss_up': [0.0, 0.0, 0.0],
        'loss_dn': [0.0, 0.0, 0.0]
    })
    
    renamed_metas = [
        WindowMetaRenamed(
            window=window1,
            is_first=True,
            start_time=1.0,
            trace_id="trace1"
        ),
        WindowMetaRenamed(
            window=window1.copy(),
            is_first=False,
            start_time=1.1,
            trace_id="trace1"
        ),
    ]
    
    datasets = {
        'train': renamed_metas,
        'val': renamed_metas,
        'test': renamed_metas
    }
    
    assets = {
        'qt_up': None,  # 这里不需要实际的量化转换器，因为我们主要测试结构
        'qt_down': None
    }
    
    network_state_map = {}
    
    # 测试条件向量计算
    final_datasets, extra_assets = stage9_recompute_condition_vectors(datasets, assets, network_state_map)
    
    assert 'train' in final_datasets
    assert 'val' in final_datasets
    assert 'test' in final_datasets
    assert len(final_datasets['train']) == 2
    # 检查字典结构，而不是使用isinstance检查TypedDict
    assert isinstance(final_datasets['train'][0], dict)
    assert 'window' in final_datasets['train'][0]
    assert 'is_first' in final_datasets['train'][0]
    assert 'start_time' in final_datasets['train'][0]
    assert 'trace_id' in final_datasets['train'][0]
    assert 'cond' in final_datasets['train'][0]
    assert 'keep' in final_datasets['train'][0]
    assert final_datasets['train'][0]['cond'].shape == (23,)


@pytest.mark.skip(reason="stage10_save_artifacts已迁移到DataSaver类")
def test_stage10_save_artifacts():
    """测试保存处理后的数据和资源"""
    # 创建简单测试数据
    window1 = pd.DataFrame({
        'timestamp': [1.0, 1.1, 1.2],
        'delay_up': [0.1, 0.2, 0.3],
        'delay_down': [0.15, 0.25, 0.35],
        'loss_up': [0.0, 0.0, 0.0],
        'loss_dn': [0.0, 0.0, 0.0]
    })
    
    normed_metas = [
        WindowMetaNormed(
            window=window1,
            is_first=False,
            start_time=1.0,
            trace_id="trace1",
            cond=np.zeros(23),
            keep=True
        )
    ]
    
    datasets = {
        'train': normed_metas,
        'val': normed_metas,
        'test': normed_metas
    }
    
    assets = {
        'qt_up': None,
        'qt_down': None,
        'init_local_cond': np.zeros(10),
        'cond_mean': np.zeros(22),
        'cond_std': np.ones(22),
        'mean_loss_cat2_up': 0.5,
        'mean_loss_cat2_dn': 0.5,
        'state_id_stats': {}
    }
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试保存
        stage10_save_artifacts(datasets, assets, Path(tmpdir))
        
        # 验证输出文件存在
        assert (Path(tmpdir) / "assets").exists()
        assert (Path(tmpdir) / "meta").exists()
        assert (Path(tmpdir) / "datasets").exists()
        assert (Path(tmpdir) / "datasets" / "train.jsonl").exists()
        assert (Path(tmpdir) / "datasets" / "val.jsonl").exists()
        assert (Path(tmpdir) / "datasets" / "test.jsonl").exists()


@pytest.mark.skip(reason="stage11_generate_report已迁移到ReportGenerator类")
def test_stage11_generate_report():
    """测试生成数据报告"""
    temp_file_name = None
    
    try:
        # 创建临时模板文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_file_name = f.name
            f.write("# 数据报告\n\n总样本数: {{stats.total_windows}}\n")
        
        # 创建测试数据
        stats = {
            'total_windows': 1000,
            'train_count': 700,
            'val_count': 200,
            'test_count': 100,
            'processed_files': 10,
            'failed_files': 0,
            'empty_files': 0,
            'timestamp': "2025-01-01T00:00:00Z"
        }
        
        config = {
            "window_size": 100,
            "step_size": 50,
            "max_delay_ms": 2000
        }
        
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试报告生成
            stage11_generate_report(stats, config, Path(temp_file_name), Path(tmpdir) / "report.md")
            
            # 验证报告文件存在
            assert (Path(tmpdir) / "report.md").exists()
            
            # 验证报告内容
            with open(Path(tmpdir) / "report.md", 'r') as f:
                content = f.read()
                # 检查是否包含预期内容，使用in操作符来忽略可能的编码差异
                assert "总样本数" in content
                assert "1000" in content
    finally:
        # 清理临时文件
        if temp_file_name and os.path.exists(temp_file_name):
            os.unlink(temp_file_name)


def test_types_defined():
    """测试类型定义"""
    # 测试WindowMetaRaw类型
    raw_meta = WindowMetaRaw(
        window=pd.DataFrame(),
        is_first=True,
        start_time=1.0,
        trace_id="test"
    )
    assert isinstance(raw_meta, dict)
    assert "window" in raw_meta
    assert "is_first" in raw_meta
    assert "start_time" in raw_meta
    assert "trace_id" in raw_meta
    
    # 测试WindowMetaRenamed类型
    renamed_meta = WindowMetaRenamed(
        window=pd.DataFrame(),
        is_first=True,
        start_time=1.0,
        trace_id="test"
    )
    assert isinstance(renamed_meta, dict)
    
    # 测试WindowMetaNormed类型
    normed_meta = WindowMetaNormed(
        window=pd.DataFrame(),
        is_first=True,
        start_time=1.0,
        trace_id="test",
        cond=np.zeros(23),
        keep=True
    )
    assert isinstance(normed_meta, dict)
    assert "cond" in normed_meta
    assert "keep" in normed_meta
    assert normed_meta["cond"].shape == (23,)
