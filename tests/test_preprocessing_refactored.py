import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.preprocessing.data import WindowMetaRaw, WindowMetaRenamed, WindowMetaNormed
from src.preprocessing.processing import (
    parse_txt, clean_and_truncate, split_and_resample, extract_windows,
    mark_first_window, group_by_trace_id, assign_split_by_trace,
    fit_and_normalize
)
from src.preprocessing.features import (
    compute_window_features, compute_local_features, merge_features, normalize_condition_vector
)


class TestDataStructures:
    """测试数据结构定义"""
    
    def test_window_meta_raw(self):
        """测试WindowMetaRaw类型"""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "delay_up": [0.1, 0.2, 0.3],
            "delay_down": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0]
        })
        
        meta: WindowMetaRaw = {
            "window": df,
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "test_trace"
        }
        
        assert isinstance(meta, dict)
        assert all(key in meta for key in ["window", "is_first", "start_time", "trace_id"])
        assert isinstance(meta["window"], pd.DataFrame)
        assert isinstance(meta["is_first"], bool)
        assert isinstance(meta["start_time"], float)
        assert isinstance(meta["trace_id"], str)
    
    def test_window_meta_renamed(self):
        """测试WindowMetaRenamed类型"""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "del_up": [0.1, 0.2, 0.3],
            "del_dn": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_dn": [0.0, 0.0, 0.0]
        })
        
        meta: WindowMetaRenamed = {
            "window": df,
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "test_trace"
        }
        
        assert isinstance(meta, dict)
        assert all(key in meta for key in ["window", "is_first", "start_time", "trace_id"])
    
    def test_window_meta_normed(self):
        """测试WindowMetaNormed类型"""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "del_up": [0.1, 0.2, 0.3],
            "del_dn": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_dn": [0.0, 0.0, 0.0]
        })
        
        meta: WindowMetaNormed = {
            "window": df,
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "test_trace",
            "cond": np.zeros(23),
            "keep": False
        }
        
        assert isinstance(meta, dict)
        assert all(key in meta for key in ["window", "is_first", "start_time", "trace_id", "cond", "keep"])
        assert isinstance(meta["cond"], np.ndarray)
        assert meta["cond"].shape == (23,)
        assert isinstance(meta["keep"], bool)


class TestProcessingFunctions:
    """测试数据处理函数"""
    
    def test_compute_window_features(self):
        """测试窗口特征计算"""
        df = pd.DataFrame({
            "delay_up": [1.0, 2.0, 3.0, 4.0, 5.0],
            "delay_down": [5.0, 4.0, 3.0, 2.0, 1.0],
            "loss_up": [0.0, 0.0, 0.0, 0.0, 0.0],
            "loss_dn": [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        features = compute_window_features(df)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (13,)
        assert features[0] == 3.0  # mean_delay_up
        assert features[4] == 3.0  # mean_delay_down
    
    def test_compute_local_features(self):
        """测试局部特征计算"""
        df = pd.DataFrame({
            "delay_up": [1.0, 2.0, 3.0, 4.0, 5.0],
            "delay_down": [5.0, 4.0, 3.0, 2.0, 1.0],
            "loss_up": [0.0, 0.0, 0.0, 0.0, 0.0],
            "loss_dn": [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        features = compute_local_features(df)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (10,)
        assert np.all(features == 0.0)  # 当前实现返回零向量
    
    def test_merge_features(self):
        """测试特征合并"""
        global_features = np.zeros(13)
        local_features = np.ones(10)
        
        merged = merge_features(global_features, local_features)
        
        assert isinstance(merged, np.ndarray)
        assert merged.shape == (23,)
        assert np.all(merged[:13] == 0.0)
        assert np.all(merged[13:] == 1.0)
    
    def test_normalize_condition_vector(self):
        """测试条件向量标准化"""
        cond_vector = np.zeros(23)
        cond_vector[11] = 1.0  # 网络状态ID
        
        cond_mean = np.zeros(22)
        cond_std = np.ones(22)
        
        normalized = normalize_condition_vector(cond_vector, cond_mean, cond_std)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == (23,)
        assert normalized[11] == 1.0  # 网络状态ID保持不变
    
    def test_mark_first_window(self):
        """测试标记首个窗口"""
        df1 = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "delay_up": [0.1, 0.2, 0.3],
            "delay_down": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0]
        })
        df2 = pd.DataFrame({
            "timestamp": [4, 5, 6],
            "delay_up": [0.1, 0.2, 0.3],
            "delay_down": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0]
        })
        
        windows = [df1, df2]
        result = mark_first_window(windows, "test_trace")
        
        assert len(result) == 2
        assert result[0]["is_first"] == True
        assert result[1]["is_first"] == False
    
    def test_group_by_trace_id(self):
        """测试按trace_id分组"""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "delay_up": [0.1, 0.2, 0.3],
            "delay_down": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0]
        })
        
        meta1 = {
            "window": df.copy(),
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "trace1"
        }
        meta2 = {
            "window": df.copy(),
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "trace2"
        }
        
        result = group_by_trace_id([meta1, meta2])
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "trace1" in result
        assert "trace2" in result
    
    def test_assign_split_by_trace(self):
        """测试按trace分配数据集"""
        df = pd.DataFrame({
            "timestamp": [1, 2, 3],
            "delay_up": [0.1, 0.2, 0.3],
            "delay_down": [0.4, 0.5, 0.6],
            "loss_up": [0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0]
        })
        
        meta1 = {
            "window": df.copy(),
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "trace1"
        }
        meta2 = {
            "window": df.copy(),
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "trace2"
        }
        meta3 = {
            "window": df.copy(),
            "is_first": True,
            "start_time": 1.0,
            "trace_id": "trace3"
        }
        
        traces = {
            "trace1": [meta1],
            "trace2": [meta2],
            "trace3": [meta3]
        }
        
        train, val, test = assign_split_by_trace(traces, train_ratio=0.6, val_ratio=0.2, random_state=42)
        
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)
        assert len(train) + len(val) + len(test) == 3


class TestPipelineIntegration:
    """测试流水线集成"""
    
    def test_basic_pipeline(self):
        """测试基本流水线流程"""
        # 创建测试数据
        df = pd.DataFrame({
            "timestamp": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            "delay_up_origin": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "delay_down_origin": [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "loss_up_origin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "loss_down_origin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "delay_up": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "delay_down": [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "loss_up": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "loss_down": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        # 测试分割和重采样
        segments = split_and_resample(df, max_gap_sec=0.5, min_rows=3, freq_hz=10)
        assert len(segments) > 0
        
        # 测试提取窗口
        windows = extract_windows(segments, window=5, step=2)
        assert len(windows) > 0
        
        # 测试标记首个窗口
        window_metas = mark_first_window(windows, "test_trace")
        assert len(window_metas) > 0
        assert window_metas[0]["is_first"] == True
        
        # 测试分组
        grouped = group_by_trace_id(window_metas)
        assert len(grouped) == 1
        
        # 测试数据集分配
        train, val, test = assign_split_by_trace(grouped, train_ratio=0.8, val_ratio=0.1, random_state=42)
        assert len(train) + len(val) + len(test) == len(window_metas)
