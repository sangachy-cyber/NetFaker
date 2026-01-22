"""窗口采样器测试模块。

测试窗口采样器的窗口加载和抽样功能。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from netfaker.simcore.synthesizer.window_sampler import WindowSampler


class TestWindowSampler:
    """测试窗口采样器类。
    """

    @pytest.fixture
    def mock_window_data(self):
        """创建模拟窗口数据。
        """
        # 创建模拟数据
        n_windows = 10
        data = {
            "state_id": np.random.choice([0, 1, 2], size=n_windows),
            "raw_delay_up": [np.random.rand(100).tolist() for _ in range(n_windows)],
            "raw_delay_down": [np.random.rand(100).tolist() for _ in range(n_windows)],
            "raw_loss_up": [[int(x < 0.1) for x in np.random.rand(100)] for _ in range(n_windows)],
            "raw_loss_down": [[int(x < 0.1) for x in np.random.rand(100)] for _ in range(n_windows)],
            "raw_bw_up": [np.random.rand(100).tolist() for _ in range(n_windows)],
            "raw_bw_down": [np.random.rand(100).tolist() for _ in range(n_windows)]
        }
        return pd.DataFrame(data)

    def test_init_valid(self, mock_window_data):
        """测试初始化有效窗口数据。
        """
        with patch("pandas.read_parquet", return_value=mock_window_data):
            sampler = WindowSampler("dummy_path.parquet")
            assert len(sampler.state_to_windows) > 0

    def test_init_invalid_path(self):
        """测试初始化无效路径。
        """
        with patch("pandas.read_parquet", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                WindowSampler("invalid_path.parquet")

    def test_sample_windows_valid(self, mock_window_data):
        """测试有效窗口抽样。
        """
        with patch("pandas.read_parquet", return_value=mock_window_data):
            sampler = WindowSampler("dummy_path.parquet")
            state_id = 0
            n_windows = 2
            
            # 确保至少有一个窗口对应state_id=0
            if state_id not in sampler.state_to_windows:
                # 修改模拟数据，确保有state_id=0的窗口
                mock_window_data.loc[0, "state_id"] = state_id
                with patch("pandas.read_parquet", return_value=mock_window_data):
                    sampler = WindowSampler("dummy_path.parquet")
            
            sampled_windows = sampler.sample_windows(state_id, n_windows, seed=42)
            assert len(sampled_windows) == n_windows

    def test_sample_windows_invalid_state(self, mock_window_data):
        """测试无效状态抽样。
        """
        with patch("pandas.read_parquet", return_value=mock_window_data):
            sampler = WindowSampler("dummy_path.parquet")
            with pytest.raises(ValueError, match="没有可用的窗口对应state_id=999"):
                sampler.sample_windows(999, 1, seed=42)
