"""HoloWAN 窗口提取模块。

负责：
1. 从单个 Parquet 文件中提取滑动窗口
2. 基于延迟规则验证窗口
3. 计算 is_valid 标志
4. 返回窗口对象列表
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class WindowExtractor:
    """从单个文件中提取和验证滑动窗口的类。"""

    def __init__(self, window_size: int = 100, step_size: int = 50):
        """初始化窗口提取器。

        Args:
            window_size: 每个窗口的大小（行数）
            step_size: 滑动窗口的步长
        """
        self.window_size = window_size
        self.step_size = step_size

    def extract_windows(self, df: pd.DataFrame, source_file: str) -> List[Dict[str, Any]]:
        """从 DataFrame 中提取滑动窗口。

        Args:
            df: 输入 DataFrame
            source_file: 源文件名（用于元数据）

        Returns:
            包含验证状态的窗口字典列表

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'raw_delay_up': [100.0] * 150,
            ...     'raw_loss_up': [0.1] * 150,
            ...     'raw_bw_up': [10.0] * 150,
            ...     'raw_delay_down': [95.0] * 150,
            ...     'raw_loss_down': [0.0] * 150,
            ...     'raw_bw_down': [12.0] * 150,
            ...     'norm_delay_up': [0.0] * 150,
            ...     'norm_loss_up': [0.001] * 150,
            ...     'norm_bw_up': [10.0] * 150,
            ...     'norm_delay_down': [0.0] * 150,
            ...     'norm_loss_down': [0.0] * 150,
            ...     'norm_bw_down': [12.0] * 150
            ... })
            >>> extractor = WindowExtractor()
            >>> windows = extractor.extract_windows(df, 'test.parquet')
            >>> len(windows)
            2
            >>> windows[0]['start_index']
            0
            >>> windows[1]['start_index']
            50
        """
        windows = []
        total_rows = len(df)

        # 生成滑动窗口
        for start in range(0, total_rows - self.window_size + 1, self.step_size):
            end = start + self.window_size

            # 提取窗口
            window_df = df.iloc[start:end]

            # 验证窗口
            is_valid = self._validate_window(window_df)

            # 创建窗口对象
            window = {
                'source_file': source_file,
                'start_index': start,
                'is_valid': is_valid,
                'raw_delay_up': window_df['raw_delay_up'].values.tolist(),
                'raw_loss_up': window_df['raw_loss_up'].values.tolist(),
                'raw_bw_up': window_df['raw_bw_up'].values.tolist(),
                'raw_delay_down': window_df['raw_delay_down'].values.tolist(),
                'raw_loss_down': window_df['raw_loss_down'].values.tolist(),
                'raw_bw_down': window_df['raw_bw_down'].values.tolist(),
                'norm_delay_up': window_df['norm_delay_up'].values.tolist(),
                'norm_loss_up': window_df['norm_loss_up'].values.tolist(),
                'norm_bw_up': window_df['norm_bw_up'].values.tolist(),
                'norm_delay_down': window_df['norm_delay_down'].values.tolist(),
                'norm_loss_down': window_df['norm_loss_down'].values.tolist(),
                'norm_bw_down': window_df['norm_bw_down'].values.tolist()
            }

            windows.append(window)

        return windows

    def _validate_window(self, window_df: pd.DataFrame) -> bool:
        """验证窗口是否符合规则。

        Args:
            window_df: 窗口 DataFrame

        Returns:
            如果窗口有效返回 True，否则返回 False
        """
        # 检查高延迟
        if self._has_high_delay(window_df):
            return False

        # 检查恒定延迟
        return not self._has_constant_delay(window_df)

    def _has_high_delay(self, window_df: pd.DataFrame) -> bool:
        """检查窗口是否存在高延迟值。

        Args:
            window_df: 窗口 DataFrame

        Returns:
            如果检测到高延迟返回 True
        """
        # 检查是否有延迟超过 2000ms
        return (
            (window_df['raw_delay_up'] > 2000).any() or
            (window_df['raw_delay_down'] > 2000).any()
        )

    def _has_constant_delay(self, window_df: pd.DataFrame) -> bool:
        """检查窗口是否存在恒定延迟值（连续 ≥10 个周期）。

        Args:
            window_df: 窗口 DataFrame

        Returns:
            如果检测到恒定延迟返回 True
        """
        # 检查两个方向
        if self._check_constant_sequence(window_df['raw_delay_up'].values):
            return True

        return bool(self._check_constant_sequence(window_df['raw_delay_down'].values))

    def _check_constant_sequence(self, values: np.ndarray) -> bool:
        """检查数组是否存在长度 ≥10 的恒定序列。

        Args:
            values: 延迟值数组

        Returns:
            如果找到恒定序列返回 True
        """
        if len(values) < 10:
            return False

        # 计算连续值之间的差异
        diffs = np.diff(values)

        # 找到差异为零的位置（恒定值）
        constant_mask = (diffs == 0)

        # 找到连续的恒定序列
        current_count = 1

        for is_constant in constant_mask:
            if is_constant:
                current_count += 1
                if current_count >= 10:
                    return True
            else:
                current_count = 1

        return False
