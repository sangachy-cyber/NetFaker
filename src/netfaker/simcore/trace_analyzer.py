"""状态序列提取器模块。

从真实HoloWAN仿真文件中自动提取结构化的状态序列（segments），
用于回放、分析或再生成。
"""

import os
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from netfaker.simcore.clustering.state_predictor import StatePredictor
from netfaker.simcore.io.holowan_loader import HoloWANLoader
from netfaker.simcore.synthesizer.state_sequence_compressor import (
    StateSequenceCompressor,
)


class TraceAnalyzer:
    """状态序列提取器类。

    负责从HoloWAN文件中提取结构化的状态序列。

    示例：
    >>> analyzer = TraceAnalyzer()
    >>> segments = analyzer.analyze("data/raw/real_trace.txt")
    >>> print(segments[:2])
    [{"type": "s2", "duration": 30}, {"type": "s5", "duration": 50}]
    """

    def __init__(self,
                 model_path: str = "data/clusters/gmm_model.joblib",
                 scaler_path: str = "data/clusters/feature_scaler.joblib",
                 metadata_path: str = "data/clusters/state_metadata.json",
                 window_size: int = 100):
        """初始化状态序列提取器。

        Args:
            model_path: 聚类模型路径
            scaler_path: 特征缩放器路径
            metadata_path: 状态元数据路径
            window_size: 窗口大小
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.window_size = window_size

    def analyze(self, input_path: str) -> List[Dict[str, Any]]:
        """从HoloWAN文件中提取状态序列。

        Args:
            input_path: 输入HoloWAN文件路径

        Returns:
            符合segments格式的字典列表
        """
        # 1. 加载HoloWAN文件
        df = self._load_holowan_file(input_path)

        # 2. 提取窗口
        file_name = os.path.basename(input_path)
        windows = self._extract_raw_windows(df, file_name)

        if not windows:
            logger.warning("没有提取到有效窗口，返回空segments")
            return []

        # 3. 初始化状态预测器
        predictor = StatePredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            metadata_path=self.metadata_path
        )

        # 4. 预测每个窗口的状态
        state_ids = self._predict_states(windows, predictor)

        if not state_ids:
            logger.warning("没有成功预测任何窗口的状态，返回空segments")
            return []

        # 5. 压缩状态序列
        compressor = StateSequenceCompressor()
        segments = compressor.compress(state_ids)

        return segments

    def _load_holowan_file(self, file_path: str) -> pd.DataFrame:
        """加载HoloWAN文件并返回DataFrame。

        Args:
            file_path: HoloWAN文件路径

        Returns:
            包含网络数据的DataFrame
        """
        logger.info(f"加载文件: {file_path}")
        loader = HoloWANLoader()
        df = loader.load_file(file_path)

        if df is None or df.empty:
            logger.error(f"文件 {file_path} 为空或加载失败")
            raise ValueError(f"文件 {file_path} 为空或加载失败")

        logger.info(f"成功加载文件 {file_path}，共 {len(df)} 行数据")
        return df

    def _extract_raw_windows(self, df: pd.DataFrame, file_name: str) -> List[Dict[str, Any]]:
        """从DataFrame中提取非重叠窗口，只包含raw_*列。

        Args:
            df: 输入DataFrame
            file_name: 源文件名

        Returns:
            窗口列表
        """
        if len(df) < self.window_size:
            logger.warning(f"文件数据不足{self.window_size}行，无法提取有效窗口")
            return []

        windows = []
        total_rows = len(df)

        # 生成非重叠窗口
        for start in range(0, total_rows - self.window_size + 1, self.window_size):
            end = start + self.window_size

            # 提取窗口数据
            window_df = df.iloc[start:end]

            # 只保留raw_*列
            window = {
                'source_file': file_name,
                'start_index': start,
                'is_valid': True,  # 简化处理，假设所有窗口都有效
                'raw_delay_up': window_df['raw_delay_up'].values.tolist(),
                'raw_loss_up': window_df['raw_loss_up'].values.tolist(),
                'raw_bw_up': window_df['raw_bw_up'].values.tolist(),
                'raw_delay_down': window_df['raw_delay_down'].values.tolist(),
                'raw_loss_down': window_df['raw_loss_down'].values.tolist(),
                'raw_bw_down': window_df['raw_bw_down'].values.tolist()
            }
            windows.append(window)

        logger.info(f"成功提取 {len(windows)} 个窗口")
        return windows

    def _predict_states(self, windows: List[Dict[str, Any]], predictor: StatePredictor) -> List[int]:
        """预测窗口的状态ID。

        Args:
            windows: 窗口列表
            predictor: 状态预测器实例

        Returns:
            每个窗口的状态ID列表
        """
        state_ids = []
        for i, window in enumerate(windows):
            try:
                state_id = predictor.predict_from_window(window)
                state_ids.append(state_id)
                logger.debug(f"窗口 {i+1} 预测状态: s{state_id}")
            except Exception as e:
                logger.error(f"预测窗口 {i+1} 失败: {e}")
                # 跳过失败的窗口
                continue

        logger.info(f"成功预测了 {len(state_ids)} 个窗口的状态")
        return state_ids
