"""窗口采样器模块。

负责加载已标注的窗口数据，并根据指定的state_id抽样窗口。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


class WindowSampler:
    """窗口采样器类。

    用于加载已标注的窗口数据，并根据指定的state_id抽样窗口。
    """

    def __init__(self, window_pool_path: str):
        """初始化窗口采样器。

        Args:
            window_pool_path: 已标注窗口数据的Parquet文件路径

        Examples:
            >>> sampler = WindowSampler("data/clusters/train_with_state.parquet")
            >>> len(sampler.state_to_windows) > 0
            True
        """
        self.window_pool_path = window_pool_path
        self.data: Optional[pd.DataFrame] = None
        self.state_to_windows: Dict[int, List[int]] = {}
        self._load_data()
        self._build_state_index()

    def _load_data(self):
        """加载已标注的窗口数据。

        Raises:
            FileNotFoundError: 当窗口数据文件不存在时
            ValueError: 当窗口数据格式不符合要求时
        """
        logger.info(f"加载窗口数据: {self.window_pool_path}")
        try:
            self.data = pd.read_parquet(self.window_pool_path)
        except Exception as e:
            raise ValueError(f"加载窗口数据失败: {str(e)}") from e

        # 校验数据格式
        required_columns = [
            "state_id",
            "raw_delay_up", "raw_delay_down",
            "raw_loss_up", "raw_loss_down",
            "raw_bw_up", "raw_bw_down"
        ]
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"窗口数据缺少必填列: {col}")

        logger.info(f"成功加载 {len(self.data)} 个窗口")

    def _build_state_index(self):
        """按state_id建立窗口索引映射。
        """
        logger.info("构建state_id到窗口的索引映射")
        
        # 按state_id分组，获取每个state_id对应的窗口索引
        grouped = self.data.groupby("state_id")
        for state_id, group in grouped:
            self.state_to_windows[state_id] = group.index.tolist()

        logger.info(f"索引映射构建完成，包含 {len(self.state_to_windows)} 个不同的state_id")
        for state_id, window_count in self.state_to_windows.items():
            logger.debug(f"state_id={state_id}: {len(window_count)} 个窗口")

    def sample_windows(self, state_id: int, n_windows: int, seed: Optional[int] = None) -> List[Dict]:
        """从指定state_id的窗口中抽样。
        
        Args:
            state_id: 要抽样的state_id
            n_windows: 要抽样的窗口数量
            seed: 随机种子，用于控制抽样的可复现性
            
        Returns:
            List[Dict]: 抽样得到的窗口字典列表
            
        Raises:
            ValueError: 当指定的state_id不存在时
            
        Examples:
            >>> sampler = WindowSampler("data/clusters/train_with_state.parquet")
            >>> windows = sampler.sample_windows(0, 2, seed=42)
            >>> len(windows)
            2
        """
        if state_id not in self.state_to_windows:
            raise ValueError(f"没有可用的窗口对应state_id={state_id}")

        # 获取该state_id对应的所有窗口索引
        available_windows = self.state_to_windows[state_id]

        # 使用随机种子控制抽样
        import numpy as np
        rng = np.random.RandomState(seed)

        # 有放回随机抽样
        sampled_indices = rng.choice(available_windows, size=n_windows, replace=True)

        # 获取抽样的窗口数据，转换为字典避免SettingWithCopyWarning
        sampled_windows = [self.data.loc[idx].to_dict() for idx in sampled_indices]

        logger.info(f"从state_id={state_id}抽样 {n_windows} 个窗口，可用窗口数: {len(available_windows)}")

        return sampled_windows
