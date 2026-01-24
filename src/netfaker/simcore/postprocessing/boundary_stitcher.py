"""边界感知拼接后处理模块。

在严格保留窗口内部结构与真实状态跃迁的前提下，仅对同状态窗口间因独立抽样导致的非物理边界不连续进行最小干预，
显著改善时序自相关性（ACF），同时保护丢包突发性（burst）与联合一致性。
"""

from typing import List, Optional

import numpy as np
from loguru import logger

from netfaker.core.config import config


class BoundaryStitcher:
    """边界感知拼接后处理器。

    用于消除合成轨迹中的窗口拼接跳跃，改善时序自相关性与分布一致性。
    """

    def __init__(self,
                 overlap_points: Optional[int] = None,
                 loss_continuation_points: Optional[int] = None,
                 loss_burst_threshold: Optional[float] = None):
        """初始化边界感知拼接后处理器。

        Args:
            overlap_points: 重叠点数，用于时延Hann局部混合
            loss_continuation_points: 丢包延续点数
            loss_burst_threshold: 丢包突发阈值（百分比）
        """
        # 使用配置值或默认值
        self.overlap_points = overlap_points or config.boundary_overlap_points
        self.loss_continuation_points = loss_continuation_points or config.loss_continuation_points
        self.loss_burst_threshold = loss_burst_threshold or config.loss_burst_threshold_pct

        # 预计算Hann权重
        self.hann_in = self._calculate_hann_weights(self.overlap_points, fade_type="in")
        self.hann_out = self._calculate_hann_weights(self.overlap_points, fade_type="out")

        logger.info(f"BoundaryStitcher初始化完成，重叠点数: {self.overlap_points}, 丢包延续点数: {self.loss_continuation_points}, 丢包突发阈值: {self.loss_burst_threshold}%")

    def _calculate_hann_weights(self, n_points: int, fade_type: str = "in") -> np.ndarray:
        """计算Hann权重。

        Args:
            n_points: 权重点数
            fade_type: 淡入/淡出类型，"in"或"out"

        Returns:
            np.ndarray: Hann权重数组
        """
        # 创建Hann窗口
        hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_points) / (n_points - 1)))
        
        if fade_type == "in":
            return hann_window
        else:  # fade_type == "out"
            return hann_window[::-1]

    def smooth_delay(self, delay_windows: List[np.ndarray], state_ids: List[int]) -> List[np.ndarray]:
        """平滑时延序列，消除窗口拼接跳跃。

        仅对同状态窗口间应用Hann局部混合，不同状态窗口间保留原始跃迁。

        Args:
            delay_windows: 时延窗口列表，每个窗口shape=(100,)
            state_ids: 每个窗口对应的状态ID列表

        Returns:
            List[np.ndarray]: 平滑后的时延窗口列表
        """
        if not delay_windows:
            return delay_windows

        logger.info(f"开始平滑时延序列，共 {len(delay_windows)} 个窗口")

        # 创建输出窗口列表，深拷贝避免修改原始数据
        smoothed_windows = [np.copy(window) for window in delay_windows]
        smoothed_count = 0

        # 遍历所有相邻窗口对
        for i in range(len(smoothed_windows) - 1):
            current_window = smoothed_windows[i]
            next_window = smoothed_windows[i + 1]
            
            # 仅对同状态窗口进行处理
            if state_ids[i] == state_ids[i + 1]:
                smoothed_count += 1
                
                # 获取当前窗口尾部和下一窗口头部的重叠区域
                current_tail = current_window[-self.overlap_points:]
                next_head = next_window[:self.overlap_points]
                
                # 应用Hann混合
                mixed = self.hann_out * current_tail + self.hann_in * next_head
                
                # 更新下一窗口的前N点
                next_window[:self.overlap_points] = mixed
                smoothed_windows[i + 1] = next_window
                
                logger.debug(f"窗口 {i}→{i+1} (state: {state_ids[i]}) 应用Hann混合")

        logger.info(f"时延序列平滑完成，共处理 {smoothed_count} 对同状态窗口")
        return smoothed_windows

    def extend_loss_bursts(self, loss_windows: List[np.ndarray], state_ids: List[int]) -> List[np.ndarray]:
        """延续丢包突发，消除窗口拼接处的丢包突变。

        仅对同状态窗口间的丢包突发进行延续，不同状态窗口间保留原始跃迁。

        Args:
            loss_windows: 丢包窗口列表，每个窗口shape=(100,)
            state_ids: 每个窗口对应的状态ID列表

        Returns:
            List[np.ndarray]: 处理后的丢包窗口列表
        """
        if not loss_windows:
            return loss_windows

        logger.info(f"开始延续丢包突发，共 {len(loss_windows)} 个窗口")

        # 创建输出窗口列表，深拷贝避免修改原始数据
        extended_windows = [np.copy(window) for window in loss_windows]
        extended_count = 0

        # 遍历所有相邻窗口对
        for i in range(len(extended_windows) - 1):
            current_window = extended_windows[i]
            next_window = extended_windows[i + 1]
            
            # 仅对同状态窗口进行处理
            if state_ids[i] == state_ids[i + 1]:
                current_last_loss = current_window[-1]
                next_first_loss = next_window[0]
                
                # 检查是否需要延续丢包突发
                if current_last_loss > self.loss_burst_threshold and next_first_loss <= self.loss_burst_threshold:
                    extended_count += 1
                    
                    # 获取当前窗口末尾丢包值
                    last_val = current_last_loss
                    
                    # 延续丢包到下一窗口开头
                    extend_points = min(self.loss_continuation_points, len(next_window))
                    extended_windows[i + 1][:extend_points] = last_val
                    
                    logger.debug(f"窗口 {i}→{i+1} (state: {state_ids[i]}) 延续丢包突发，从 {last_val:.2f}% 开始")

        logger.info(f"丢包突发延续完成，共处理 {extended_count} 对同状态窗口")
        return extended_windows
    
    # 兼容旧接口
    def smooth_delay_sequence(self, delay_windows: List[np.ndarray]) -> List[np.ndarray]:
        """平滑时延序列（兼容旧接口）。

        Args:
            delay_windows: 时延窗口列表，每个窗口shape=(100,)

        Returns:
            List[np.ndarray]: 平滑后的时延窗口列表
        """
        # 为兼容旧接口，创建默认state_ids（所有窗口同一状态）
        state_ids = [0] * len(delay_windows)
        return self.smooth_delay(delay_windows, state_ids)
    
    # 兼容旧接口
    def extend_loss_bursts_old(self, loss_windows: List[np.ndarray]) -> List[np.ndarray]:
        """延续丢包突发（兼容旧接口）。

        Args:
            loss_windows: 丢包窗口列表，每个窗口shape=(100,)

        Returns:
            List[np.ndarray]: 处理后的丢包窗口列表
        """
        # 为兼容旧接口，创建默认state_ids（所有窗口同一状态）
        state_ids = [0] * len(loss_windows)
        return self.extend_loss_bursts(loss_windows, state_ids)
