#!/usr/bin/env python3
"""网络行为模式发现模块（简化版）
负责使用规则事件检测引擎发现网络行为模式
"""

import pandas as pd
import numpy as np
from typing import Dict


# 行为标签定义
BEHAVIOR_LABELS = {
    "STABLE": 0,
    "WEAK_BURST": 1,
    "FREQUENT_FLUCTUATION": 2,
    "HIGH_DELAY_NO_LOSS": 3,
    "HIGH_LOSS_STEADY": 4,
    "LOW_DELAY_HIGH_LOSS": 5,
    "STRONG_BURST": 6,
    "INSTANT_SPIKE": 7,
    "INVALID": 8
}

# 行为标签反向映射
LABEL_TO_NAME = {v: k for k, v in BEHAVIOR_LABELS.items()}


class PatternIdentifier:
    """使用规则事件检测引擎识别网络行为模式（简化版）"""
    
    def __init__(self):
        """初始化模式识别器，使用默认配置"""
        # 默认配置参数
        self.STRONG_BURST_DELAY_THRESHOLD = 400
        self.STRONG_BURST_LOSS_THRESHOLD = 0.25
        self.STRONG_BURST_MIN_RUN = 15
        self.INSTANT_SPIKE_DELAY_THRESHOLD = 800
        self.INSTANT_SPIKE_LOSS_THRESHOLD = 0.8
        self.INSTANT_SPIKE_MAX_COUNT = 5
        self.INSTANT_SPIKE_MAX_RATIO = 0.1
        self.WEAK_BURST_LOSS_NONZERO_RATIO = 0.3
        self.WEAK_BURST_MIN_CONDITIONS = 2
        # 静态阈值（调整为更适合游戏场景）
        self.DEFAULT_DELAY_MEAN_LOW = 80
        self.DEFAULT_DELAY_MEAN_HIGH = 300
        self.DEFAULT_DELAY_STD_HIGH = 200
        self.DEFAULT_LOSS_MEAN_LOW = 0.05
        self.DEFAULT_LOSS_MEAN_HIGH = 0.2
        self.DEFAULT_LOSS_STD_HIGH = 0.1
    
    def identify(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame) -> Dict:
        """使用规则事件检测引擎识别网络行为模式
        
        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame，必须包含window_start和window_end列
            raw_data_df (pd.DataFrame): 原始数据DataFrame，用于检测所有行为模式
        
        Returns:
            Dict: 包含识别结果的字典，包括标签等信息
        """
        # 检查必要列
        required_cols = ["window_start", "window_end"]
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"features_df必须包含以下列: {required_cols}。缺少: {missing_cols}")
        
        # 重置raw_data_df索引
        if not all(raw_data_df.index == range(len(raw_data_df))):
            raw_data_df = raw_data_df.reset_index(drop=True)
        
        # 计算动态阈值（简化版，使用静态阈值）
        thresholds = {
            "delay_mean_low": self.DEFAULT_DELAY_MEAN_LOW,
            "delay_mean_high": self.DEFAULT_DELAY_MEAN_HIGH,
            "delay_std_high": self.DEFAULT_DELAY_STD_HIGH,
            "loss_mean_low": self.DEFAULT_LOSS_MEAN_LOW,
            "loss_mean_high": self.DEFAULT_LOSS_MEAN_HIGH,
            "loss_std_high": self.DEFAULT_LOSS_STD_HIGH,
        }
        
        # 执行规则事件检测
        labels = self._perform_rule_based_detection(features_df, raw_data_df, thresholds)
        
        # 准备结果
        results = {
            "method": "rule",
            "labels": labels.tolist(),
            "behavior_stats": self._calculate_behavior_statistics(labels)
        }
        
        return results
    
    def _calculate_window_stats(self, delays: np.ndarray, loss_rates: np.ndarray) -> tuple:
        """计算窗口统计信息"""
        if len(delays) == 0 or len(loss_rates) == 0:
            return None
        else:
            delay_mean = np.mean(delays)
            delay_std = np.std(delays)
            loss_mean = np.mean(loss_rates)
            loss_std = np.std(loss_rates)
            return delay_mean, delay_std, loss_mean, loss_std
    
    def _detect_strong_burst(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测Strong Burst行为（强突发）"""
        # 检测条件1：高延迟且高丢包的连续序列
        congested = (delays >= self.STRONG_BURST_DELAY_THRESHOLD) & (
            loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD
        )
        
        # 计算连续True序列的最大长度
        def calculate_max_consecutive_true(arr):
            max_run = 0
            current_run = 0
            for val in arr:
                if val:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run
        
        max_run = calculate_max_consecutive_true(congested)
        
        # 检测条件2：整体窗口的平均延迟和丢包率都较高
        delay_mean = np.mean(delays) if len(delays) > 0 else 0
        loss_mean = np.mean(loss_rates) if len(loss_rates) > 0 else 0
        
        overall_high = (
            delay_mean > self.STRONG_BURST_DELAY_THRESHOLD * 0.8 and 
            loss_mean > self.STRONG_BURST_LOSS_THRESHOLD * 0.8
        )
        
        # 检测条件3：窗口内有多个高延迟或高丢包的峰值
        high_delay_count = np.sum(delays >= self.STRONG_BURST_DELAY_THRESHOLD * 1.2)
        high_loss_count = np.sum(loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD * 1.2)
        multiple_peaks = high_delay_count >= 5 or high_loss_count >= 5
        
        return max_run >= self.STRONG_BURST_MIN_RUN and overall_high and multiple_peaks
    
    def _detect_instant_spike(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测瞬时峰值行为"""
        instant_spike_count = np.sum(
            (delays >= self.INSTANT_SPIKE_DELAY_THRESHOLD) | 
            (loss_rates >= self.INSTANT_SPIKE_LOSS_THRESHOLD)
        )
        total_points = len(delays)
        spike_ratio = instant_spike_count / total_points if total_points > 0 else 0
        return 0 < instant_spike_count <= self.INSTANT_SPIKE_MAX_COUNT and spike_ratio < self.INSTANT_SPIKE_MAX_RATIO
    
    def _detect_behavior_with_raw_data(
        self, delays: np.ndarray, loss_rates: np.ndarray, thresholds: Dict[str, float]
    ) -> int:
        """使用原始数据检测行为"""
        # 计算窗口统计信息
        stats = self._calculate_window_stats(delays, loss_rates)
        
        # 如果窗口无效，标记为INVALID
        if stats is None:
            return BEHAVIOR_LABELS["INVALID"]
        
        delay_mean, delay_std, loss_mean, loss_std = stats
        
        # 按行为严重性从高到低检测
        # 1. 瞬时峰值 (INSTANT_SPIKE)
        if self._detect_instant_spike(delays, loss_rates):
            return BEHAVIOR_LABELS["INSTANT_SPIKE"]
        
        # 2. 低延迟高丢包 (LOW_DELAY_HIGH_LOSS)
        if (
            delay_mean < thresholds["delay_mean_low"] * 0.8 and 
            loss_mean > thresholds["loss_mean_high"] * 1.2 and 
            delay_mean > 0 and 
            loss_mean > 0.1
        ):
            return BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]
        
        # 3. 强突发 (STRONG_BURST)
        if self._detect_strong_burst(delays, loss_rates):
            return BEHAVIOR_LABELS["STRONG_BURST"]
        
        # 4. 持续高丢包 (HIGH_LOSS_STEADY)
        if (
            loss_mean > thresholds["loss_mean_high"] and 
            loss_std < thresholds["loss_std_high"]
        ):
            return BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]
        
        # 5. 高延迟无丢包 (HIGH_DELAY_NO_LOSS)
        if (
            delay_mean > thresholds["delay_mean_high"] and 
            loss_mean <= thresholds["loss_mean_low"]
        ):
            return BEHAVIOR_LABELS["HIGH_DELAY_NO_LOSS"]
        
        # 6. 频繁波动 (FREQUENT_FLUCTUATION)
        delay_cv = delay_std / delay_mean if delay_mean > 0 else 0
        loss_cv = loss_std / loss_mean if loss_mean > 0 else 0
        delay_cv_threshold = 0.5
        loss_cv_threshold = 1.0
        
        # 重新调整分类逻辑，从最不稳定到最稳定依次判断
        
        # 9. 频繁波动判断（最不稳定）
        frequent_fluctuation = (
            delay_cv > delay_cv_threshold or 
            loss_cv > loss_cv_threshold or
            delay_std > thresholds["delay_std_high"] / 1.2 or 
            loss_std > thresholds["loss_std_high"] / 1.2
        )
        
        if frequent_fluctuation:
            return BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]
        
        # 10. 弱突发判断
        # 弱突发条件：存在一定波动但尚未达到频繁波动
        weak_burst = (
            (loss_mean > thresholds["loss_mean_low"] * 1.2 and loss_mean < thresholds["loss_mean_high"] * 0.8) or
            (delay_std > thresholds["delay_std_high"] / 3.5 and delay_std <= thresholds["delay_std_high"] / 1.2) or
            (loss_std > thresholds["loss_std_high"] / 3.5 and loss_std <= thresholds["loss_std_high"] / 1.2)
        )
        
        if weak_burst:
            return BEHAVIOR_LABELS["WEAK_BURST"]
        
        # 11. 稳定行为判断（最稳定）
        # 稳定状态条件：严格的低丢包、低时延、低波动
        stable = (
            loss_mean <= thresholds["loss_mean_low"] * 1.2 and 
            delay_mean < thresholds["delay_mean_high"] * 0.6 and
            delay_std < thresholds["delay_std_high"] / 4.0 and 
            loss_std < thresholds["loss_std_high"] / 4.0 and 
            # 严格的峰值检测：窗口内没有明显的峰值
            np.max(delays) < self.STRONG_BURST_DELAY_THRESHOLD * 0.3 and
            np.max(loss_rates) < self.STRONG_BURST_LOSS_THRESHOLD * 0.3
        )
        
        if stable:
            return BEHAVIOR_LABELS["STABLE"]
        
        # 12. 其他情况默认返回弱突发（接近稳定但又不完全稳定的状态）
        return BEHAVIOR_LABELS["WEAK_BURST"]
    
    def _perform_rule_based_detection(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame, thresholds: Dict[str, float]
    ) -> np.ndarray:
        """执行规则事件检测"""
        # 初始化合并标签数组，默认为STABLE
        merged_labels = np.full(len(features_df), BEHAVIOR_LABELS["STABLE"], dtype=int)
        
        # 检查raw_data_df是否包含必要的列
        # 优先使用带有_origin后缀的原始数据列
        if all(
            col in raw_data_df.columns
            for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"]
        ):
            # 使用格式1: delay1, loss_rate1, delay2, loss_rate2
            delay_col1, loss_col1, delay_col2, loss_col2 = "delay1", "loss_rate1", "delay2", "loss_rate2"
        elif all(
            col in raw_data_df.columns
            for col in ["delay_up_origin", "loss_up_origin", "delay_down_origin", "loss_down_origin"]
        ):
            # 使用格式4: delay_up_origin, loss_up_origin, delay_down_origin, loss_down_origin (原始数据)
            delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up_origin", "loss_up_origin", "delay_down_origin", "loss_down_origin"
        elif all(
            col in raw_data_df.columns
            for col in ["delay_up", "loss_up", "delay_down", "loss_down"]
        ):
            # 使用格式2: delay_up, loss_up, delay_down, loss_down
            delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up", "loss_up", "delay_down", "loss_down"
        elif all(
            col in raw_data_df.columns
            for col in ["delay_up", "loss_up", "delay_down", "loss_dn"]
        ):
            # 使用格式3: delay_up, loss_up, delay_down, loss_dn
            delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up", "loss_up", "delay_down", "loss_dn"
        else:
            raise ValueError(
                "raw_data_df必须包含以下列之一: [delay1, loss_rate1, delay2, loss_rate2] 或 [delay_up_origin, loss_up_origin, delay_down_origin, loss_down_origin] 或 [delay_up, loss_up, delay_down, loss_down] 或 [delay_up, loss_up, delay_down, loss_dn]"
            )
        
        # 逐窗口处理
        for i, (_, row) in enumerate(features_df.iterrows()):
            window_start = int(row["window_start"])
            window_end = int(row["window_end"])
            
            # 检查窗口有效性
            if window_start >= window_end or window_start >= len(raw_data_df):
                merged_labels[i] = BEHAVIOR_LABELS["INVALID"]
                continue
            
            # 调整window_end
            if window_end > len(raw_data_df):
                window_end = len(raw_data_df)
                if window_start >= window_end:
                    merged_labels[i] = BEHAVIOR_LABELS["INVALID"]
                    continue
            
            # 提取窗口数据
            window_data = raw_data_df.iloc[window_start:window_end]
            
            # 获取上下行数据
            delay_up = window_data[delay_col1].values
            loss_rate_up = window_data[loss_col1].values
            delay_down = window_data[delay_col2].values
            loss_rate_down = window_data[loss_col2].values
            
            # 检查窗口数据是否为空
            if len(delay_up) == 0 or len(loss_rate_up) == 0 or len(delay_down) == 0 or len(loss_rate_down) == 0:
                merged_labels[i] = BEHAVIOR_LABELS["INVALID"]
                continue
            
            # 对上下行数据分别检测行为
            behavior_up = self._detect_behavior_with_raw_data(delay_up, loss_rate_up, thresholds)
            behavior_down = self._detect_behavior_with_raw_data(delay_down, loss_rate_down, thresholds)
            
            # 合并上下行标签，取更严重的行为（数值更大表示更严重）
            merged_labels[i] = max(behavior_up, behavior_down)
        
        return merged_labels
    
    def _calculate_behavior_statistics(self, labels: np.ndarray) -> Dict:
        """计算行为统计信息"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        behavior_stats = {}
        for label, count in zip(unique_labels, counts):
            behavior_stats[LABEL_TO_NAME[label]] = {
                "label": int(label),
                "count": int(count),
                "percentage": (count / len(labels)) * 100
            }
        return behavior_stats
