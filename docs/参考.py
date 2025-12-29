#!/usr/bin/env python3
"""网络行为模式发现模块
负责使用规则事件检测引擎发现网络行为模式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from ..utils.logger import get_logger

from ..config.config_loader import config

BEHAVIOR_LABELS = config("behavior_labels")

logger = get_logger(__name__)


class PatternIdentifier:
    """使用规则事件检测引擎识别网络行为模式

    该类使用基于规则的事件检测算法识别网络行为模式，支持上下行独立检测，
    能够识别多种网络行为模式，包括稳定、弱突发、强突发、瞬时峰值等。
    """

    # 行为标签反向映射，便于外部解读数字标签
    # 使用配置中的BEHAVIOR_LABELS，确保与配置保持一致
    LABEL_TO_NAME = {v: k for k, v in BEHAVIOR_LABELS.items()}

    def __init__(self, method: str = "rule", config: Dict = None):
        """初始化模式识别器

        Args:
            method (str, optional): 检测方法，目前只支持 "rule"，默认为 "rule"
            config (Dict): 配置字典，必须包含以下参数：
                - STRONG_BURST_DELAY_THRESHOLD: Strong Burst的延迟阈值
                - STRONG_BURST_LOSS_THRESHOLD: Strong Burst的丢包率阈值
                - STRONG_BURST_MIN_RUN: Strong Burst的最小连续点数量
                - INSTANT_SPIKE_DELAY_THRESHOLD: 瞬时峰值的延迟阈值
                - INSTANT_SPIKE_LOSS_THRESHOLD: 瞬时峰值的丢包率阈值
                - INSTANT_SPIKE_MAX_COUNT: 瞬时峰值的最大峰值点数量
                - INSTANT_SPIKE_MAX_RATIO: 瞬时峰值的最大峰值占比
                - WEAK_BURST_LOSS_NONZERO_RATIO: Weak Burst的丢包率非零比例阈值
                - WEAK_BURST_MIN_CONDITIONS: Weak Burst的最小满足条件数量
                - DYNAMIC_THRESHOLD_WINDOW_SIZE: 动态阈值计算的窗口大小
                - save_allowed_prefixes: 保存时允许的列名前缀
                - DEFAULT_DELAY_MEAN_LOW: 低延迟阈值
                - DEFAULT_DELAY_MEAN_HIGH: 高延迟阈值
                - DEFAULT_DELAY_STD_HIGH: 高延迟标准差阈值
                - DEFAULT_LOSS_MEAN_LOW: 低丢包率阈值
                - DEFAULT_LOSS_MEAN_HIGH: 高丢包率阈值
                - DEFAULT_LOSS_STD_HIGH: 高丢包率标准差阈值

        Raises:
            ValueError: 当config为None时抛出

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = {
            ...     'STRONG_BURST_DELAY_THRESHOLD': 400,
            ...     'STRONG_BURST_LOSS_THRESHOLD': 0.25,
            ...     'STRONG_BURST_MIN_RUN': 15,
            ...     'INSTANT_SPIKE_DELAY_THRESHOLD': 800,
            ...     'INSTANT_SPIKE_LOSS_THRESHOLD': 0.8,
            ...     'INSTANT_SPIKE_MAX_COUNT': 5,
            ...     'INSTANT_SPIKE_MAX_RATIO': 0.1,
            ...     'WEAK_BURST_LOSS_NONZERO_RATIO': 0.3,
            ...     'WEAK_BURST_MIN_CONDITIONS': 2,
            ...     'DYNAMIC_THRESHOLD_WINDOW_SIZE': 100,
            ...     'save_allowed_prefixes': ['feat_', 'window_'],
            ...     'DEFAULT_DELAY_MEAN_LOW': 50,
            ...     'DEFAULT_DELAY_MEAN_HIGH': 200,
            ...     'DEFAULT_DELAY_STD_HIGH': 100,
            ...     'DEFAULT_LOSS_MEAN_LOW': 0.01,
            ...     'DEFAULT_LOSS_MEAN_HIGH': 0.1,
            ...     'DEFAULT_LOSS_STD_HIGH': 0.05
            ... }
            >>> pattern_identifier = PatternIdentifier(method="rule", config=config)

        """
        self.method = method
        # 初始化为空列表，将在identify方法中动态获取
        self.feature_columns: List[str] = []

        # 强制要求配置
        if config is None:
            raise ValueError("Config must be provided")

        self.config = config

        # 从config中强制获取参数，支持Pydantic模型和字典
        if hasattr(self.config, "__dict__"):
            # Pydantic模型或对象，使用属性访问
            self.STRONG_BURST_DELAY_THRESHOLD = self.config.strong_burst_delay_threshold
            self.STRONG_BURST_LOSS_THRESHOLD = self.config.strong_burst_loss_threshold
            self.STRONG_BURST_MIN_RUN = self.config.strong_burst_min_run
            self.INSTANT_SPIKE_DELAY_THRESHOLD = (
                self.config.instant_spike_delay_threshold
            )
            self.INSTANT_SPIKE_LOSS_THRESHOLD = self.config.instant_spike_loss_threshold
            self.INSTANT_SPIKE_MAX_COUNT = self.config.instant_spike_max_count
            self.INSTANT_SPIKE_MAX_RATIO = self.config.instant_spike_max_ratio
            self.WEAK_BURST_LOSS_NONZERO_RATIO = (
                self.config.weak_burst_loss_nonzero_ratio
            )
            self.WEAK_BURST_MIN_CONDITIONS = self.config.weak_burst_min_conditions
            self.DYNAMIC_THRESHOLD_WINDOW_SIZE = (
                self.config.dynamic_threshold_window_size
            )
            self.SAVE_ALLOWED_PREFIXES = self.config.save_allowed_prefixes
            self.DEFAULT_DELAY_MEAN_LOW = self.config.default_delay_mean_low
            self.DEFAULT_DELAY_MEAN_HIGH = self.config.default_delay_mean_high
            self.DEFAULT_DELAY_STD_HIGH = self.config.default_delay_std_high
            self.DEFAULT_LOSS_MEAN_LOW = self.config.default_loss_mean_low
            self.DEFAULT_LOSS_MEAN_HIGH = self.config.default_loss_mean_high
            self.DEFAULT_LOSS_STD_HIGH = self.config.default_loss_std_high
        else:
            # 字典，使用下标访问
            self.STRONG_BURST_DELAY_THRESHOLD = self.config[
                "strong_burst_delay_threshold"
            ]
            self.STRONG_BURST_LOSS_THRESHOLD = self.config[
                "strong_burst_loss_threshold"
            ]
            self.STRONG_BURST_MIN_RUN = self.config["strong_burst_min_run"]
            self.INSTANT_SPIKE_DELAY_THRESHOLD = self.config[
                "instant_spike_delay_threshold"
            ]
            self.INSTANT_SPIKE_LOSS_THRESHOLD = self.config[
                "instant_spike_loss_threshold"
            ]
            self.INSTANT_SPIKE_MAX_COUNT = self.config["instant_spike_max_count"]
            self.INSTANT_SPIKE_MAX_RATIO = self.config["instant_spike_max_ratio"]
            self.WEAK_BURST_LOSS_NONZERO_RATIO = self.config[
                "weak_burst_loss_nonzero_ratio"
            ]
            self.WEAK_BURST_MIN_CONDITIONS = self.config["weak_burst_min_conditions"]
            self.DYNAMIC_THRESHOLD_WINDOW_SIZE = self.config[
                "dynamic_threshold_window_size"
            ]
            self.SAVE_ALLOWED_PREFIXES = self.config["save_allowed_prefixes"]
            self.DEFAULT_DELAY_MEAN_LOW = self.config["default_delay_mean_low"]
            self.DEFAULT_DELAY_MEAN_HIGH = self.config["default_delay_mean_high"]
            self.DEFAULT_DELAY_STD_HIGH = self.config["default_delay_std_high"]
            self.DEFAULT_LOSS_MEAN_LOW = self.config["default_loss_mean_low"]
            self.DEFAULT_LOSS_MEAN_HIGH = self.config["default_loss_mean_high"]
            self.DEFAULT_LOSS_STD_HIGH = self.config["default_loss_std_high"]

        logger.debug(f"PatternIdentifier 初始化完成，使用配置: {self.config}")

    def identify(self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame) -> Dict:
        """使用规则事件检测引擎识别网络行为模式

        注意：
        1. 该方法必须提供raw_data_df，以确保所有行为模式都能被正确检测
        2. 支持检测所有8种网络行为模式
        3. 不再区分上下行，只返回合并后的行为标签和指标

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame，必须包含window_start和window_end列
            raw_data_df (pd.DataFrame): 原始数据DataFrame，用于检测所有行为模式

        Returns:
            Dict: 包含识别结果的字典，包括标签、转移矩阵和各种指标
                - method: 检测方法
                - labels: 合并后的行为标签列表
                - metrics: 合并后行为转移质量指标
                - transition_matrix: 合并后行为转移矩阵
                - behavior_stats: 合并后行为统计信息
                - separation_metrics: 合并后行为分离度指标
                - features: 特征矩阵
                - feature_columns: 特征列名列表

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = {
            ...     # 配置参数...
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'window_start': [0, 50, 100],
            ...     'window_end': [50, 100, 150],
            ...     'feat_up_delay_mean': np.random.normal(50, 10, 3),
            ...     'feat_up_loss_mean': np.random.choice([0, 0.01, 0.05], 3),
            ...     'feat_down_delay_mean': np.random.normal(60, 15, 3),
            ...     'feat_down_loss_mean': np.random.choice([0, 0.01, 0.05], 3)
            ... })
            >>> raw_data_df = pd.DataFrame({
            ...     'delay_up': np.random.normal(50, 10, 150),
            ...     'loss_rate_up': np.random.choice([0, 0.01, 0.05], 150),
            ...     'delay_down': np.random.normal(60, 15, 150),
            ...     'loss_rate_down': np.random.choice([0, 0.01, 0.05], 150)
            ... })
            >>> results = pattern_identifier.identify(features_df, raw_data_df)
            >>> print(f"合并标签数量: {len(results['labels'])}")

        """
        # 检查输入DataFrame是否为空
        if features_df.empty:
            raise ValueError("输入特征DataFrame为空")

        if raw_data_df is None:
            raise ValueError("raw_data_df必须提供，以确保所有行为模式都能被正确检测")

        # 检查必要列
        required_cols = ["window_start", "window_end"]
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(
                f"features_df必须包含以下列: {required_cols}。缺少: {missing_cols}"
            )

        # 检查raw_data_df的索引是否为连续整数
        if not all(raw_data_df.index == range(len(raw_data_df))):
            logger.warning("raw_data_df的索引不是连续整数，将重置索引")
            raw_data_df = raw_data_df.reset_index(drop=True)

        # 动态获取特征列
        self.feature_columns = [
            col for col in features_df.columns if col.startswith("feat_")
        ]
        logger.info(f"动态获取特征列: {self.feature_columns}")

        # 检查是否有特征列
        if not self.feature_columns:
            raise ValueError("未找到特征列。期望列名以'feat_'开头")

        # 提取特征数据
        X = features_df[self.feature_columns].values

        # 过滤掉包含NaN值的行，确保所有特征值都是有效的
        valid_rows = np.logical_not(np.isnan(X).any(axis=1))
        X = X[valid_rows]
        logger.info(f"过滤掉包含NaN值的行，有效行数: {len(X)}")

        # 检查是否还有有效行
        if len(X) == 0:
            raise ValueError("所有行都包含NaN值，无法进行特征提取")

        # 计算动态阈值
        thresholds = self._calculate_dynamic_thresholds(features_df, raw_data_df)

        # 执行规则事件检测
        if self.method == "rule":
            labels = self._perform_rule_based_detection(
                features_df, raw_data_df, thresholds
            )
        else:
            raise ValueError(f"未知的检测方法: {self.method}")

        # 过滤掉包含NaN值的行对应的标签
        labels = labels[valid_rows]
        logger.info(f"过滤掉无效行对应的标签，有效标签数: {len(labels)}")

        # 计算合并标签的转移矩阵和指标
        transition_matrix = self._calculate_transition_matrix(labels)
        transition_metrics = self._calculate_transition_metrics(transition_matrix)
        separation_metrics = self._calculate_separation_metrics(X, labels)
        behavior_stats = self._calculate_behavior_statistics(X, labels)

        # 合并行为标签统计
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_stats = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
        logger.info(f"合并行为标签统计: {label_stats}")

        # 准备结果，确保所有值都是JSON可序列化的
        results = {
            "method": self.method,
            "labels": labels.tolist(),
            "metrics": transition_metrics,
            "transition_matrix": transition_matrix.tolist(),
            "behavior_stats": behavior_stats,
            "separation_metrics": separation_metrics,
            "features": X.tolist(),
            "feature_columns": self.feature_columns,
        }

        return results

    def _calculate_window_stats(
        self, delays: np.ndarray, loss_rates: np.ndarray, window_idx: int = None
    ) -> tuple:
        """计算窗口统计信息

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组
            window_idx (int, optional): 窗口索引，可选，用于日志输出

        Returns:
            tuple: 包含延迟均值、延迟标准差、丢包率均值、丢包率标准差的元组；若窗口为空返回None

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> delays = np.random.normal(50, 10, 100)
            >>> loss_rates = np.random.choice([0, 0.01, 0.05], 100)
            >>> stats = pattern_identifier._calculate_window_stats(delays, loss_rates)
            >>> delay_mean, delay_std, loss_mean, loss_std = stats
            >>> print(f"延迟均值: {delay_mean:.2f}ms, 延迟标准差: {delay_std:.2f}ms")
            >>> print(f"丢包率均值: {loss_mean:.4f}, 丢包率标准差: {loss_std:.4f}")

        """
        if len(delays) == 0 or len(loss_rates) == 0:
            # 如果窗口内没有数据，返回None表示无效窗口
            if window_idx is not None:
                logger.warning(f"窗口 {window_idx} 无数据，标记为无效窗口")
            return None
        else:
            delay_mean = np.mean(delays)
            delay_std = np.std(delays)
            loss_mean = np.mean(loss_rates)
            loss_std = np.std(loss_rates)
            return delay_mean, delay_std, loss_mean, loss_std

    def _detect_strong_burst(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测Strong Burst行为（强突发）

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组

        Returns:
            bool: 是否为Strong Burst行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = {
            ...     'STRONG_BURST_DELAY_THRESHOLD': 400,
            ...     'STRONG_BURST_LOSS_THRESHOLD': 0.25,
            ...     'STRONG_BURST_MIN_RUN': 15,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 强突发数据
            >>> delays = np.array([500, 550, 600, 580, 520] * 10)  # 连续50个点超过阈值
            >>> loss_rates = np.array([0.3, 0.35, 0.28, 0.32, 0.26] * 10)  # 连续50个点超过阈值
            >>> is_strong_burst = pattern_identifier._detect_strong_burst(delays, loss_rates)
            >>> print(f"是否为强突发: {is_strong_burst}")
            True

        """
        # 检测条件1：高延迟且高丢包的连续序列
        congested = (delays >= self.STRONG_BURST_DELAY_THRESHOLD) & (
            loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD
        )

        # 使用公共函数计算连续True序列的最大长度
        from network_simulation.utils.utils import calculate_max_consecutive_true

        max_run = calculate_max_consecutive_true(congested)

        # 检测条件2：整体窗口的平均延迟和丢包率都较高
        delay_mean = np.mean(delays) if len(delays) > 0 else 0
        loss_mean = np.mean(loss_rates) if len(loss_rates) > 0 else 0

        # 添加联合条件，确保强突发检测的准确性
        overall_high = (
            delay_mean
            > self.STRONG_BURST_DELAY_THRESHOLD * 0.8  # 平均延迟接近高延迟阈值
            and loss_mean
            > self.STRONG_BURST_LOSS_THRESHOLD * 0.8  # 平均丢包率接近高丢包率阈值
        )

        # 检测条件3：窗口内有多个高延迟或高丢包的峰值
        high_delay_count = np.sum(delays >= self.STRONG_BURST_DELAY_THRESHOLD * 1.2)
        high_loss_count = np.sum(loss_rates >= self.STRONG_BURST_LOSS_THRESHOLD * 1.2)

        multiple_peaks = high_delay_count >= 5 or high_loss_count >= 5

        # 联合条件：必须同时满足连续高延迟高丢包，且整体较高，且有多个峰值
        # 增强强突发的检测条件，使其更加严格，避免与弱突发混淆
        if max_run >= self.STRONG_BURST_MIN_RUN and overall_high and multiple_peaks:
            return True

        return False

    def _detect_weak_burst(
        self,
        loss_mean: float,
        loss_std: float,
        delay_std: float,
        delays: np.ndarray,
        row: pd.Series,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> bool:
        """检测Weak Burst行为（弱突发）

        Args:
            loss_mean (float): 原始数据计算的丢包率均值
            loss_std (float): 原始数据计算的丢包率标准差
            delay_std (float): 原始数据计算的延迟标准差
            delays (np.ndarray): 原始延迟数据数组
            row (pd.Series): 特征数据行
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 基于原始数据计算的阈值字典

        Returns:
            bool: 是否为Weak Burst行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> import pandas as pd
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> loss_mean = 0.08
            >>> loss_std = 0.05
            >>> delay_std = 15.0
            >>> delays = np.random.normal(50, 10, 100)
            >>> row = pd.Series({})
            >>> available_features = []
            >>> thresholds = {
            ...     'loss_mean_low': 0.01,
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 20.0,
            ...     'loss_std_high': 0.1
            ... }
            >>> is_weak_burst = pattern_identifier._detect_weak_burst(
            ...     loss_mean, loss_std, delay_std, delays, row, available_features, thresholds
            ... )
            >>> print(f"是否为弱突发: {is_weak_burst}")

        """
        # Stable行为条件：低延迟，低丢包，低波动
        # 使用与主检测逻辑完全相同的条件，确保一致性
        delay_mean = np.mean(delays) if len(delays) > 0 else 0
        stable = (
            loss_mean <= thresholds["loss_mean_low"] * 1.3  # 适度严格的丢包率阈值
            and delay_std < thresholds["delay_std_high"] / 2.0  # 适度严格的延迟波动要求
            and loss_std < thresholds["loss_std_high"] / 2.0  # 适度严格的丢包率波动要求
            and delay_mean
            < thresholds["delay_mean_high"] * 0.85  # 适度严格的延迟均值要求
        )

        if stable:
            return False

        # 收集所有满足的条件
        conditions = []

        # 计算延迟均值，用于后续判断
        delay_mean = np.mean(delays) if len(delays) > 0 else 0

        # 排除条件：如果接近强突发的阈值，不应该被判定为弱突发
        # 明确区分弱突发和强突发的边界
        near_strong_burst = (
            loss_mean > self.STRONG_BURST_LOSS_THRESHOLD * 0.6  # 丢包率接近强突发阈值
            or delay_mean
            > self.STRONG_BURST_DELAY_THRESHOLD * 0.6  # 延迟接近强突发阈值
        )

        # 条件1: 中等丢包率，且不接近强突发阈值
        # 调整为更合理的丢包率范围
        conditions.append(
            loss_mean
            >= thresholds["loss_mean_low"] * 1.0  # 适度提高下限，区分STABLE和WEAK_BURST
            and loss_mean < thresholds["loss_mean_high"]
            and not near_strong_burst  # 确保不是接近强突发的情况
        )

        # 条件2: 延迟上升趋势 - 使用简单的斜率计算
        delay_trend = 0.0
        if len(delays) > 5:  # 至少需要6个点才能计算有效趋势
            try:
                # 简化的延迟趋势计算：使用最后5个点和前5个点的平均延迟比较
                # 或者使用线性回归的斜率
                x = np.arange(len(delays))
                # 确保延迟值非负，避免log1p计算错误
                delays_non_negative = np.maximum(delays, 0)
                # 使用log(1+delay)计算趋势
                delays_log1 = np.log1p(delays_non_negative)
                # 使用简单的线性回归计算斜率
                if len(delays_log1) > 1:
                    slope = np.polyfit(x, delays_log1, 1)[0]
                    delay_trend = slope
            except Exception as e:
                logger.debug(f"计算延迟趋势时出错: {e}")
        # 检测延迟上升趋势，斜率为正
        conditions.append(delay_trend > 0.0)  # 降低斜率阈值，允许更轻微的上升趋势

        # 条件3: 延迟波动大，调整为适中的阈值
        conditions.append(delay_std > thresholds["delay_std_high"] / 2.0)

        # 条件4: 丢包率波动大，调整为适中的阈值
        conditions.append(loss_std > thresholds["loss_std_high"] / 2.0)

        # 条件5: 丢包率非零比例高
        if "feat_loss_nonzero_ratio" in available_features:
            loss_nonzero_value = row["feat_loss_nonzero_ratio"]
            # 检查特征是否可能已归一化（值不在0-1范围内）
            if 0 <= loss_nonzero_value <= 1:
                # 原始比例值，使用默认阈值
                conditions.append(
                    loss_nonzero_value > self.WEAK_BURST_LOSS_NONZERO_RATIO
                )
            else:
                # 可能已归一化，跳过此条件，避免误判
                # 归一化后的特征值不能直接与原始阈值比较
                logger.debug(
                    f"检测到归一化的feat_loss_nonzero_ratio值: {loss_nonzero_value}，跳过此条件判断"
                )

        # 动态调整需要满足的条件数量
        # 基于可用条件的数量动态调整，确保检测的灵活性和准确性
        num_conditions = len(conditions)
        if num_conditions == 0:
            return False
        elif num_conditions <= 2:
            # 只有2个或更少条件时，需要满足所有条件
            required_conditions = num_conditions
        elif num_conditions == 3:
            # 3个条件时，需要满足至少2个
            required_conditions = 2
        else:
            # 4个或更多条件时，需要满足至少一半条件
            required_conditions = max(2, num_conditions // 2)

        # 条件满足原则：满足至少required_conditions个条件才判定为Weak Burst
        return sum(conditions) >= required_conditions

    def _detect_instant_spike(self, delays: np.ndarray, loss_rates: np.ndarray) -> bool:
        """检测瞬时峰值行为

        Args:
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组

        Returns:
            bool: 是否为瞬时峰值行为

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = {
            ...     'INSTANT_SPIKE_DELAY_THRESHOLD': 800,
            ...     'INSTANT_SPIKE_LOSS_THRESHOLD': 0.8,
            ...     'INSTANT_SPIKE_MAX_COUNT': 5,
            ...     'INSTANT_SPIKE_MAX_RATIO': 0.1,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 瞬时峰值数据
            >>> delays = np.array([900, 100, 120, 110, 130])  # 只有1个点超过阈值
            >>> loss_rates = np.array([0.9, 0.01, 0.02, 0.01, 0.03])  # 只有1个点超过阈值
            >>> is_instant_spike = pattern_identifier._detect_instant_spike(delays, loss_rates)
            >>> print(f"是否为瞬时峰值: {is_instant_spike}")
            True

        """
        # ≤5个点满足：loss ≥ 0.8或delay ≥ 800ms，且占比小于10%
        instant_spike_count = np.sum(
            (delays >= self.INSTANT_SPIKE_DELAY_THRESHOLD)
            | (loss_rates >= self.INSTANT_SPIKE_LOSS_THRESHOLD)
        )
        total_points = len(delays)
        spike_ratio = instant_spike_count / total_points if total_points > 0 else 0
        return (
            0 < instant_spike_count <= self.INSTANT_SPIKE_MAX_COUNT
            and spike_ratio < self.INSTANT_SPIKE_MAX_RATIO
        )

    def _detect_behavior_with_raw_data(
        self,
        i: int,
        row: pd.Series,
        delays: np.ndarray,
        loss_rates: np.ndarray,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> int:
        """使用原始数据检测行为

        Args:
            i (int): 窗口索引
            row (pd.Series): 特征数据行
            delays (np.ndarray): 延迟数据数组
            loss_rates (np.ndarray): 丢包率数据数组
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典

        Returns:
            int: 行为标签

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> import pandas as pd
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> delays = np.random.normal(50, 10, 100)
            >>> loss_rates = np.random.choice([0, 0.01, 0.05], 100)
            >>> row = pd.Series({})
            >>> available_features = []
            >>> thresholds = {
            ...     'delay_mean_low': 50,
            ...     'delay_mean_high': 200,
            ...     'loss_mean_low': 0.01,
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 100,
            ...     'loss_std_high': 0.05
            ... }
            >>> behavior_label = pattern_identifier._detect_behavior_with_raw_data(
            ...     0, row, delays, loss_rates, available_features, thresholds
            ... )
            >>> print(f"检测到的行为标签: {behavior_label}")

        """
        # 计算窗口统计信息
        stats = self._calculate_window_stats(delays, loss_rates, window_idx=i)

        # 如果窗口无效，标记为INVALID
        if stats is None:
            return BEHAVIOR_LABELS["INVALID"]

        delay_mean, delay_std, loss_mean, loss_std = stats

        # 按行为严重性从高到低检测，确保每种行为都能被正确识别
        # 1. 瞬时峰值 (INSTANT_SPIKE) - 最严重，延迟或丢包达到极值且持续时间极短
        if self._detect_instant_spike(delays, loss_rates):
            return BEHAVIOR_LABELS["INSTANT_SPIKE"]

        # 2. 低延迟高丢包 (LOW_DELAY_HIGH_LOSS) - 延迟低但丢包严重，网络不稳定
        # 添加明确的阈值范围：延迟足够低，丢包率足够高
        if (
            delay_mean < thresholds["delay_mean_low"] * 0.8  # 延迟低于低延迟阈值的80%
            and loss_mean
            > thresholds["loss_mean_high"] * 1.2  # 丢包率高于高丢包率阈值的120%
            and delay_mean > 0  # 确保延迟是正数
            and loss_mean > 0.1  # 确保丢包率至少为10%
        ):
            return BEHAVIOR_LABELS["LOW_DELAY_HIGH_LOSS"]

        # 3. 强突发 (STRONG_BURST) - 高延迟高丢包且持续时间长
        if self._detect_strong_burst(delays, loss_rates):
            return BEHAVIOR_LABELS["STRONG_BURST"]

        # 4. 持续高丢包 (HIGH_LOSS_STEADY) - 持续严重丢包，波动小
        if (
            loss_mean > thresholds["loss_mean_high"]
            and loss_std < thresholds["loss_std_high"]
        ):
            return BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

        # 5. 高延迟无丢包 (HIGH_DELAY_NO_LOSS) - 延迟高但无数据丢失
        if (
            delay_mean > thresholds["delay_mean_high"]
            and loss_mean <= thresholds["loss_mean_low"]
        ):
            return BEHAVIOR_LABELS["HIGH_DELAY_NO_LOSS"]

        # 6. 频繁波动 (FREQUENT_FLUCTUATION) - 延迟和丢包率都有较大波动
        # 使用变异系数作为波动指标，更能反映数据的相对波动程度
        delay_cv = delay_std / delay_mean if delay_mean > 0 else 0
        loss_cv = loss_std / loss_mean if loss_mean > 0 else 0

        # 计算变异系数阈值
        delay_cv_threshold = 0.5  # 变异系数超过50%表示波动较大
        loss_cv_threshold = 1.0  # 丢包率变异系数超过100%表示波动较大

        if (
            delay_cv > delay_cv_threshold or delay_std > thresholds["delay_std_high"]
        ) and (loss_cv > loss_cv_threshold or loss_std > thresholds["loss_std_high"]):
            return BEHAVIOR_LABELS["FREQUENT_FLUCTUATION"]

        # 7. 弱突发 (WEAK_BURST) - 中等丢包率，轻微网络波动
        if self._detect_weak_burst(
            loss_mean, loss_std, delay_std, delays, row, available_features, thresholds
        ):
            return BEHAVIOR_LABELS["WEAK_BURST"]

        # 8. 稳定行为 (STABLE) - 正常网络状态，低延迟，低丢包，低波动
        # 稳定行为条件：优化条件，提高与WEAK_BURST的区分度
        is_stable = (
            loss_mean <= thresholds["loss_mean_low"] * 1.3  # 适度严格的丢包率阈值
            and delay_std < thresholds["delay_std_high"] / 2.0  # 适度严格的延迟波动要求
            and loss_std < thresholds["loss_std_high"] / 2.0  # 适度严格的丢包率波动要求
            and delay_mean
            < thresholds["delay_mean_high"] * 0.85  # 适度严格的延迟均值要求
        )
        if is_stable:
            return BEHAVIOR_LABELS["STABLE"]

        # 默认行为：稳定 (STABLE) - 所有条件都不满足，网络状态稳定
        return BEHAVIOR_LABELS["STABLE"]

    def _calculate_dynamic_thresholds(
        self, features_df: pd.DataFrame, raw_data_df: pd.DataFrame
    ) -> Dict[str, float]:
        """使用配置中的静态阈值，不再从原始数据计算动态阈值

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            raw_data_df (pd.DataFrame): 原始数据DataFrame

        Returns:
            Dict[str, float]: 配置中的静态阈值字典

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_up_delay_mean': np.random.normal(50, 10, 100),
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05], 100)
            ... })
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 1000),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 1000),
            ...     'delay2': np.random.normal(60, 15, 1000),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 1000)
            ... })
            >>> thresholds = pattern_identifier._calculate_dynamic_thresholds(features_df, raw_data_df)
            >>> print(f"使用配置中的静态阈值: {thresholds}")

        """
        # 直接使用配置中的静态阈值
        del features_df, raw_data_df  # 避免未使用参数警告
        thresholds = {
            "delay_mean_low": self.DEFAULT_DELAY_MEAN_LOW,
            "delay_mean_high": self.DEFAULT_DELAY_MEAN_HIGH,
            "delay_std_high": self.DEFAULT_DELAY_STD_HIGH,
            "loss_mean_low": self.DEFAULT_LOSS_MEAN_LOW,
            "loss_mean_high": self.DEFAULT_LOSS_MEAN_HIGH,
            "loss_std_high": self.DEFAULT_LOSS_STD_HIGH,
        }

        logger.debug(f"使用配置中的静态阈值: {thresholds}")
        return thresholds

    def _calculate_thresholds_from_raw_data(
        self, raw_data_df: pd.DataFrame, thresholds: Dict
    ) -> None:
        """从原始数据计算阈值

        Args:
            raw_data_df (pd.DataFrame): 原始数据DataFrame
            thresholds (Dict): 阈值字典，用于存储计算结果

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 1000),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05, 0.1], 1000),
            ...     'delay2': np.random.normal(60, 15, 1000),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05, 0.1], 1000)
            ... })
            >>> thresholds = {}
            >>> pattern_identifier._calculate_thresholds_from_raw_data(raw_data_df, thresholds)
            >>> print(f"从原始数据计算的阈值: {thresholds}")

        """
        # 合并上下行数据用于计算阈值
        delays = np.concatenate(
            [raw_data_df["delay1"].values, raw_data_df["delay2"].values]
        )
        loss_rates = np.concatenate(
            [raw_data_df["loss_rate1"].values, raw_data_df["loss_rate2"].values]
        )

        # 计算分位数作为阈值
        thresholds["delay_mean_low"] = np.percentile(delays, 25)  # 低延迟阈值
        thresholds["delay_mean_high"] = np.percentile(delays, 75)  # 高延迟阈值
        # 调整丢包率阈值，让更多数据被判定为稳定类
        thresholds["loss_mean_low"] = np.percentile(
            loss_rates, 50
        )  # 低丢包率阈值 - 使用50%分位数，让更多数据被认为是低丢包率
        thresholds["loss_mean_high"] = np.percentile(loss_rates, 75)  # 高丢包率阈值

        # 安全计算延迟标准差阈值
        window_size = self.DYNAMIC_THRESHOLD_WINDOW_SIZE
        self._calculate_windowed_std_threshold(
            delays, window_size, "delay_std_high", thresholds
        )

        # 安全计算丢包率标准差阈值
        self._calculate_windowed_std_threshold(
            loss_rates, window_size, "loss_std_high", thresholds
        )

    def _calculate_windowed_std_threshold(
        self, data: np.ndarray, window_size: int, threshold_key: str, thresholds: Dict
    ) -> None:
        """计算窗口标准差阈值

        Args:
            data (np.ndarray): 输入数据数组
            window_size (int): 窗口大小
            threshold_key (str): 阈值字典中的键名
            thresholds (Dict): 阈值字典，用于存储计算结果

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> data = np.random.normal(50, 10, 1000)  # 生成1000个数据点
            >>> thresholds = {}
            >>> pattern_identifier._calculate_windowed_std_threshold(data, 100, "delay_std_high", thresholds)
            >>> print(f"计算的窗口标准差阈值: {thresholds['delay_std_high']}")

        """
        if len(data) >= window_size:
            # 使用滑动窗口计算标准差，设置 min_periods 为窗口大小的一半，避免早期噪声
            rolling_std = (
                pd.Series(data)
                .rolling(window=window_size, min_periods=window_size // 2)
                .std()
                .dropna()
                .values
            )
            thresholds[threshold_key] = np.percentile(
                rolling_std, 50
            )  # 调整为50%分位数，让更多窗口被认为是低波动
        elif len(data) > 1:
            # 数据点不足一个窗口但大于1个，计算整体标准差
            thresholds[threshold_key] = (
                np.std(data) * 0.5
            )  # 调整为整体标准差的一半，让更多窗口被认为是低波动
        else:
            # 数据点过少，使用全局默认阈值
            if threshold_key == "delay_std_high":
                thresholds[threshold_key] = (
                    self.DEFAULT_DELAY_STD_HIGH * 0.5
                )  # 调整为默认值的一半，让更多窗口被认为是低波动
            elif threshold_key == "loss_std_high":
                thresholds[threshold_key] = (
                    self.DEFAULT_LOSS_STD_HIGH * 0.5
                )  # 调整为默认值的一半，让更多窗口被认为是低波动
            else:
                thresholds[threshold_key] = 0.0

    def _set_default_thresholds(self, thresholds: Dict) -> None:
        """设置默认阈值

        Args:
            thresholds (Dict): 阈值字典，用于存储默认值

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = {
            ...     'DEFAULT_DELAY_MEAN_LOW': 50,
            ...     'DEFAULT_DELAY_MEAN_HIGH': 200,
            ...     'DEFAULT_DELAY_STD_HIGH': 100,
            ...     'DEFAULT_LOSS_MEAN_LOW': 0.01,
            ...     'DEFAULT_LOSS_MEAN_HIGH': 0.1,
            ...     'DEFAULT_LOSS_STD_HIGH': 0.05,
            ...     /* 其他配置参数 */
            ... }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> thresholds = {'delay_mean_low': 45}
            >>> pattern_identifier._set_default_thresholds(thresholds)
            >>> print(f"设置默认阈值后: {thresholds}")

        """
        thresholds.setdefault("delay_mean_low", self.DEFAULT_DELAY_MEAN_LOW)
        thresholds.setdefault("delay_mean_high", self.DEFAULT_DELAY_MEAN_HIGH)
        thresholds.setdefault("delay_std_high", self.DEFAULT_DELAY_STD_HIGH)
        thresholds.setdefault("loss_mean_low", self.DEFAULT_LOSS_MEAN_LOW)
        thresholds.setdefault("loss_mean_high", self.DEFAULT_LOSS_MEAN_HIGH)
        thresholds.setdefault("loss_std_high", self.DEFAULT_LOSS_STD_HIGH)

    def _perform_rule_based_detection(
        self,
        features_df: pd.DataFrame,
        raw_data_df: pd.DataFrame,
        thresholds: Dict[str, float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行规则事件检测，识别网络行为模式

        重要说明：
        - 该方法返回上下行独立的行为标签数组，这是因为网络的上行和下行可能表现出不同的行为模式
        - 建议用户分别处理上下行标签，以获得更准确的网络状态分析
        - 如果上下行特征相同，上下行标签也会相同
        - 该方法必须提供raw_data_df，以确保所有8种行为模式都能被正确检测

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame，必须包含window_start和window_end列
            raw_data_df (pd.DataFrame): 原始数据DataFrame，其索引必须是连续的行号[0, 1, 2, ...]
            thresholds (Dict[str, float], optional): 动态计算的阈值字典

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组
            - 第一个数组为上行行为标签
            - 第二个数组为下行行为标签

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'window_start': [0, 50, 100],
            ...     'window_end': [50, 100, 150],
            ...     'feat_delay1_mean': np.random.normal(3.9, 0.2, 3),
            ...     'feat_loss1_mean': np.random.choice([0, 0.01, 0.05], 3),
            ...     'feat_down_delay_mean': np.random.normal(4.0, 0.25, 3),
            ...     'feat_loss2_mean': np.random.choice([0, 0.01, 0.05, 0.1], 3)
            ... })
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 150),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.05], 150),
            ...     'delay2': np.random.normal(60, 15, 150),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 150)
            ... })
            >>> labels_up, labels_down = pattern_identifier._perform_rule_based_detection(features_df, raw_data_df)
            >>> print(f"上行标签数量: {len(labels_up)}, 下行标签数量: {len(labels_down)}")

        """
        if thresholds is None:
            # 如果未提供阈值，重新计算
            thresholds = self._calculate_dynamic_thresholds(features_df, raw_data_df)
        logger.info("开始规则事件检测")

        # 获取可用的特征列
        available_features = features_df.columns.tolist()

        # 执行规则事件检测，仅使用有原始数据的检测方法
        merged_labels = self._detect_with_raw_data(
            features_df, raw_data_df, available_features, thresholds
        )

        logger.info("规则事件检测完成")

        # 使用np.unique替代np.bincount，处理非连续标签
        # 合并标签统计
        unique_labels, counts = np.unique(merged_labels, return_counts=True)
        label_stats = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
        logger.info(f"合并行为标签统计: {label_stats}")

        # 只返回合并标签，不再保持与现有代码的兼容性
        return merged_labels

    def _detect_with_raw_data(
        self,
        features_df: pd.DataFrame,
        raw_data_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """使用原始数据进行行为检测

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            raw_data_df (pd.DataFrame): 原始数据DataFrame
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典

        Returns:
            np.ndarray: 合并后的行为标签数组（取上下行中更严重的行为）

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'window_start': [0, 50, 100],
            ...     'window_end': [50, 100, 150],
            ...     'feat_up_delay_mean': [3.9, 4.2, 3.8],
            ...     'feat_up_loss_mean': [0.01, 0.25, 0.02]
            ... })
            >>> raw_data_df = pd.DataFrame({
            ...     'delay1': np.random.normal(50, 10, 150),
            ...     'loss_rate1': np.random.choice([0, 0.01, 0.25], 150),
            ...     'delay2': np.random.normal(60, 15, 150),
            ...     'loss_rate2': np.random.choice([0, 0.01, 0.05], 150)
            ... })
            >>> merged_labels = pattern_identifier._detect_with_raw_data(
            ...     features_df, raw_data_df, features_df.columns.tolist(), {})
            >>> print(f"合并标签: {merged_labels}")

        """
        # 统一输入检查
        if raw_data_df is None:
            raise ValueError("raw_data_df必须提供，以确保所有行为模式都能被正确检测")

        if raw_data_df.empty:
            raise ValueError("raw_data_df不能为空")

        # 检查raw_data_df是否包含必要的列
        # 支持两种列名格式：
        # 格式1: delay_up, loss_rate_up, delay_down, loss_rate_down
        # 格式2: delay1, loss_rate1, delay2, loss_rate2
        if all(
            col in raw_data_df.columns
            for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"]
        ):
            # 使用新格式（delay1/delay2）
            pass  # 不需要重命名，直接使用
        elif all(
            col in raw_data_df.columns
            for col in ["delay_up", "loss_rate_up", "delay_down", "loss_rate_down"]
        ):
            # 使用旧格式（delay_up/delay_down）
            pass  # 不需要重命名，直接使用
        else:
            # 检查具体缺少哪些列
            required_cols = ["delay_up", "loss_rate_up", "delay_down", "loss_rate_down"]
            missing_cols = [
                col for col in required_cols if col not in raw_data_df.columns
            ]
            raise ValueError(
                f"raw_data_df必须包含以下列之一: [{required_cols}] 或 [delay1, loss_rate1, delay2, loss_rate2]。缺少: {missing_cols}"
            )

        # 初始化上下行行为标签数组，默认为STABLE
        labels_up = np.full(len(features_df), BEHAVIOR_LABELS["STABLE"], dtype=int)
        labels_down = np.full(len(features_df), BEHAVIOR_LABELS["STABLE"], dtype=int)

        # 假设所有数据都包含完整的上下行数据
        logger.debug("检测到完整的上下行数据")

        # 有原始数据时，需要逐窗口处理原始数据
        for i, (_, row) in enumerate(features_df.iterrows()):
            # 获取当前窗口的原始数据，用于计算更详细的统计信息
            window_start = int(row["window_start"])
            window_end = int(row["window_end"])

            # 检查窗口有效性
            if window_start >= window_end or window_start >= len(raw_data_df):
                # 只打印一次警告，避免大量重复日志
                if i == 0 or np.random.rand() < 0.01:  # 1%的概率打印警告
                    logger.warning(
                        f"窗口 {i} 无效: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}"
                    )
                labels_up[i] = BEHAVIOR_LABELS["INVALID"]
                labels_down[i] = BEHAVIOR_LABELS["INVALID"]
                continue  # 直接跳过无效窗口，避免冗余计算

            # 如果window_end超过raw_data_df的长度，动态调整
            if window_end > len(raw_data_df):
                # 只打印一次警告，避免大量重复日志
                if i == 0 or np.random.rand() < 0.01:  # 1%的概率打印警告
                    logger.warning(
                        f"窗口 {i} 超出原始数据范围，动态调整: window_start={window_start}, window_end={window_end}, raw_data_length={len(raw_data_df)}"
                    )
                # 调整window_end为raw_data_df的长度
                window_end = len(raw_data_df)
                # 如果调整后window_start >= window_end，跳过
                if window_start >= window_end:
                    labels_up[i] = BEHAVIOR_LABELS["INVALID"]
                    labels_down[i] = BEHAVIOR_LABELS["INVALID"]
                    continue

            # 提取窗口数据
            window_data = raw_data_df.iloc[window_start:window_end]

            # 上行数据
            if all(col in window_data.columns for col in ["delay1", "loss_rate1"]):
                # 使用新格式（delay1/delay2）
                delay_up = window_data["delay1"].values
                loss_rate_up = window_data["loss_rate1"].values
                delay_down = window_data["delay2"].values
                loss_rate_down = window_data["loss_rate2"].values
            else:
                # 使用旧格式（delay_up/delay_down）
                delay_up = window_data["delay_up"].values
                loss_rate_up = window_data["loss_rate_up"].values
                delay_down = window_data["delay_down"].values
                loss_rate_down = window_data["loss_rate_down"].values

            # 检查窗口数据是否为空
            if (
                len(delay_up) == 0
                or len(loss_rate_up) == 0
                or len(delay_down) == 0
                or len(loss_rate_down) == 0
            ):
                logger.warning(f"窗口 {i} 数据为空，标记为无效窗口")
                labels_up[i] = BEHAVIOR_LABELS["INVALID"]
                labels_down[i] = BEHAVIOR_LABELS["INVALID"]
                continue  # 直接跳过空数据窗口

            # 对上下行数据分别检测行为
            # 上行行为
            behavior1 = self._detect_behavior_with_raw_data(
                i, row, delay_up, loss_rate_up, available_features, thresholds
            )

            # 下行行为
            behavior2 = self._detect_behavior_with_raw_data(
                i, row, delay_down, loss_rate_down, available_features, thresholds
            )

            # 分别存储上下行行为标签
            labels_up[i] = behavior1
            labels_down[i] = behavior2

        # 合并上下行标签，取更严重的行为
        merged_labels = np.copy(labels_up)
        for i in range(len(merged_labels)):
            # 选择更严重的行为标签（数值更大表示更严重）
            merged_labels[i] = max(labels_up[i], labels_down[i])
        return merged_labels

    def _detect_without_raw_data(
        self,
        features_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
    ) -> np.ndarray:
        """不使用原始数据进行行为检测（向量化操作）

        该函数使用向量化操作，基于提取的特征直接进行行为检测，无需访问原始数据。
        支持上下行数据的单独检测和综合分析。

        Args:
            features_df (pd.DataFrame): 包含特征数据的DataFrame
            available_features (list): 可用特征列表
            thresholds (Dict[str, float]): 动态计算的阈值字典，用于行为检测

        Returns:
            tuple[np.ndarray, np.ndarray]: 上行行为标签数组和下行行为标签数组

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> import pandas as pd
            >>> import numpy as np
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> features_df = pd.DataFrame({
            ...     'feat_loss1_mean': [0.01, 0.25, 0.02],
            ...     'feat_delay1_std': [0.1, 0.3, 0.15],
            ...     'feat_loss1_std': [0.005, 0.05, 0.01],
            ...     'feat_max_congestion_run1': [0, 15, 0],
            ...     'feat_loss2_mean': [0.02, 0.03, 0.15],
            ...     'feat_delay2_std': [0.12, 0.18, 0.25],
            ...     'feat_loss2_std': [0.01, 0.015, 0.03],
            ...     'feat_max_congestion_run2': [0, 0, 8]
            ... })
            >>> available_features = features_df.columns.tolist()
            >>> thresholds = {
            ...     'loss_mean_high': 0.1,
            ...     'delay_std_high': 0.2,
            ...     'loss_std_high': 0.05
            ... }
            >>> merged_labels = pattern_identifier._detect_without_raw_data(features_df, available_features, thresholds)
            >>> print(f"合并标签: {merged_labels}")

        """
        logger.info("无原始数据，使用向量化操作进行批量行为检测")

        # 初始化上下行行为标签
        labels_up = np.full(len(features_df), BEHAVIOR_LABELS["STABLE"], dtype=int)
        labels_down = np.full(len(features_df), BEHAVIOR_LABELS["STABLE"], dtype=int)

        # 检测上行行为
        if self._has_up_features(available_features):
            logger.debug("检测到上行特征，开始上行行为检测")
            # 首先检测稳定类 (STABLE) - 正常网络状态，优先级最高
            stable_mask1 = np.ones(len(features_df), dtype=bool)

            # 条件1: 低丢包率
            loss_mean_col = "feat_up_loss_mean"
            if loss_mean_col in available_features:
                stable_mask1 &= (
                    features_df[loss_mean_col] <= thresholds["loss_mean_low"] * 2
                )  # 放宽丢包率阈值

            # 条件2: 低延迟波动
            delay_std_col = "feat_up_delay_std"
            if delay_std_col in available_features:
                stable_mask1 &= (
                    features_df[delay_std_col] < thresholds["delay_std_high"] / 2
                )

            # 条件3: 低丢包率波动
            loss_std_col = "feat_up_loss_std"
            if loss_std_col in available_features:
                stable_mask1 &= (
                    features_df[loss_std_col] < thresholds["loss_std_high"] / 2
                )

            # 条件4: 无Strong Burst
            congestion_col = "feat_up_max_congestion_run"
            if congestion_col in available_features:
                stable_mask1 &= features_df[congestion_col] < self.STRONG_BURST_MIN_RUN

            # 设置稳定类标签
            labels_up[stable_mask1] = BEHAVIOR_LABELS["STABLE"]

            # 只在非稳定窗口上检测其他行为
            non_stable_mask1 = ~stable_mask1

            # 检测行为 5: 持续高丢包
            if "feat_up_loss_mean" in available_features:
                high_loss_mask1 = (
                    features_df["feat_up_loss_mean"] > thresholds["loss_mean_high"]
                ) & non_stable_mask1
                labels_up[high_loss_mask1] = BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 更新非稳定窗口掩码
            non_stable_mask1 = labels_up == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask1 = ~non_stable_mask1

            # 检测行为 6: 频繁波动
            if (
                "feat_up_delay_std" in available_features
                and "feat_up_loss_std" in available_features
            ):
                frequent_fluctuation_mask1 = (
                    (features_df["feat_up_delay_std"] > thresholds["delay_std_high"])
                    & (features_df["feat_up_loss_std"] > thresholds["loss_std_high"])
                    & non_stable_mask1
                )
                labels_up[frequent_fluctuation_mask1] = BEHAVIOR_LABELS[
                    "FREQUENT_FLUCTUATION"
                ]

            # 更新非稳定窗口掩码
            non_stable_mask1 = labels_up == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask1 = ~non_stable_mask1

            # 检测行为 7: 低延迟高丢包 - 只使用原始尺度延迟均值
            # 基于绝对延迟值的行为判断必须使用原始尺度
            if (
                "feat_delay1_raw_mean" in available_features
                and "feat_loss1_mean" in available_features
            ):
                low_delay_high_loss_mask1 = (
                    (features_df["feat_delay1_raw_mean"] < thresholds["delay_mean_low"])
                    & (features_df["feat_loss1_mean"] > thresholds["loss_mean_high"])
                    & non_stable_mask1
                )
                labels_up[low_delay_high_loss_mask1] = BEHAVIOR_LABELS[
                    "LOW_DELAY_HIGH_LOSS"
                ]

            # 更新非稳定窗口掩码
            non_stable_mask1 = labels_up == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask1 = ~non_stable_mask1

            # 检测行为 2: Strong Burst
            if "feat_max_congestion_run1" in available_features:
                strong_burst_mask1 = (
                    features_df["feat_max_congestion_run1"] >= self.STRONG_BURST_MIN_RUN
                ) & non_stable_mask1
                labels_up[strong_burst_mask1] = BEHAVIOR_LABELS["STRONG_BURST"]

            # 更新非稳定窗口掩码
            non_stable_mask1 = labels_up == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask1 = ~non_stable_mask1

            # 检测行为 1: Weak Burst
            weak_burst_mask1 = (
                self._detect_weak_burst_without_raw_data(
                    features_df, available_features, thresholds, direction="up"
                )
                & non_stable_mask1
            )
            labels_up[weak_burst_mask1] = BEHAVIOR_LABELS["WEAK_BURST"]
        else:
            logger.warning("未检测到完整的上行特征，跳过上行行为检测")

        # 检测下行行为
        if self._has_down_features(available_features):
            logger.debug("检测到下行特征，开始下行行为检测")
            # 首先检测稳定类 (STABLE) - 正常网络状态，优先级最高
            stable_mask2 = np.ones(len(features_df), dtype=bool)

            # 条件1: 低丢包率
            loss_mean_col = "feat_down_loss_mean"
            if loss_mean_col in available_features:
                stable_mask2 &= (
                    features_df[loss_mean_col] <= thresholds["loss_mean_low"] * 2
                )  # 放宽丢包率阈值

            # 条件2: 低延迟波动
            delay_std_col = "feat_down_delay_std"
            if delay_std_col in available_features:
                stable_mask2 &= (
                    features_df[delay_std_col] < thresholds["delay_std_high"] / 2
                )

            # 条件3: 低丢包率波动
            loss_std_col = "feat_down_loss_std"
            if loss_std_col in available_features:
                stable_mask2 &= (
                    features_df[loss_std_col] < thresholds["loss_std_high"] / 2
                )

            # 条件4: 无Strong Burst
            congestion_col = "feat_down_max_congestion_run"
            if congestion_col in available_features:
                stable_mask2 &= features_df[congestion_col] < self.STRONG_BURST_MIN_RUN

            # 设置稳定类标签
            labels_down[stable_mask2] = BEHAVIOR_LABELS["STABLE"]

            # 只在非稳定窗口上检测其他行为
            non_stable_mask2 = ~stable_mask2

            # 检测行为 5: 持续高丢包
            if "feat_down_loss_mean" in available_features:
                high_loss_mask2 = (
                    features_df["feat_down_loss_mean"] > thresholds["loss_mean_high"]
                ) & non_stable_mask2
                labels_down[high_loss_mask2] = BEHAVIOR_LABELS["HIGH_LOSS_STEADY"]

            # 更新非稳定窗口掩码
            non_stable_mask2 = labels_down == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask2 = ~non_stable_mask2

            # 检测行为 6: 频繁波动
            if (
                "feat_down_delay_std" in available_features
                and "feat_down_loss_std" in available_features
            ):
                frequent_fluctuation_mask2 = (
                    (features_df["feat_down_delay_std"] > thresholds["delay_std_high"])
                    & (features_df["feat_down_loss_std"] > thresholds["loss_std_high"])
                    & non_stable_mask2
                )
                labels_down[frequent_fluctuation_mask2] = BEHAVIOR_LABELS[
                    "FREQUENT_FLUCTUATION"
                ]

            # 更新非稳定窗口掩码
            non_stable_mask2 = labels_down == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask2 = ~non_stable_mask2

            # 检测行为 7: 低延迟高丢包 - 只使用原始尺度延迟均值
            # 基于绝对延迟值的行为判断必须使用原始尺度
            if (
                "feat_down_delay_raw_mean" in available_features
                and "feat_down_loss_mean" in available_features
            ):
                low_delay_high_loss_mask2 = (
                    (
                        features_df["feat_down_delay_raw_mean"]
                        < thresholds["delay_mean_low"]
                    )
                    & (
                        features_df["feat_down_loss_mean"]
                        > thresholds["loss_mean_high"]
                    )
                    & non_stable_mask2
                )
                labels_down[low_delay_high_loss_mask2] = BEHAVIOR_LABELS[
                    "LOW_DELAY_HIGH_LOSS"
                ]

            # 更新非稳定窗口掩码
            non_stable_mask2 = labels_down == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask2 = ~non_stable_mask2

            # 检测行为 2: Strong Burst
            if "feat_down_max_congestion_run" in available_features:
                strong_burst_mask2 = (
                    features_df["feat_down_max_congestion_run"]
                    >= self.STRONG_BURST_MIN_RUN
                ) & non_stable_mask2
                labels_down[strong_burst_mask2] = BEHAVIOR_LABELS["STRONG_BURST"]

            # 更新非稳定窗口掩码
            non_stable_mask2 = labels_down == BEHAVIOR_LABELS["STABLE"]
            non_stable_mask2 = ~non_stable_mask2

            # 检测行为 1: Weak Burst
            weak_burst_mask2 = (
                self._detect_weak_burst_without_raw_data(
                    features_df, available_features, thresholds, direction="down"
                )
                & non_stable_mask2
            )
            labels_down[weak_burst_mask2] = BEHAVIOR_LABELS["WEAK_BURST"]
        else:
            logger.warning("未检测到完整的下行特征，跳过下行行为检测")

        return labels_up, labels_down

    def _has_up_features(self, available_features: list) -> bool:
        """检查是否存在足够的上行特征用于行为检测

        Args:
            available_features (list): 可用特征列表

        Returns:
            bool: 是否存在足够的上行特征

        Examples:
            >>> from network_simulation.pattern_discovery.pattern_identifier import PatternIdentifier
            >>> config = { /* 配置参数 */ }
            >>> pattern_identifier = PatternIdentifier(config=config)
            >>> # 有上行特征
            >>> available_features = ['feat_up_delay_mean', 'feat_up_loss_mean', 'feat_down_delay_std']
            >>> has_up = pattern_identifier._has_up_features(available_features)
            >>> print(f"是否有上行特征: {has_up}")  # 是
            >>> # 没有上行特征
            >>> available_features = ['feat_down_delay_mean', 'feat_down_loss_mean']
            >>> has_up = pattern_identifier._has_up_features(available_features)
            >>> print(f"是否有上行特征: {has_up}")  # 否

        """
        # 检查是否至少存在一个上行特征
        up_features = [
            col
            for col in available_features
            if "1" in col and ("delay" in col or "loss" in col)
        ]
        return len(up_features) > 0

    def _has_down_features(self, available_features: list) -> bool:
        """检查是否存在足够的下行特征用于行为检测

        Args:
            available_features: 可用特征列表

        Returns:
            bool: 是否存在足够的下行特征

        """
        # 检查是否至少存在一个下行特征
        down_features = [
            col
            for col in available_features
            if "2" in col and ("delay" in col or "loss" in col)
        ]
        return len(down_features) > 0

    def _detect_weak_burst_without_raw_data(
        self,
        features_df: pd.DataFrame,
        available_features: list,
        thresholds: Dict[str, float],
        direction: str = None,
    ) -> np.ndarray:
        """无原始数据时检测Weak Burst行为

        Args:
            features_df: 包含特征数据的DataFrame
            available_features: 可用特征列表
            thresholds: 动态计算的阈值字典
            direction: 方向，"up"表示上行，"down"表示下行，None表示单通道

        Returns:
            np.ndarray: Weak Burst行为的掩码数组

        """
        # 确定使用的特征后缀
        suffix = "" if direction is None else "1" if direction == "up" else "2"

        # 获取特征列名
        loss_mean_col = f"feat_loss{suffix}_mean"
        delay_std_col = f"feat_delay{suffix}_std"
        loss_std_col = f"feat_loss{suffix}_std"
        loss_nonzero_col = f"feat_loss{suffix}_nonzero_ratio"
        delay_trend_col = f"feat_delay{suffix}_trend"

        # Stable行为条件：低延迟，低丢包，低波动
        # 使用与主检测逻辑相同的宽松条件，确保稳定类成为多数类
        stable_mask = np.ones(len(features_df), dtype=bool)
        if loss_mean_col in available_features:
            stable_mask &= (
                features_df[loss_mean_col] <= thresholds["loss_mean_low"] * 2
            )  # 放宽丢包率阈值
        if delay_std_col in available_features:
            stable_mask &= features_df[delay_std_col] < thresholds["delay_std_high"] / 2
        if loss_std_col in available_features:
            stable_mask &= features_df[loss_std_col] < thresholds["loss_std_high"] / 2

        # 收集所有满足的条件，采用与带原始数据版本一致的条件顺序和判定原则
        weak_burst_conditions = []

        # 条件1: 中等丢包率
        if loss_mean_col in available_features:
            condition1 = (features_df[loss_mean_col] >= thresholds["loss_mean_low"]) & (
                features_df[loss_mean_col] < thresholds["loss_mean_high"]
            )
            weak_burst_conditions.append(condition1)

        # 条件2: 延迟上升趋势
        if delay_trend_col in available_features:
            condition2 = features_df[delay_trend_col] > 0.0
            weak_burst_conditions.append(condition2)

        # 条件3: 丢包率波动大
        if loss_std_col in available_features:
            condition3 = features_df[loss_std_col] > thresholds["loss_std_high"] / 2
            weak_burst_conditions.append(condition3)

        # 条件4: 延迟波动大
        if delay_std_col in available_features:
            condition4 = features_df[delay_std_col] > thresholds["delay_std_high"] / 2
            weak_burst_conditions.append(condition4)

        # 条件5: 丢包率非零比例高
        if loss_nonzero_col in available_features:
            condition5 = (
                features_df[loss_nonzero_col] > self.WEAK_BURST_LOSS_NONZERO_RATIO
            )
            weak_burst_conditions.append(condition5)

        # 应用多数条件满足原则（至少2个条件），并且排除Stable窗口
        if weak_burst_conditions:
            return (
                np.sum(weak_burst_conditions, axis=0) >= self.WEAK_BURST_MIN_CONDITIONS
            ) & (~stable_mask)
        else:
            return np.zeros(len(features_df), dtype=bool)

    def _calculate_transition_metrics(self, transition_matrix: np.ndarray) -> Dict:
        """计算行为转移质量指标

        Args:
            transition_matrix: 行为转移矩阵

        Returns:
            Dict: 包含转移熵、转移稀疏性和典型路径的字典

        """
        metrics = {}

        # 平均转移熵
        entropy = 0.0
        for row in transition_matrix:
            # 移除零概率值以避免log(0)
            row = row[row > 0]
            if len(row) > 0:
                entropy -= np.sum(row * np.log2(row))
        metrics["average_transition_entropy"] = float(entropy / len(transition_matrix))

        # 转移稀疏性
        total_elements = transition_matrix.size
        non_zero_elements = np.count_nonzero(transition_matrix)
        metrics["transition_sparsity"] = float(non_zero_elements / total_elements)

        # 典型路径分析 - 找出最可能的转移路径
        # 对于每个状态，找出前3个最可能的下一个状态，跳过全零行
        typical_paths = {}
        num_states = transition_matrix.shape[0]
        for i in range(num_states):
            # 检查该行是否全为零
            if np.all(transition_matrix[i, :] == 0):
                continue  # 跳过全零行

            # 获取前3个最可能的下一个状态
            top_indices = np.argsort(transition_matrix[i, :])[::-1][:3]
            top_probs = transition_matrix[i, top_indices]
            typical_paths[str(i)] = {
                "next_states": [int(idx) for idx in top_indices],
                "probabilities": [float(prob) for prob in top_probs],
            }
        metrics["typical_paths"] = typical_paths

        return metrics

    def _calculate_separation_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """计算行为分离度指标

        Args:
            X: 特征数据
            labels: 行为标签

        Returns:
            Dict: 包含各类别特征均值和标准差的字典

        """
        unique_labels = np.unique(labels)
        separation_metrics = {"label_means": {}, "label_stds": {}}

        for label in unique_labels:
            cluster_data = X[labels == label]
            separation_metrics["label_means"][str(label)] = cluster_data.mean(
                axis=0
            ).tolist()
            separation_metrics["label_stds"][str(label)] = cluster_data.std(
                axis=0
            ).tolist()

        return separation_metrics

    def identify_and_save(
        self,
        input_features_path: Path,
        input_processed_path: Path,
        output_patterns_dir: Path,
    ) -> Dict:
        """从特征数据中识别行为模式并保存结果

        Args:
            input_features_path: 特征数据文件路径
            input_processed_path: 处理后的数据文件路径
            output_patterns_dir: 行为模式结果的输出目录路径

        Returns:
            Dict: 识别结果字典

        """
        logger.info(
            f"开始识别行为模式并保存结果，输入特征: {input_features_path}, 输入处理数据: {input_processed_path}, 输出: {output_patterns_dir}"
        )

        # 加载特征数据
        features_df = pd.read_csv(input_features_path)

        # 加载处理后的数据
        processed_df = pd.read_csv(input_processed_path, parse_dates=["timestamp"])

        # 识别行为模式
        patterns = self.identify(features_df, processed_df)

        # 创建输出目录结构
        output_patterns_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir = output_patterns_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # 保存模式识别结果
        self.save(patterns, output_patterns_dir)

        logger.info("行为识别和结果保存完成")

        logger.info(f"行为模式已保存到: {output_patterns_dir}")
        return patterns

    def _calculate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """计算状态转移矩阵

        Args:
            labels: 行为标签数组

        Returns:
            np.ndarray: 归一化的状态转移矩阵，只包含有效行为标签

        """
        # 过滤掉INVALID标签，只保留有效标签
        valid_mask = labels != BEHAVIOR_LABELS["INVALID"]
        valid_labels = labels[valid_mask]

        # 显式定义状态空间：所有有效行为标签（排除INVALID）
        # 使用列表推导式获取所有非INVALID的标签值
        valid_behavior_ids = [v for k, v in BEHAVIOR_LABELS.items() if k != "INVALID"]
        # 按顺序排序，确保一致性
        valid_behavior_ids.sort()
        num_clusters = len(valid_behavior_ids)

        # 创建从标签值到矩阵索引的映射
        label_to_index = {lbl: i for i, lbl in enumerate(valid_behavior_ids)}

        transition_matrix = np.zeros((num_clusters, num_clusters))

        # 只计算有效标签之间的转移
        if len(valid_labels) > 1:
            for i in range(len(valid_labels) - 1):
                current_state = valid_labels[i]
                next_state = valid_labels[i + 1]
                # 确保当前状态和下一状态都在有效行为标签中
                if (
                    current_state in valid_behavior_ids
                    and next_state in valid_behavior_ids
                ):
                    # 将标签映射到索引
                    current_idx = label_to_index[current_state]
                    next_idx = label_to_index[next_state]
                    transition_matrix[current_idx, next_idx] += 1

        # 归一化行以获取概率
        row_sums = transition_matrix.sum(axis=1, keepdims=True)

        # 处理行和为0的情况：保持全零行不变，不引入虚假转移概率
        # 全零行表示该状态不会转移到任何其他状态（吸收态）
        for i in range(len(row_sums)):
            if row_sums[i, 0] != 0:
                transition_matrix[i, :] = transition_matrix[i, :] / row_sums[i, 0]

        return transition_matrix

    def _calculate_behavior_statistics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """计算每个行为的统计信息

        Args:
            X: 特征数据
            labels: 行为标签

        Returns:
            Dict: 包含每个行为统计信息的字典

        """
        behavior_stats = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            behavior_data = X[labels == label]
            behavior_stats[str(label)] = {
                "mean": behavior_data.mean(axis=0).tolist(),
                "std": behavior_data.std(axis=0).tolist(),
                "min": behavior_data.min(axis=0).tolist(),
                "max": behavior_data.max(axis=0).tolist(),
                "count": int(len(behavior_data)),
            }

        return behavior_stats

    def save(
        self,
        results: Dict,
        output_dir: Path,
        valid_loss_values: list = None,
        save_all_columns: bool = True,
    ) -> None:
        """保存规则检测结果到文件

        该函数将行为检测的结果保存到指定目录，包括标签文件、行为统计信息、转移矩阵等。
        支持多种检测方法的结果保存。

        Args:
            results: 检测结果字典，包含labels_up、labels_down、method、behavior_stats等字段
            output_dir: 输出目录，用于保存所有结果文件
            valid_loss_values: 合法丢包值列表，可选，用于覆盖自动推导的值
                - 预期格式：[0.0, 0.1, 0.2, ..., 1.0]，表示网络中可能出现的所有丢包率值
                - 使用场景：当自动推导的丢包值不完整或不准确时，可手动指定
                - 示例：valid_loss_values=[0.0, 0.5, 1.0] 表示网络中只有0%、50%和100%三种丢包率
            save_all_columns: 是否保存所有列，包括timestamp等非数值列

                - 设置为False时，只保存数值列（delay*、loss_rate*、bandwidth*），节省存储空间

        保存的文件包括：
            - window_features_rule.npy: 窗口特征矩阵
            - behavior_statistics_up.json: 上行行为统计信息
            - behavior_statistics_down.json: 下行行为统计信息
            - transition_matrix_rule_up.json: 上行行为转移矩阵
            - transition_matrix_rule_down.json: 下行行为转移矩阵
            - valid_loss_values.json: 合并的上下行合法丢包值列表
            - behavior_{behavior_id}_windows.csv: 按行为类别合并的窗口数据文件

        """
        output_dir.mkdir(parents=True, exist_ok=True)

        del save_all_columns  # 避免未使用参数警告
        method = results["method"]

        # 保存完整的patterns字典为JSON文件，以便可视化使用
        patterns_file = output_dir / "patterns.json"
        with open(patterns_file, "w") as f:
            import json

            json.dump(results, f, indent=2)
        logger.info(f"保存行为识别结果: {patterns_file}")

        # 使用合并后的标签
        labels = np.array(results["labels"])

        # 保存窗口特征矩阵
        if method == "rule":
            if "features" in results and results["features"]:
                features = np.array(results["features"])
                try:
                    feat_mean = np.mean(features, axis=0)
                    feat_std = np.std(features, axis=0)
                    low_var_idx = [int(i) for i, s in enumerate(feat_std) if s < 1e-6]
                    logger.info("=== 窗口特征矩阵分布统计（逐维） ===")
                    logger.info(f"  形状: {features.shape}")
                    logger.info(f"  前10维均值: {feat_mean[:10]}")
                    logger.info(f"  前10维标准差: {feat_std[:10]}")
                    if low_var_idx:
                        logger.warning(
                            f"  低方差维度（std<1e-6）数量: {len(low_var_idx)}，示例: {low_var_idx[:20]}"
                        )
                    else:
                        logger.info("  未检测到低方差维度（std<1e-6）")
                    logger.info("=== 特征样本快照（前3行） ===")
                    logger.info(f"{features[:3]}")
                except Exception as e:
                    logger.warning(f"窗口特征矩阵分布统计日志输出失败: {str(e)}")
                features_file = output_dir / "window_features_rule.npy"
                np.save(features_file, features)
                logger.info(f"保存窗口特征矩阵: {features_file}")
                # 生成CSV格式
                features_df = pd.DataFrame(
                    features, columns=[f"feature_{i}" for i in range(features.shape[1])]
                )
                features_csv_file = output_dir / "window_features_rule.csv"
                features_df.to_csv(features_csv_file, index=False)
                logger.info(f"保存窗口特征矩阵为CSV文件: {features_csv_file}")

                # 生成并保存行为条件映射文件

                # 计算行为条件映射（基于合并标签）
                behavior_condition_map = {}
                labels = np.array(results["labels"])
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    if label == BEHAVIOR_LABELS["INVALID"]:
                        continue
                    # 收集包含该行为的窗口索引
                    window_indices = [
                        i for i, label_i in enumerate(labels) if label_i == label
                    ]
                    if window_indices:
                        all_features_for_label = features[window_indices]
                        avg_feature = np.mean(all_features_for_label, axis=0)
                        behavior_condition_map[int(label)] = avg_feature

                # 保存行为条件映射文件
                condition_map_file = output_dir / "behavior_condition_map.npy"
                np.save(condition_map_file, behavior_condition_map)
                logger.info(f"保存行为条件映射文件: {condition_map_file}")
                logger.info(f"行为条件映射包含 {len(behavior_condition_map)} 个行为ID")
                # 生成CSV格式
                if behavior_condition_map:
                    # 转换字典为DataFrame
                    condition_map_df = pd.DataFrame.from_dict(
                        behavior_condition_map, orient="index"
                    )
                    condition_map_df.index.name = "behavior_id"
                    # 使用实际的特征列名，而不是硬编码的feature_i
                    if not hasattr(self, "feature_columns") or not self.feature_columns:
                        raise ValueError("特征列名不存在，无法生成行为条件映射文件")
                    condition_map_df.columns = self.feature_columns
                    condition_map_csv_file = output_dir / "behavior_condition_map.csv"
                    condition_map_df.to_csv(condition_map_csv_file)
                    logger.info(f"保存行为条件映射为CSV文件: {condition_map_csv_file}")

        # 保存行为统计信息
        behavior_stats_file = output_dir / "behavior_statistics.json"
        with open(behavior_stats_file, "w") as f:
            json.dump(results["behavior_stats"], f, indent=2)
        logger.info(f"保存行为统计信息: {behavior_stats_file}")

        # 保存转移矩阵
        with open(output_dir / f"behavior_transition_graph_{method}.json", "w") as f:
            json.dump(
                {
                    "transition_matrix": results["transition_matrix"],
                    "num_behaviors": len(np.unique(results["labels"])),
                    "method": method,
                },
                f,
                indent=2,
            )
            # 构建完整的文件路径
            transition_graph_file = (
                output_dir / f"behavior_transition_graph_{method}.json"
            )
            logger.info(f"保存转移矩阵: {transition_graph_file}")

        # 自动从raw_data_df中推导合法丢包值并保存到metadata目录
        if "raw_data" in results and results["raw_data"] is not None:
            raw_data_df = results["raw_data"]
            # 使用工具函数提取合法丢包值
            from network_simulation.utils.utils import (
                extract_directional_valid_loss_values,
            )

            valid_loss_values_up, valid_loss_values_down, valid_loss_values = (
                extract_directional_valid_loss_values(raw_data_df)
            )

            if valid_loss_values is not None:
                metadata_dir = output_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                # 保存合并的合法丢包值
                valid_loss_file = metadata_dir / "valid_loss_values.json"
                with open(valid_loss_file, "w") as f:
                    json.dump(valid_loss_values, f, indent=2)
                logger.info(f"保存合并的合法丢包值: {valid_loss_file}")

        # 不再生成按行为类别保存的原始数据片段
        # 用户要求不再生成behavior_*_windows.csv文件
