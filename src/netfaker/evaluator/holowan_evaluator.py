"""HoloWAN 合成数据评估主引擎。

负责：
1. 加载真实与合成 HoloWAN 文件
2. 按窗口一一对应切分数据
3. 调用各项指标评估函数（窗口级+序列级）
4. 聚合评估结果
5. 输出评估报告
"""

import os
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from netfaker.evaluator.config import EvaluationConfig
from netfaker.evaluator.metrics.burst import BurstMetrics
from netfaker.evaluator.metrics.joint import JointMetrics
from netfaker.evaluator.metrics.spectral import SpectralMetrics
from netfaker.evaluator.metrics.statistical import StatisticalMetrics
from netfaker.evaluator.metrics.temporal import TemporalMetrics
from netfaker.evaluator.metrics.sequence import (
    LongTermACFMetrics,
    StateTransitionMetrics,
    RollingStatsMetrics,
    BurstTimingMetrics
)
from netfaker.evaluator.reporter import EvaluationReporter
from netfaker.simcore.io.holowan_loader import HoloWANLoader


class HoloWANEvaluator:
    """HoloWAN 合成数据评估主类。
    
    实现真实与合成 HoloWAN 文件的窗口级评估，
    支持多维度指标计算和结构化报告生成。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化评估器。
        
        Args:
            config_path: 配置文件路径，默认使用内置配置
        """
        self.config = EvaluationConfig(config_path)
        self.loader = HoloWANLoader()
        self.window_size = 100  # 10秒/100点

        # 初始化各项指标计算器
        # 窗口级指标
        self.stat_metrics = StatisticalMetrics()
        self.temporal_metrics = TemporalMetrics()
        self.spectral_metrics = SpectralMetrics()
        self.burst_metrics = BurstMetrics()
        self.joint_metrics = JointMetrics()
        
        # 序列级指标
        self.long_term_acf_metrics = LongTermACFMetrics()
        self.state_transition_metrics = StateTransitionMetrics()
        self.rolling_stats_metrics = RollingStatsMetrics()
        self.burst_timing_metrics = BurstTimingMetrics()

    def evaluate(self, real_file: str, syn_file: str, output_dir: str) -> Dict:
        """执行评估流程。
        
        Args:
            real_file: 真实 HoloWAN 文件路径
            syn_file: 合成 HoloWAN 文件路径
            output_dir: 输出目录路径
            
        Returns:
            Dict: 评估结果汇总
        """
        logger.info(f"开始评估：真实文件={real_file}，合成文件={syn_file}")

        # 1. 加载文件
        real_df = self.loader.load_file(real_file)
        syn_df = self.loader.load_file(syn_file)

        if real_df is None or real_df.empty:
            logger.error(f"真实文件加载失败或为空：{real_file}")
            raise ValueError(f"真实文件加载失败或为空：{real_file}")

        if syn_df is None or syn_df.empty:
            logger.error(f"合成文件加载失败或为空：{syn_file}")
            raise ValueError(f"合成文件加载失败或为空：{syn_file}")

        # 2. 验证数据长度
        real_df, syn_df = self._validate_and_truncate(real_df, syn_df)

        # 3. 切分窗口
        real_windows = self._split_into_windows(real_df)
        syn_windows = self._split_into_windows(syn_df)

        if len(real_windows) != len(syn_windows):
            logger.error(f"窗口数量不匹配：真实={len(real_windows)}，合成={len(syn_windows)}")
            raise ValueError(f"窗口数量不匹配：真实={len(real_windows)}，合成={len(syn_windows)}")

        logger.info(f"成功切分 {len(real_windows)} 个窗口")

        # 4. 逐窗口评估
        all_metrics = []
        for window_id, (real_window, syn_window) in enumerate(zip(real_windows, syn_windows, strict=True)):
            logger.debug(f"评估窗口 {window_id}")

            # 提取通道数据
            real_delay_up = real_window['raw_delay_up'].values
            syn_delay_up = syn_window['raw_delay_up'].values
            real_loss_up = real_window['raw_loss_up'].values
            syn_loss_up = syn_window['raw_loss_up'].values

            # 计算各项指标
            metrics = {
                'window_id': window_id,
                'real_file': os.path.basename(real_file),
                'syn_file': os.path.basename(syn_file),
            }

            # 统计分布指标
            metrics.update(self.stat_metrics.calculate(real_delay_up, syn_delay_up))

            # 时序依赖指标
            metrics.update(self.temporal_metrics.calculate(real_delay_up, syn_delay_up))

            # 频域特性指标
            metrics.update(self.spectral_metrics.calculate(real_delay_up, syn_delay_up))

            # 丢包突发性指标
            metrics.update(self.burst_metrics.calculate(real_loss_up, syn_loss_up))

            # 联合一致性指标
            metrics.update(self.joint_metrics.calculate(
                real_delay_up, syn_delay_up, real_loss_up, syn_loss_up
            ))

            all_metrics.append(metrics)

        # 5. 序列级评估
        logger.info("开始序列级评估")
        
        # 提取完整序列
        real_delay = real_df['raw_delay_up'].values
        syn_delay = syn_df['raw_delay_up'].values
        real_loss = real_df['raw_loss_up'].values
        syn_loss = syn_df['raw_loss_up'].values
        
        # 提取状态ID（如果有的话）
        # 对于真实文件，尝试使用TraceAnalyzer提取状态ID
        from netfaker.simcore.trace_analyzer import TraceAnalyzer
        real_states = None
        try:
            analyzer = TraceAnalyzer()
            # 提取真实文件的状态序列
            real_segments = analyzer.analyze(real_file)
            if real_segments:
                # 将segments转换为state_ids列表
                real_states = []
                for segment in real_segments:
                    state_id = int(segment['type'].replace('s', ''))
                    # 每个segment的时长对应多个窗口
                    window_count = segment['duration'] // 10  # 10秒/窗口
                    real_states.extend([state_id] * window_count)
        except Exception as e:
            logger.warning(f"提取真实文件状态序列失败: {e}")
        
        # 对于合成文件，直接使用文件中的state_id（如果有）
        syn_states = syn_df['state_id'].tolist() if 'state_id' in syn_df.columns else None
        
        # 计算序列级指标
        sequence_metrics = {}
        
        # 长程自相关指标
        sequence_metrics.update(self.long_term_acf_metrics.calculate(real_delay, syn_delay))
        
        # 状态转移指标
        sequence_metrics.update(self.state_transition_metrics.calculate(real_states, syn_states))
        
        # 滚动统计指标
        sequence_metrics.update(self.rolling_stats_metrics.calculate(real_delay, syn_delay))
        
        # burst时序指标
        sequence_metrics.update(self.burst_timing_metrics.calculate(real_loss, syn_loss))
        
        logger.info("序列级评估完成")

        # 6. 生成报告
        reporter = EvaluationReporter(self.config, output_dir)
        report = reporter.generate_report(all_metrics, real_file, syn_file, sequence_metrics)

        logger.info(f"评估完成，报告已生成：{output_dir}")
        return report

    def _validate_and_truncate(self, real_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple:
        """验证并截断数据长度，确保为100的整数倍。
        
        Args:
            real_df: 真实数据 DataFrame
            syn_df: 合成数据 DataFrame
            
        Returns:
            tuple: (截断后的真实数据, 截断后的合成数据)
        """
        # 确保长度为100的整数倍
        real_len = len(real_df) - len(real_df) % self.window_size
        syn_len = len(syn_df) - len(syn_df) % self.window_size

        # 取最小值
        min_len = min(real_len, syn_len)

        if min_len < self.window_size:
            logger.error(f"数据长度不足一个窗口：真实={real_len}，合成={syn_len}")
            raise ValueError(f"数据长度不足一个窗口：真实={real_len}，合成={syn_len}")

        if real_len != min_len:
            logger.warning(f"截断真实数据：从 {len(real_df)} 到 {min_len} 行")
        if syn_len != min_len:
            logger.warning(f"截断合成数据：从 {len(syn_df)} 到 {min_len} 行")

        return real_df.iloc[:min_len], syn_df.iloc[:min_len]

    def _split_into_windows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """将 DataFrame 按窗口大小切分为多个窗口。
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            List[pd.DataFrame]: 窗口列表
        """
        windows = []
        total_rows = len(df)

        for start in range(0, total_rows, self.window_size):
            end = start + self.window_size
            window_df = df.iloc[start:end]
            windows.append(window_df)

        return windows
