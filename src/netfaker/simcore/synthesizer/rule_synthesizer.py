"""序列合成器模块。

负责将抽样得到的窗口数据展开并转换为仿真流量文件。
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
from loguru import logger

from netfaker.core.config import config
from netfaker.simcore.postprocessing import BoundaryStitcher
from netfaker.simcore.synthesizer.rule_parser import RuleParser
from netfaker.simcore.synthesizer.window_sampler import WindowSampler
from netfaker.simcore.utils.holowan import HoloWANFile


class RuleSynthesizer:
    """基于规则的序列合成器类。

    用于根据用户提供的状态序列模板，从已标注的窗口数据中抽样并生成仿真流量文件。
    """

    def __init__(self, window_pool_path: str):
        """初始化序列合成器。

        Args:
            window_pool_path: 已标注窗口数据的Parquet文件路径
        """
        self.window_pool_path = window_pool_path
        self.parser = RuleParser()
        self.sampler = WindowSampler(window_pool_path)

    def generate(self, template: List[dict], output_path: str, seed: Optional[int] = None) -> str:
        """生成仿真流量文件。

        Args:
            template: 用户提供的状态序列模板
            output_path: 输出文件路径
            seed: 随机种子，用于控制抽样的可复现性

        Returns:
            str: 生成的文件路径

        Raises:
            ValueError: 当模板格式不符合规范时
            FileNotFoundError: 当窗口数据文件不存在时

        Examples:
            >>> template = [{"type": "s0", "duration": 10}, {"type": "s1", "duration": 20}]
            >>> synth = RuleSynthesizer("data/clusters/train_with_state.parquet")
            >>> output_file = synth.generate(template, "output/rule_seq_01.txt", seed=42)
            >>> output_file
            'output/rule_seq_01.txt'
        """
        logger.info("开始基于规则的状态序列生成")
        logger.debug(f"输入模板: {template}")
        logger.debug(f"输出路径: {output_path}")
        logger.debug(f"随机种子: {seed}")

        # 1. 解析模板
        parsed_template = self.parser.parse_template(template)
        logger.info(f"模板解析完成: {parsed_template}")

        # 2. 计算总时长
        total_duration = sum(state[1] * 10 for state in parsed_template)
        total_points = total_duration * 10
        logger.info(f"总时长: {total_duration} 秒，总数据点: {total_points}")

        # 3. 收集所有抽样窗口
        all_windows = []
        for state_id, n_windows in parsed_template:
            logger.info(f"处理state_id={state_id}，需要 {n_windows} 个窗口")

            # 抽样窗口
            sampled_windows = self.sampler.sample_windows(state_id, n_windows, seed)
            all_windows.extend(sampled_windows)

        logger.info(f"完成所有窗口抽样，共 {len(all_windows)} 个窗口")

        # 4. 校验所有窗口数据
        for _window_idx, window in enumerate(all_windows):
            for field in ["raw_delay_up", "raw_delay_down", "raw_loss_up", "raw_loss_down", "raw_bw_up", "raw_bw_down"]:
                if len(window[field]) != 100:
                    raise AssertionError(f"窗口数据长度不匹配: {field} 长度为 {len(window[field])}，预期为 100")

        # 5. 应用边界平滑（如果启用）
        if config.enable_boundary_smoothing:
            logger.info("启用边界平滑后处理")

            # 收集各通道窗口数据和状态ID
            ul_delay_windows = [np.array(window["raw_delay_up"]) for window in all_windows]
            dl_delay_windows = [np.array(window["raw_delay_down"]) for window in all_windows]
            ul_loss_windows = [np.array(window["raw_loss_up"]) for window in all_windows]
            dl_loss_windows = [np.array(window["raw_loss_down"]) for window in all_windows]
            state_ids = [window["state_id"] for window in all_windows]

            # 初始化边界拼接器
            stitcher = BoundaryStitcher(
                overlap_points=config.boundary_overlap_points,
                loss_continuation_points=config.loss_continuation_points,
                loss_burst_threshold=config.loss_burst_threshold_pct
            )

            # 应用平滑处理
            ul_delay_windows = stitcher.smooth_delay(ul_delay_windows, state_ids)
            dl_delay_windows = stitcher.smooth_delay(dl_delay_windows, state_ids)
            ul_loss_windows = stitcher.extend_loss_bursts(ul_loss_windows, state_ids)
            dl_loss_windows = stitcher.extend_loss_bursts(dl_loss_windows, state_ids)

            # 更新窗口数据
            for i in range(len(all_windows)):
                all_windows[i]["raw_delay_up"] = ul_delay_windows[i].tolist()
                all_windows[i]["raw_delay_down"] = dl_delay_windows[i].tolist()
                all_windows[i]["raw_loss_up"] = ul_loss_windows[i].tolist()
                all_windows[i]["raw_loss_down"] = dl_loss_windows[i].tolist()
        else:
            logger.info("跳过边界平滑后处理（已禁用）")

        # 6. 创建HoloWANFile实例
        test_name = f"rule_based_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        holowan_file = HoloWANFile(
            operator="Synthetic",
            test_name=test_name,
            interval_sec=0.1,
            switch="1,1,0,1,1,0"  # 默认配置：启用delay+loss，禁用bandwidth
        )
        logger.info(f"创建HoloWANFile实例，test_name: {test_name}")

        # 7. 将处理后的窗口数据添加到HoloWANFile
        # 收集状态序列，每个窗口对应一个state_id
        state_sequence = []
        for window_idx, window in enumerate(all_windows):
            logger.debug(f"处理第 {window_idx+1}/{len(all_windows)} 个窗口")
            # 获取当前窗口的状态ID
            window_state_id = window.get("state_id", -1)
            state_sequence.append(window_state_id)

            # 遍历窗口内的每个时间点
            for t in range(100):
                # 提取6通道数据
                ul_delay = window["raw_delay_up"][t]
                ul_loss = window["raw_loss_up"][t]  # 已经是百分比形式
                ul_bw = window["raw_bw_up"][t]
                dl_delay = window["raw_delay_down"][t]
                dl_loss = window["raw_loss_down"][t]  # 已经是百分比形式
                dl_bw = window["raw_bw_down"][t]

                # 添加数据点，不再传递state_id参数
                holowan_file._add_data(ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw)

        # 8. 写入文件，传递state_sequence参数
        logger.info(f"开始写入文件: {output_path}")
        output_file = holowan_file.write_to_file(output_path, state_sequence)
        logger.info(f"文件写入完成: {output_file}")
        logger.info(f"生成的文件包含 {len(holowan_file.data)} 个数据点")

        return output_file
