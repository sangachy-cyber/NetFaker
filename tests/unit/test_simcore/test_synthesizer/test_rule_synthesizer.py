"""序列合成器测试模块。

测试序列合成器的仿真流量生成功能。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from netfaker.simcore.synthesizer.rule_synthesizer import RuleSynthesizer


class TestRuleSynthesizer:
    """测试序列合成器类。
    """

    @pytest.fixture
    def mock_sampled_window(self):
        """创建模拟抽样窗口。
        """
        # 创建模拟窗口数据
        mock_window = MagicMock()

        # 设置 __getitem__ 方法，返回正确长度的列表
        def getitem_side_effect(key):
            if key == "state_id":
                return 0
            elif key in ["raw_delay_up", "raw_delay_down", "raw_bw_up", "raw_bw_down"]:
                return np.random.rand(100).tolist()
            elif key in ["raw_loss_up", "raw_loss_down"]:
                return [int(x < 0.1) for x in np.random.rand(100)]
            return []

        mock_window.__getitem__.side_effect = getitem_side_effect

        # 设置 __len__ 方法（如果需要）
        mock_window.__len__.return_value = 7

        return mock_window

    @patch("netfaker.simcore.synthesizer.rule_synthesizer.WindowSampler")
    def test_generate_valid(self, mock_window_sampler, mock_sampled_window, tmp_path):
        """测试有效生成仿真流量文件。
        """
        # 设置mock
        mock_sampler_instance = mock_window_sampler.return_value
        mock_sampler_instance.sample_windows.return_value = [mock_sampled_window]

        # 创建合成器实例
        synthesizer = RuleSynthesizer("dummy_path.parquet")

        # 创建输出路径
        output_path = tmp_path / "test_output.txt"

        # 使用简单模板
        template = [{"type": "s0", "duration": 10}]

        # 调用generate方法
        result_path = synthesizer.generate(template, str(output_path), seed=42)

        # 验证结果
        assert result_path == str(output_path)
        mock_sampler_instance.sample_windows.assert_called_once()

    @patch("netfaker.simcore.synthesizer.rule_synthesizer.WindowSampler")
    def test_generate_invalid_template(self, mock_window_sampler, tmp_path):
        """测试生成无效模板。
        """
        # 创建合成器实例
        synthesizer = RuleSynthesizer("dummy_path.parquet")

        # 创建输出路径
        output_path = tmp_path / "test_output.txt"

        # 使用无效模板
        template = [{"type": "invalid", "duration": 10}]

        # 调用generate方法，预期抛出异常
        with pytest.raises(ValueError, match="无效的 type 格式"):
            synthesizer.generate(template, str(output_path), seed=42)

    @patch("netfaker.simcore.synthesizer.rule_synthesizer.WindowSampler")
    def test_generate_missing_state(self, mock_window_sampler, tmp_path):
        """测试生成缺少状态的模板。
        """
        # 设置mock
        mock_sampler_instance = mock_window_sampler.return_value
        mock_sampler_instance.sample_windows.side_effect = ValueError("没有可用的窗口对应state_id=999")

        # 创建合成器实例
        synthesizer = RuleSynthesizer("dummy_path.parquet")

        # 创建输出路径
        output_path = tmp_path / "test_output.txt"

        # 使用包含不存在state_id的模板
        template = [{"type": "s999", "duration": 10}]

        # 调用generate方法，预期抛出异常
        with pytest.raises(ValueError, match="没有可用的窗口对应state_id=999"):
            synthesizer.generate(template, str(output_path), seed=42)
