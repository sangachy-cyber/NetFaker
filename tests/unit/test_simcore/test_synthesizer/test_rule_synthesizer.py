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

    def _create_mock_window(self, state_id):
        """创建模拟窗口数据。"""
        mock_window = MagicMock()

        # 根据状态生成不同范围的延迟数据
        def get_delay_values(state_id):
            if state_id == 0 or state_id == 2:  # 低延迟状态
                return [np.random.normal(50, 10) for _ in range(100)]
            else:  # 高延迟状态
                return [np.random.normal(500, 50) for _ in range(100)]

        # 根据状态生成不同范围的丢包率数据
        def get_loss_values(state_id):
            if state_id == 0 or state_id == 1:  # 低丢包率状态
                return [np.random.uniform(0, 10) for _ in range(100)]
            else:  # 高丢包率状态
                return [np.random.uniform(30, 50) for _ in range(100)]

        def getitem_side_effect(key):
            if key == "state_id":
                return state_id
            elif key in ["raw_delay_up", "raw_delay_down", "raw_bw_up", "raw_bw_down"]:
                return get_delay_values(state_id)
            elif key in ["raw_loss_up", "raw_loss_down"]:
                return get_loss_values(state_id)
            return []

        mock_window.__getitem__.side_effect = getitem_side_effect
        mock_window.__len__.return_value = 7
        return mock_window

    def _setup_window_sampler_mock(self, mock_window_sampler):
        """设置窗口采样器mock。"""
        def window_sampler_side_effect(state_id, n_windows, seed=None):
            return [self._create_mock_window(state_id) for _ in range(n_windows)]

        mock_sampler_instance = mock_window_sampler.return_value
        mock_sampler_instance.sample_windows.side_effect = window_sampler_side_effect

    def _setup_clusterer_mock(self, mock_gmm_load):
        """设置GMMClusterer mock。"""
        mock_clusterer = MagicMock()
        # 模拟predict_proba返回高置信度的状态概率
        def predict_proba_side_effect(features):
            n_samples = features.shape[0]
            probabilities = np.zeros((n_samples, 4))

            for i in range(n_samples):
                # 根据特征的均值判断应该属于哪个状态
                feature_mean = np.mean(features[i])

                if feature_mean < 100:  # 低延迟、低丢包
                    state_id = 0
                elif feature_mean < 200:  # 低延迟、高丢包
                    state_id = 2
                elif feature_mean < 600:  # 高延迟、低丢包
                    state_id = 1
                else:  # 高延迟、高丢包
                    state_id = 3

                probabilities[i, state_id] = 0.9
                probabilities[i, (state_id + 1) % 4] = 0.1
            return probabilities
        mock_clusterer.predict_proba.side_effect = predict_proba_side_effect
        mock_gmm_load.return_value = mock_clusterer

    def _setup_scaler_mock(self, mock_joblib_load):
        """设置scaler mock。"""
        mock_scaler = MagicMock()
        # 简单缩放特征，使不同状态的特征有明显区别
        def scale_side_effect(features):
            scaled_features = np.copy(features)
            for i in range(scaled_features.shape[0]):
                feature_mean = np.mean(scaled_features[i])
                if feature_mean < 100:
                    scaled_features[i] *= 0.5  # 状态0：进一步降低特征值
                elif feature_mean < 200:
                    scaled_features[i] *= 1.5  # 状态2：提高特征值
                elif feature_mean < 600:
                    scaled_features[i] *= 1.2  # 状态1：适度提高特征值
                else:
                    scaled_features[i] *= 1.8  # 状态3：大幅提高特征值
            return scaled_features
        mock_scaler.transform.side_effect = scale_side_effect
        mock_joblib_load.return_value = mock_scaler

    def _setup_mocks(self, mock_joblib_load, mock_gmm_load, mock_window_sampler):
        """设置测试所需的各种mock。"""
        # 设置窗口采样器mock
        self._setup_window_sampler_mock(mock_window_sampler)

        # 设置GMMClusterer mock
        self._setup_clusterer_mock(mock_gmm_load)

        # 设置scaler mock
        self._setup_scaler_mock(mock_joblib_load)

    def _verify_data_quality(self, data_points):
        """验证生成的数据质量。"""
        for i, point in enumerate(data_points):
            ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw = point

            # 验证延迟值范围 (0-2000ms)
            assert 0 <= ul_delay <= 2000, f"上行延迟异常值: {ul_delay} 在第 {i+1} 行"
            assert 0 <= dl_delay <= 2000, f"下行延迟异常值: {dl_delay} 在第 {i+1} 行"

            # 验证丢包率范围 (0-100%)
            assert 0 <= ul_loss <= 100, f"上行丢包率异常值: {ul_loss} 在第 {i+1} 行"
            assert 0 <= dl_loss <= 100, f"下行丢包率异常值: {dl_loss} 在第 {i+1} 行"

            # 验证带宽值非负
            assert ul_bw >= 0, f"上行带宽异常值: {ul_bw} 在第 {i+1} 行"
            assert dl_bw >= 0, f"下行带宽异常值: {dl_bw} 在第 {i+1} 行"

    def _verify_window_features(self, windows, window_states):
        """验证每个窗口的数据特征符合预期。"""
        for i, (window, expected_state) in enumerate(zip(windows, window_states, strict=True)):
            # 计算窗口的统计特征
            window_data = np.array(window)
            # 注意HoloWAN文件列顺序：[ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw]
            ul_delay_mean = np.mean(window_data[:, 0])
            ul_loss_mean = np.mean(window_data[:, 1])
            dl_delay_mean = np.mean(window_data[:, 3])
            dl_loss_mean = np.mean(window_data[:, 4])

            # 根据预期状态验证特征范围
            if expected_state == 0:
                # 状态0：低延迟、低丢包
                assert ul_delay_mean < 100, f"窗口 {i+1} 预期为状态0，实际上行延迟均值: {ul_delay_mean}"
                assert ul_loss_mean < 15, f"窗口 {i+1} 预期为状态0，实际上行丢包率均值: {ul_loss_mean}"
                assert dl_delay_mean < 100, f"窗口 {i+1} 预期为状态0，实际下行延迟均值: {dl_delay_mean}"
                assert dl_loss_mean < 15, f"窗口 {i+1} 预期为状态0，实际下行丢包率均值: {dl_loss_mean}"
            elif expected_state == 1:
                # 状态1：高延迟、低丢包
                assert ul_delay_mean > 300, f"窗口 {i+1} 预期为状态1，实际上行延迟均值: {ul_delay_mean}"
                assert ul_loss_mean < 15, f"窗口 {i+1} 预期为状态1，实际上行丢包率均值: {ul_loss_mean}"
                assert dl_delay_mean > 300, f"窗口 {i+1} 预期为状态1，实际下行延迟均值: {dl_delay_mean}"
                assert dl_loss_mean < 15, f"窗口 {i+1} 预期为状态1，实际下行丢包率均值: {dl_loss_mean}"
            elif expected_state == 2:
                # 状态2：低延迟、高丢包
                assert ul_delay_mean < 100, f"窗口 {i+1} 预期为状态2，实际上行延迟均值: {ul_delay_mean}"
                assert ul_loss_mean > 25, f"窗口 {i+1} 预期为状态2，实际上行丢包率均值: {ul_loss_mean}"
                assert dl_delay_mean < 100, f"窗口 {i+1} 预期为状态2，实际下行延迟均值: {dl_delay_mean}"
                assert dl_loss_mean > 25, f"窗口 {i+1} 预期为状态2，实际下行丢包率均值: {dl_loss_mean}"
            elif expected_state == 3:
                # 状态3：高延迟、高丢包
                assert ul_delay_mean > 300, f"窗口 {i+1} 预期为状态3，实际上行延迟均值: {ul_delay_mean}"
                assert ul_loss_mean > 25, f"窗口 {i+1} 预期为状态3，实际上行丢包率均值: {ul_loss_mean}"
                assert dl_delay_mean > 300, f"窗口 {i+1} 预期为状态3，实际下行延迟均值: {dl_delay_mean}"
                assert dl_loss_mean > 25, f"窗口 {i+1} 预期为状态3，实际下行丢包率均值: {dl_loss_mean}"

    def _verify_data_diversity(self, data_points):
        """验证生成的数据包含多种状态特征。"""
        # 计算所有窗口的平均延迟和丢包率
        all_data = np.array(data_points)
        all_ul_delay = all_data[:, 0]
        all_ul_loss = all_data[:, 1]

        # 检查是否有不同范围的数据
        assert np.max(all_ul_delay) > 300, f"生成的数据中没有高延迟数据，最大延迟: {np.max(all_ul_delay)}"
        assert np.min(all_ul_delay) < 100, f"生成的数据中没有低延迟数据，最小延迟: {np.min(all_ul_delay)}"
        assert np.max(all_ul_loss) > 25, f"生成的数据中没有高丢包率数据，最大丢包率: {np.max(all_ul_loss)}"
        assert np.min(all_ul_loss) < 15, f"生成的数据中没有低丢包率数据，最小丢包率: {np.min(all_ul_loss)}"

    @patch("netfaker.simcore.synthesizer.rule_synthesizer.WindowSampler")
    @patch("netfaker.simcore.clustering.state_predictor.GMMClusterer.load")
    @patch("joblib.load")
    def test_complex_template_with_state_prediction(self, mock_joblib_load, mock_gmm_load, mock_window_sampler, tmp_path):
        """测试复杂模板生成并验证状态预测结果。

        测试包含多个状态转换的复杂模板，验证生成的数据质量和状态预测结果。
        """
        # 设置mock
        self._setup_mocks(mock_joblib_load, mock_gmm_load, mock_window_sampler)

        # 创建合成器实例
        synthesizer = RuleSynthesizer("dummy_path.parquet")

        # 创建输出路径
        output_path = tmp_path / "test_complex_output.txt"

        # 使用复杂模板：多个状态转换（duration必须是10的倍数）
        template = [
            {"type": "s0", "duration": 10},   # 10秒状态0
            {"type": "s1", "duration": 20},   # 20秒状态1
            {"type": "s2", "duration": 10},   # 10秒状态2
            {"type": "s3", "duration": 30},  # 30秒状态3
            {"type": "s0", "duration": 10}    # 10秒状态0
        ]

        # 调用generate方法
        result_path = synthesizer.generate(template, str(output_path), seed=42)

        # 验证结果
        assert result_path == str(output_path)
        assert mock_window_sampler.return_value.sample_windows.called

        # 读取生成的文件并验证内容
        with open(result_path, "r") as f:
            lines = f.readlines()

        # 跳过文件头，找到数据开始位置
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "------------------------------------------------":
                data_start_idx = i + 1
                break
        else:
            pytest.fail("文件格式错误，未找到数据分隔线")

        # 提取数据点
        data_points = []
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
            values = list(map(float, line.split(",")))
            if len(values) != 6:
                continue
            data_points.append(values)

        # 验证数据点数量
        total_duration = sum(segment["duration"] for segment in template)
        expected_points = total_duration * 10  # 每秒10个数据点
        assert len(data_points) == expected_points, f"数据点数量不符，预期 {expected_points}，实际 {len(data_points)}"

        # 验证数据质量：没有异常值
        self._verify_data_quality(data_points)

        # 验证状态预测结果
        # 将数据点划分为窗口（每个窗口100个数据点，对应10秒）
        window_size = 100
        windows = []
        for i in range(0, len(data_points), window_size):
            window = data_points[i:i+window_size]
            if len(window) == window_size:
                windows.append(window)

        # 验证窗口数量
        expected_windows = total_duration // 10
        assert len(windows) == expected_windows, f"窗口数量不符，预期 {expected_windows}，实际 {len(windows)}"

        # 验证每个窗口的数据特征符合预期
        window_states = [0, 1, 1, 2, 3, 3, 3, 0]  # 预期的窗口状态序列
        self._verify_window_features(windows, window_states)

        # 验证生成的数据包含多种状态特征
        self._verify_data_diversity(data_points)
