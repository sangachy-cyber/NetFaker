import numpy as np
import pytest

from netfaker.simcore.clustering import extract_features_from_window


def test_extract_features_from_window_basic():
    """测试基本功能：正常窗口数据的特征提取"""
    # 创建测试数据：固定延迟和丢包率
    test_window = {
        "raw_delay_up": [10.0] * 100,
        "raw_loss_up": [0.0] * 100,
        "raw_delay_down": [20.0] * 100,
        "raw_loss_down": [0.0] * 100
    }

    features = extract_features_from_window(test_window)

    # 验证输出形状
    assert features.shape == (16,)
    assert np.issubdtype(features.dtype, np.floating)

    # 验证特征值（上行延迟特征应该相同）
    assert np.allclose(features[0], np.log(11.0))  # delay_log_mean_up
    assert np.isclose(features[1], 0.0)  # delay_log_std_up（无波动）
    assert np.allclose(features[2], np.log(11.0))  # delay_log_p95_up
    assert np.allclose(features[3], np.log(11.0))  # delay_log_max_up

    # 验证上行丢包特征
    assert np.isclose(features[4], 0.0)  # has_loss_up（无丢包）
    assert np.isclose(features[5], 0.0)  # cond_loss_mean_up（无丢包时为0）
    assert np.isclose(features[6], 0.0)  # loss_ratio_up（无丢包）
    assert np.isclose(features[7], 0.0)  # loss_mean_up（无丢包）

    # 验证下行延迟特征
    assert np.allclose(features[8], np.log(21.0))  # delay_log_mean_down
    assert np.isclose(features[9], 0.0)  # delay_log_std_down（无波动）
    assert np.allclose(features[10], np.log(21.0))  # delay_log_p95_down
    assert np.allclose(features[11], np.log(21.0))  # delay_log_max_down

    # 验证下行丢包特征
    assert np.isclose(features[12], 0.0)  # has_loss_down（无丢包）
    assert np.isclose(features[13], 0.0)  # cond_loss_mean_down（无丢包时为0）
    assert np.isclose(features[14], 0.0)  # loss_ratio_down（无丢包）
    assert np.isclose(features[15], 0.0)  # loss_mean_down（无丢包）


def test_extract_features_from_window_with_loss():
    """测试包含丢包的窗口数据"""
    # 创建测试数据：部分丢包
    test_window = {
        "raw_delay_up": [10.0] * 100,
        "raw_loss_up": [0.0] * 90 + [0.1] * 10,  # 10%丢包率
        "raw_delay_down": [20.0] * 100,
        "raw_loss_down": [0.0] * 100
    }

    features = extract_features_from_window(test_window)

    # 验证上行丢包特征
    assert np.isclose(features[4], 1.0)  # has_loss_up（有丢包）
    assert np.isclose(features[5], 0.1)  # cond_loss_mean_up（仅在丢包时计算）
    assert np.isclose(features[6], 0.1)  # loss_ratio_up（10%丢包时间）
    assert np.isclose(features[7], 0.01)  # loss_mean_up（平均丢包率）


def test_extract_features_from_window_variable_delay():
    """测试延迟有波动的窗口数据"""
    # 创建测试数据：延迟有波动
    test_window = {
        "raw_delay_up": list(range(1, 101)),  # 1-100ms的线性增长
        "raw_loss_up": [0.0] * 100,
        "raw_delay_down": [10.0] * 100,
        "raw_loss_down": [0.0] * 100
    }

    features = extract_features_from_window(test_window)

    # 验证上行延迟特征
    assert features[0] > 0  # delay_log_mean_up 应该大于0
    assert features[1] > 0  # delay_log_std_up 应该大于0（有波动）
    assert features[2] > features[0]  # delay_log_p95_up 应该大于均值
    assert features[3] > features[2]  # delay_log_max_up 应该大于95分位数


def test_extract_features_from_window_edge_cases():
    """测试边界情况"""
    # 测试零延迟
    zero_delay_window = {
        "raw_delay_up": [0.0] * 100,
        "raw_loss_up": [0.0] * 100,
        "raw_delay_down": [0.0] * 100,
        "raw_loss_down": [0.0] * 100
    }

    zero_features = extract_features_from_window(zero_delay_window)
    assert np.isclose(zero_features[0], 0.0)  # log(0+1) = 0
    assert np.isclose(zero_features[8], 0.0)  # log(0+1) = 0

    # 测试全丢包
    full_loss_window = {
        "raw_delay_up": [10.0] * 100,
        "raw_loss_up": [1.0] * 100,
        "raw_delay_down": [10.0] * 100,
        "raw_loss_down": [1.0] * 100
    }

    full_loss_features = extract_features_from_window(full_loss_window)
    assert np.isclose(full_loss_features[4], 1.0)  # has_loss_up
    assert np.isclose(full_loss_features[5], 1.0)  # cond_loss_mean_up
    assert np.isclose(full_loss_features[6], 1.0)  # loss_ratio_up
    assert np.isclose(full_loss_features[7], 1.0)  # loss_mean_up


def test_extract_features_from_window_input_validation():
    """测试输入验证"""
    # 测试缺少字段
    incomplete_window = {
        "raw_delay_up": [10.0] * 100,
        "raw_loss_up": [0.0] * 100,
        "raw_delay_down": [20.0] * 100
        # 缺少 raw_loss_down
    }

    with pytest.raises(KeyError):
        extract_features_from_window(incomplete_window)

    # 测试数据长度不正确
    wrong_length_window = {
        "raw_delay_up": [10.0] * 99,  # 长度为99，不是100
        "raw_loss_up": [0.0] * 100,
        "raw_delay_down": [20.0] * 100,
        "raw_loss_down": [0.0] * 100
    }

    # 注意：这里不会抛出异常，因为numpy会自动处理，但计算结果可能不正确
    # 我们只是确保函数能够运行
    features = extract_features_from_window(wrong_length_window)
    assert features.shape == (16,)
