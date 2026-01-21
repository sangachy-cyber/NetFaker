"""特征提取模块，用于从网络状态数据中提取特征向量。

该模块提供了以下功能：
- 从完整网络状态窗口中提取16维特征向量
- 从单个方向（上行/下行）的网络状态数据中提取8维特征向量
- 支持延迟和丢包率的特征提取
- 使用log变换处理延迟数据，增强特征的区分度

特征向量结构：
- 上行延迟特征（4维）：均值、标准差、95分位数、最大值
- 上行丢包特征（4维）：是否有丢包、条件丢包均值、丢包比例、丢包均值
- 下行延迟特征（4维）：均值、标准差、95分位数、最大值
- 下行丢包特征（4维）：是否有丢包、条件丢包均值、丢包比例、丢包均值
"""

import numpy as np


def extract_features_from_window(window: dict) -> np.ndarray:
    """从单个网络状态窗口中提取16维特征向量。

    该函数处理完整的网络状态窗口，分别提取上行和下行方向的特征，
    然后组合成一个16维的特征向量。

    Args:
        window: 包含网络状态数据的字典，必须包含以下字段：
            - raw_delay_up: 上行延迟序列，长度为100
            - raw_loss_up: 上行丢包率序列，长度为100
            - raw_delay_down: 下行延迟序列，长度为100
            - raw_loss_down: 下行丢包率序列，长度为100

    Returns:
        np.ndarray: 16维特征向量，数据类型为float64，顺序为：
            [上行延迟特征4维, 上行丢包特征4维, 下行延迟特征4维, 下行丢包特征4维]

    示例：
    ```python
    # 准备网络状态窗口数据
    window = {
        "raw_delay_up": [10.1, 12.3, 9.8, ...],  # 长度为100的列表
        "raw_loss_up": [0.0, 0.0, 0.1, ...],      # 长度为100的列表
        "raw_delay_down": [15.2, 13.7, 14.5, ...], # 长度为100的列表
        "raw_loss_down": [0.0, 0.2, 0.0, ...]      # 长度为100的列表
    }

    # 提取特征向量
    features = extract_features_from_window(window)
    print("特征向量形状:", features.shape)  # 输出 (16,)
    print("特征向量:", features)
    ```
    """
    features = []

    # 处理上行数据
    up_features = _extract_direction_features(
        window["raw_delay_up"],
        window["raw_loss_up"]
    )
    features.extend(up_features)

    # 处理下行数据
    down_features = _extract_direction_features(
        window["raw_delay_down"],
        window["raw_loss_down"]
    )
    features.extend(down_features)

    return np.array(features, dtype=np.float64)


def _extract_direction_features(delay: list, loss: list) -> list:
    """从单个方向的网络状态数据中提取8维特征向量。

    该函数从单个方向（上行或下行）的网络状态数据中提取特征，
    包括延迟特征和丢包特征，共8维。

    处理步骤：
    1. 对延迟数据进行log变换，增强特征的区分度
    2. 提取延迟的统计特征：均值、标准差、95分位数、最大值
    3. 提取丢包特征：是否有丢包、条件丢包均值、丢包比例、丢包均值

    Args:
        delay: 延迟序列，长度为100
        loss: 丢包率序列，长度为100

    Returns:
        list: 8维特征向量，顺序为：
            [delay_log_mean, delay_log_std, delay_log_p95, delay_log_max,
             has_loss, cond_loss_mean, loss_ratio, loss_mean]
            - delay_log_mean: 对数变换后的延迟均值
            - delay_log_std: 对数变换后的延迟标准差
            - delay_log_p95: 对数变换后的延迟95分位数
            - delay_log_max: 对数变换后的延迟最大值
            - has_loss: 是否存在丢包（0或1）
            - cond_loss_mean: 存在丢包时的条件均值
            - loss_ratio: 丢包比例
            - loss_mean: 丢包率均值

    示例：
    ```python
    # 准备单个方向的网络状态数据
    delay = [10.1, 12.3, 9.8, ...]  # 长度为100的列表
    loss = [0.0, 0.0, 0.1, ...]      # 长度为100的列表

    # 提取方向特征
    dir_features = _extract_direction_features(delay, loss)
    print("方向特征长度:", len(dir_features))  # 输出 8
    print("方向特征:", dir_features)
    ```
    """
    features = []

    # 步骤1：对延迟进行log变换
    delay_array = np.array(delay, dtype=np.float64)
    delay_log = np.log(delay_array + 1)

    # 步骤2：提取延迟特征
    features.append(np.mean(delay_log))
    features.append(np.std(delay_log, ddof=1))
    features.append(np.percentile(delay_log, 95))
    features.append(np.max(delay_log))

    # 步骤3：提取丢包特征
    loss_array = np.array(loss, dtype=np.float64)
    has_loss = 1.0 if np.any(loss_array > 0) else 0.0

    cond_loss_mean = np.mean(loss_array[loss_array > 0]) if has_loss else 0.0

    loss_ratio = np.sum(loss_array > 0) / 100.0
    loss_mean = np.mean(loss_array)

    features.extend([has_loss, cond_loss_mean, loss_ratio, loss_mean])

    return features
