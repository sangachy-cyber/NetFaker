import numpy as np


def extract_features_from_window(window: dict) -> np.ndarray:
    """
    从单个网络状态窗口中提取特征向量。

    Args:
        window: 包含网络状态数据的字典，必须包含以下字段：
            - raw_delay_up: 上行延迟序列，长度为100
            - raw_loss_up: 上行丢包率序列，长度为100
            - raw_delay_down: 下行延迟序列，长度为100
            - raw_loss_down: 下行丢包率序列，长度为100

    Returns:
        16维特征向量，顺序为：
        [上行延迟特征4维, 上行丢包特征4维, 下行延迟特征4维, 下行丢包特征4维]
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
    """
    从单个方向的网络状态数据中提取8维特征向量。

    Args:
        delay: 延迟序列，长度为100
        loss: 丢包率序列，长度为100

    Returns:
        8维特征向量，顺序为：
        [delay_log_mean, delay_log_std, delay_log_p95, delay_log_max,
         has_loss, cond_loss_mean, loss_ratio, loss_mean]
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

    if has_loss:
        cond_loss_mean = np.mean(loss_array[loss_array > 0])
    else:
        cond_loss_mean = 0.0

    loss_ratio = np.sum(loss_array > 0) / 100.0
    loss_mean = np.mean(loss_array)

    features.extend([has_loss, cond_loss_mean, loss_ratio, loss_mean])

    return features
