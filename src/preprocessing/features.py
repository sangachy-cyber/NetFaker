import numpy as np
import pandas as pd


def compute_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """计算窗口的全局特征
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        15维全局特征数组，第0维为网络状态ID，1-7为上行分位点，8-14为下行分位点
    """
    features = np.zeros(15)

    # 网络状态ID（默认0，后续会被行为发现模块覆盖）
    features[0] = 0.0  # network_state_id
    
    # 全局目标特征：只保留关键分位点
    # 使用7个关键分位点：1%,10%,25%,50%,75%,90%,99%
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    
    # 上行分位数特征
    up_quantiles = np.percentile(window_df["delay_up_qt"], quantiles)
    features[1:8] = up_quantiles
    
    # 下行分位数特征
    down_quantiles = np.percentile(window_df["delay_down_qt"], quantiles)
    features[8:15] = down_quantiles
    
    return features


def normalize_condition_vector(cond_vector: np.ndarray, delay_up_mean: float, delay_up_std: float, delay_down_mean: float, delay_down_std: float) -> np.ndarray:
    """对条件向量进行Z-score标准化

    Args:
        cond_vector: 15维条件向量（只包含全局特征）
        delay_up_mean: 上行延迟的z-score均值
        delay_up_std: 上行延迟的z-score标准差
        delay_down_mean: 下行延迟的z-score均值
        delay_down_std: 下行延迟的z-score标准差
        
    Returns:
        15维标准化条件向量，第0维为网络状态ID，1-7为上行分位点，8-14为下行分位点
    """
    # 最终输出的条件向量，15维（1个network_state_id + 14个分位点特征）
    normalized_cond = np.zeros(15)
    
    # 第0维：网络状态ID，保持不变
    normalized_cond[0] = cond_vector[0]
    
    # 上行分位点特征：索引1-7，共7个特征
    normalized_cond[1:8] = (cond_vector[1:8] - delay_up_mean) / delay_up_std
    
    # 下行分位点特征：索引8-14，共7个特征
    normalized_cond[8:15] = (cond_vector[8:15] - delay_down_mean) / delay_down_std
    
    return normalized_cond
