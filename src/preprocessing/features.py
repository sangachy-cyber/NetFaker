import numpy as np
import pandas as pd


def compute_window_features(window_df: pd.DataFrame) -> np.ndarray:
    """计算窗口的全局特征
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        13维全局特征数组
    """
    features = np.zeros(13)

    # 全局目标特征（13维中的前13个）
    features[0] = np.mean(window_df["delay_up"])   # mean_delay_up
    features[1] = np.std(window_df["delay_up"])    # std_delay_up
    features[2] = np.percentile(window_df["delay_up"], 5)  # p5_delay_up
    features[3] = np.percentile(window_df["delay_up"], 95)  # p95_delay_up
    features[4] = np.mean(window_df["delay_down"])   # mean_delay_down
    features[5] = np.std(window_df["delay_down"])    # std_delay_down
    features[6] = np.percentile(window_df["delay_down"], 5)  # p5_delay_down
    features[7] = np.percentile(window_df["delay_down"], 95)  # p95_delay_down
    
    # 新增：range特征
    features[8] = features[3] - features[2]  # range_up = p95_up - p5_up
    features[9] = features[7] - features[6]  # range_dn = p95_dn - p5_dn
    # 简化丢包分类，只保留有意义的特征
    # 由于loss_up和loss_dn只有0和1两个值，frac_cat1等于均值
    features[10] = np.mean(window_df["loss_up"])    # frac_cat1_up (等于均值)
    features[11] = np.mean(window_df["loss_dn"])    # frac_cat1_dn (等于均值) - 后续会被网络状态ID覆盖
    features[12] = 0.0  # reserved2

    return features


def compute_local_features(window_df: pd.DataFrame, last_n: int = 5) -> np.ndarray:
    """计算窗口的局部特征（最后n行）
    
    Args:
        window_df: 窗口DataFrame
        last_n: 使用最后几行计算特征
        
    Returns:
        10维局部特征数组
    """
    # 忽略局部特征，直接返回零向量
    return np.zeros(10)


def merge_features(global_features: np.ndarray, local_features: np.ndarray) -> np.ndarray:
    """合并全局特征和局部特征
    
    Args:
        global_features: 13维全局特征数组
        local_features: 10维局部特征数组
        
    Returns:
        23维条件向量
    """
    return np.concatenate([global_features, local_features])


def normalize_condition_vector(cond_vector: np.ndarray, cond_mean: np.ndarray, cond_std: np.ndarray) -> np.ndarray:
    """对条件向量进行Z-score标准化
    
    Args:
        cond_vector: 23维条件向量
        cond_mean: 22维条件向量均值
        cond_std: 22维条件向量标准差
        
    Returns:
        23维标准化条件向量
    """
    # 分离出需要标准化的维度（除了维度11网络状态ID）
    # 全局特征取前11维+保留位12，局部特征是10维，总共22维
    features_to_normalize = np.concatenate([cond_vector[:11], [cond_vector[12]], cond_vector[13:]])
    normalized_features = (features_to_normalize - cond_mean) / cond_std
    
    # 重新组合，保持网络状态ID不变
    normalized_cond_vector = np.concatenate([
        normalized_features[:11],
        [cond_vector[11]],  # 网络状态ID
        [normalized_features[11]],  # 标准化后的保留位
        normalized_features[12:],
    ])
    
    return normalized_cond_vector
