#!/usr/bin/env python3

"""
10维特征t-SNE可视化脚本
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import json


def compute_n_regime_changes(window_df):
    """计算regime变化次数
    
    Args:
        window_df: 窗口数据，包含del_up列
        
    Returns:
        regime变化次数
    """
    del_up = window_df["del_up"].values
    mean = np.mean(del_up)
    std = np.std(del_up)
    threshold = mean + 2 * std
    
    # 标记是否处于regime
    in_regime = del_up > threshold
    
    # 计算regime变化次数
    changes = np.diff(in_regime).astype(int)
    n_changes = np.sum(np.abs(changes))
    return n_changes


def compute_max_burst_duration(window_df):
    """计算最大突发持续时间
    
    Args:
        window_df: 窗口数据，包含del_up列
        
    Returns:
        最大突发持续时间（步数）
    """
    del_up = window_df["del_up"].values
    mean = np.mean(del_up)
    std = np.std(del_up)
    threshold = mean + 2 * std
    
    # 标记是否处于突发
    in_burst = del_up > threshold
    
    max_duration = 0
    current_duration = 0
    
    for is_burst in in_burst:
        if is_burst:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0
    
    return max_duration


def compute_spike_ratio(window_df):
    """计算尖峰比率
    
    Args:
        window_df: 窗口数据，包含del_up列
        
    Returns:
        尖峰比率
    """
    del_up = window_df["del_up"].values
    mean = np.mean(del_up)
    std = np.std(del_up)
    threshold = mean + 2 * std
    
    # 计算尖峰比例
    spike_count = np.sum(del_up > threshold)
    spike_ratio = spike_count / len(del_up)
    return spike_ratio


def compute_autocorr_lag5(window_df):
    """计算lag=5的自相关系数
    
    Args:
        window_df: 窗口数据，包含del_up列
        
    Returns:
        lag=5的自相关系数
    """
    del_up = window_df["del_up"].values
    
    # 计算自相关
    lag = 5
    if len(del_up) < lag * 2:
        return 0.0
    
    x = del_up[:-lag]
    y = del_up[lag:]
    
    # 计算相关系数
    corr = np.corrcoef(x, y)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def extract_enhanced_features(window_data):
    """提取10维增强特征
    
    Args:
        window_data: 窗口数据列表，每个元素是包含del_up, del_dn, loss_up, loss_dn的字典
    
    Returns:
        10维特征数组
    """
    # 转换为DataFrame
    df = pd.DataFrame(window_data)
    
    # 1. 分布统计
    p99_up = np.percentile(df["del_up"], 99)
    std_up = np.std(df["del_up"])
    mean_up = np.mean(df["del_up"])
    
    # 2. 突发特性
    max_burst_duration = compute_max_burst_duration(df)
    spike_ratio = compute_spike_ratio(df)
    
    # 3. 动态结构
    n_regime_changes = compute_n_regime_changes(df)
    autocorr_lag5 = compute_autocorr_lag5(df)
    
    # 4. 丢包
    loss_up = np.mean(df["loss_up"])
    loss_dn = np.mean(df["loss_dn"])
    
    # 5. 不对称性
    p99_dn = np.percentile(df["del_dn"], 99)
    asymmetry = p99_up - p99_dn
    
    # 组合10维特征
    features = np.array([
        p99_up,
        std_up,
        mean_up,
        max_burst_duration,
        spike_ratio,
        n_regime_changes,
        autocorr_lag5,
        loss_up,
        loss_dn,
        asymmetry
    ])
    
    return features


def main():
    # 读取数据集
    datasets_dir = Path("output/datasets")
    
    # 读取训练集
    train_file = datasets_dir / "train.jsonl"
    
    all_features = []
    with open(train_file, "r") as f:
        for line in f:
            data = json.loads(line)
            window_data = data["window"]
            features = extract_enhanced_features(window_data)
            all_features.append(features)
    
    # 转换为numpy数组
    X_new = np.array(all_features)
    print(f"提取的10维特征数量: {X_new.shape[0]}")
    print(f"特征维度: {X_new.shape[1]}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    Z = tsne.fit_transform(X_scaled)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.7, cmap="viridis")
    plt.title("t-SNE of 10D Enhanced Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    output_dir = Path("output/visualization")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "10d_features_tsne.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"t-SNE可视化结果已保存到: {output_path}")
    
    # 显示图像
    plt.show()


if __name__ == "__main__":
    main()