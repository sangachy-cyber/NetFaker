#!/usr/bin/env python3

"""
6维特征t-SNE可视化脚本
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from pathlib import Path


def extract_trace_features(window_data):
    """从窗口数据中提取6维特征
    
    Args:
        window_data: 窗口数据列表，每个元素是包含del_up, del_dn, loss_up, loss_dn的字典
    
    Returns:
        6维特征数组 [p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn]
    """
    # 转换为DataFrame
    df = pd.DataFrame(window_data)
    
    # 计算6维特征
    p99_up = np.percentile(df["del_up"], 99)
    std_up = np.std(df["del_up"])
    p99_dn = np.percentile(df["del_dn"], 99)
    std_dn = np.std(df["del_dn"])
    loss_up = np.mean(df["loss_up"])
    loss_dn = np.mean(df["loss_dn"])
    
    return np.array([p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn])


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
            features = extract_trace_features(window_data)
            all_features.append(features)
    
    # 转换为numpy数组
    X = np.array(all_features)
    print(f"提取的特征数量: {X.shape[0]}")
    print(f"特征维度: {X.shape[1]}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    Z = tsne.fit_transform(X_scaled)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.7, cmap="viridis")
    plt.title("t-SNE of 6D Features (p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    output_dir = Path("output/visualization")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "6d_features_tsne.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"t-SNE可视化结果已保存到: {output_path}")
    
    # 显示图像
    plt.show()


if __name__ == "__main__":
    main()