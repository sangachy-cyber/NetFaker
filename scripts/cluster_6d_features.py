#!/usr/bin/env python3

"""
6维特征聚类脚本
使用KMeans对6维特征进行聚类，将聚类结果作为新的state_id
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


def cluster_features(features, n_clusters=6):
    """对特征进行KMeans聚类
    
    Args:
        features: 特征数组 (N, 6)
        n_clusters: 聚类数量
    
    Returns:
        聚类标签 (N,)
    """
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    return cluster_labels


def main():
    # 读取数据集
    datasets_dir = Path("output/datasets")
    
    # 读取所有数据集文件
    dataset_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    all_data = []
    all_features = []
    all_file_ids = []
    
    # 提取所有数据和特征
    for file_name in dataset_files:
        file_path = datasets_dir / file_name
        print(f"处理文件: {file_path}")
        
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(data)
                window_data = data["window"]
                features = extract_trace_features(window_data)
                all_features.append(features)
                all_file_ids.append(file_name)
    
    # 转换为numpy数组
    features_array = np.array(all_features)
    print(f"总样本数: {len(all_data)}")
    print(f"特征维度: {features_array.shape[1]}")
    
    # 进行聚类
    cluster_labels = cluster_features(features_array, n_clusters=6)
    
    # 统计聚类结果
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\n聚类结果分布:")
    for label, count in zip(unique, counts):
        print(f"簇 {label}: {count} 个样本 ({count/len(cluster_labels)*100:.2f}%)")
    
    # 创建输出目录
    output_dir = Path("output/clustering")
    output_dir.mkdir(exist_ok=True)
    
    # 保存聚类结果
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    print(f"\n聚类标签已保存到: {output_dir / 'cluster_labels.npy'}")
    
    # 保存每个样本的聚类结果
    with open(output_dir / "cluster_results.jsonl", "w") as f:
        for data, label, file_id in zip(all_data, cluster_labels, all_file_ids):
            result = {
                "trace_id": data["trace_id"],
                "start_time": data["start_time"],
                "cluster_label": int(label),
                "file_id": file_id
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"聚类结果详情已保存到: {output_dir / 'cluster_results.jsonl'}")
    
    # 生成新的network_state_map
    network_state_map = {}
    for data, label in zip(all_data, cluster_labels):
        network_state_map[data["trace_id"]] = int(label)
    
    with open(output_dir / "network_state_map.json", "w") as f:
        json.dump(network_state_map, f, ensure_ascii=False, indent=2)
    print(f"新的network_state_map已保存到: {output_dir / 'network_state_map.json'}")
    
    # 可视化聚类结果
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # t-SNE降维
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    Z = tsne.fit_transform(features_scaled)
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7, s=50)
    plt.title("t-SNE of 6D Features with Cluster Labels")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, ticks=range(6), label="Cluster Label")
    plt.grid(True, alpha=0.3)
    
    # 保存可视化结果
    plt.savefig(output_dir / "clustering_tsne.png", dpi=300, bbox_inches="tight")
    print(f"聚类可视化结果已保存到: {output_dir / 'clustering_tsne.png'}")
    plt.show()
    
    print("\n聚类完成！")


if __name__ == "__main__":
    main()