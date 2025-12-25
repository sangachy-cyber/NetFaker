#!/usr/bin/env python3
"""
独立的聚类分析脚本，直接计算10D特征并进行聚类分析
"""

import os
import sys
import json
import numpy as np
from pathlib import Path


def _compute_10d_features(window_df):
    """计算窗口的10维特征（用于聚类）
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        10维特征数组 [p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn, p1_up, p1_dn, range_up, range_dn]
    """
    # 计算基础统计特征
    p99_up = np.percentile(window_df["del_up"], 99)
    std_up = np.std(window_df["del_up"])
    p1_up = np.percentile(window_df["del_up"], 1)
    range_up = p99_up - p1_up
    
    p99_dn = np.percentile(window_df["del_dn"], 99)
    std_dn = np.std(window_df["del_dn"])
    p1_dn = np.percentile(window_df["del_dn"], 1)
    range_dn = p99_dn - p1_dn
    
    # 丢包特征
    loss_up = np.mean(window_df["loss_up"])
    loss_dn = np.mean(window_df["loss_dn"])
    
    return np.array([p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn, p1_up, p1_dn, range_up, range_dn])


def load_jsonl_file(file_path):
    """加载JSONL文件
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        list: JSON对象列表
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_10d_features_for_windows(windows):
    """为多个窗口计算10D特征
    
    Args:
        windows: 窗口列表
    
    Returns:
        np.ndarray: 10D特征数组
    """
    import pandas as pd
    features = []
    for window in windows:
        window_df = pd.DataFrame(window)
        feature = _compute_10d_features(window_df)
        features.append(feature)
    return np.array(features)


def analyze_clustering_results():
    """分析聚类结果"""
    print("=== 10D特征聚类分析 ===")
    
    # 1. 加载已处理的数据集
    datasets_dir = Path("output/datasets")
    
    # 2. 为每个数据集计算10D特征
    datasets = {}
    all_features = []
    
    for dataset_name in ["train", "val", "test"]:
        file_path = datasets_dir / f"{dataset_name}.jsonl"
        if not file_path.exists():
            print(f"警告: {file_path} 不存在")
            continue
        
        print(f"\n加载 {dataset_name} 集...")
        data = load_jsonl_file(file_path)
        
        # 提取窗口数据
        windows = [item["window"] for item in data]
        print(f"  {dataset_name}集窗口数量: {len(windows)}")
        
        # 计算10D特征
        features = compute_10d_features_for_windows(windows)
        datasets[dataset_name] = {
            "data": data,
            "features": features
        }
        all_features.append(features)
    
    # 3. 合并所有特征
    all_features = np.vstack(all_features)
    print(f"\n总特征数量: {all_features.shape[0]}")
    print(f"特征维度: {all_features.shape[1]}")
    
    # 4. 执行KMeans聚类
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # 测试不同的K值
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_silhouette = -1
    best_k = 2
    
    print("\n=== 聚类质量指标 ===")
    print("K值\t轮廓系数\tDavies-Bouldin\tCalinski-Harabasz")
    print("-" * 60)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 计算聚类质量指标
        silhouette = silhouette_score(features_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_labels)
        
        print(f"{k}	{silhouette:.4f}	{davies_bouldin:.4f}		{calinski_harabasz:.2f}")
        
        # 更新最佳K值
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    print(f"\n最佳K值: {best_k} (轮廓系数: {best_silhouette:.4f})")
    
    # 5. 使用最佳K值进行最终聚类
    print(f"\n=== 使用最佳K值 {best_k} 进行聚类 ===")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # 6. 统计各数据集的state_id分布
    print("\n=== 各数据集state_id分布 ===")
    
    # 划分聚类标签到不同数据集
    start_idx = 0
    distribution = {}
    
    for dataset_name, dataset in datasets.items():
        end_idx = start_idx + len(dataset["features"])
        dataset_labels = cluster_labels[start_idx:end_idx]
        start_idx = end_idx
        
        # 统计分布
        unique, counts = np.unique(dataset_labels, return_counts=True)
        total = len(dataset_labels)
        
        state_dist = {}
        for state_id, count in zip(unique, counts):
            percentage = (count / total) * 100
            state_dist[int(state_id)] = {
                "count": int(count),
                "percentage": percentage
            }
        
        distribution[dataset_name] = {
            "total_samples": total,
            "state_distribution": state_dist
        }
        
        # 打印分布
        print(f"\n{dataset_name.upper()}集:")
        print(f"  总样本数: {total}")
        print(f"  状态ID分布:")
        for state_id in sorted(state_dist.keys()):
            stat = state_dist[state_id]
            print(f"    状态ID {state_id}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
    
    # 7. 保存结果
    output_dir = Path("output/10d_clustering_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存聚类结果
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    
    # 保存分布统计
    with open(output_dir / "distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_dir}")
    
    return distribution


def main():
    """主函数"""
    analyze_clustering_results()


if __name__ == "__main__":
    main()
