#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弱网络行为聚类模块

将聚类功能从预处理中分离，专注于网络行为的聚类分析
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

# 添加项目根目录到Python搜索路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _compute_features(window):
    """计算窗口的10维特征
    
    Args:
        window: 网络窗口数据
        
    Returns:
        10维特征数组: [p99_up, std_up, p99_dn, std_dn, loss_up, loss_dn, p1_up, p1_dn, range_up, range_dn]
    """
    # 提取窗口数据
    delays_up = []
    delays_dn = []
    loss_up = []
    loss_dn = []
    
    for point in window:
        delays_up.append(point.get('delay_up', 0.0))
        delays_dn.append(point.get('delay_down', 0.0))
        loss_up.append(point.get('loss_up', 0.0))
        loss_dn.append(point.get('loss_dn', 0.0))
    
    delays_up = np.array(delays_up)
    delays_dn = np.array(delays_dn)
    loss_up = np.array(loss_up)
    loss_dn = np.array(loss_dn)
    
    # 计算特征
    p99_up = np.percentile(delays_up, 99) if len(delays_up) > 0 else 0
    std_up = np.std(delays_up) if len(delays_up) > 1 else 0
    p99_dn = np.percentile(delays_dn, 99) if len(delays_dn) > 0 else 0
    std_dn = np.std(delays_dn) if len(delays_dn) > 1 else 0
    
    loss_up_mean = np.mean(loss_up) if len(loss_up) > 0 else 0
    loss_dn_mean = np.mean(loss_dn) if len(loss_dn) > 0 else 0
    
    p1_up = np.percentile(delays_up, 1) if len(delays_up) > 0 else 0
    p1_dn = np.percentile(delays_dn, 1) if len(delays_dn) > 0 else 0
    
    range_up = np.max(delays_up) - np.min(delays_up) if len(delays_up) > 0 else 0
    range_dn = np.max(delays_dn) - np.min(delays_dn) if len(delays_dn) > 0 else 0
    
    return np.array([p99_up, std_up, p99_dn, std_dn, loss_up_mean, loss_dn_mean, p1_up, p1_dn, range_up, range_dn])


def perform_clustering(features: np.ndarray, fixed_k: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    执行网络行为聚类分析
    
    Args:
        features: 形状为(n_samples, n_features)的特征数组
        fixed_k: 固定K值，None表示自动选择
        
    Returns:
        (cluster_labels, clustering_info) 二元组
        cluster_labels: 聚类标签数组
        clustering_info: 聚类信息字典
    """
    print(f"\n=== 开始聚类分析 ===")
    print(f"特征数量: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    
    # 标准化特征用于聚类和可视化（使用RobustScaler，抗异常值）
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 校验：确保特征是基于原始延迟数据，显示整体统计信息
    print(f"\n=== 特征校验 ===")
    print(f"特征统计（归一化前）：")
    print(f"  整体均值: {np.mean(features):.2f}")
    print(f"  整体中位数: {np.median(features):.2f}")
    print(f"  整体最小值: {np.min(features):.2f}")
    print(f"  整体最大值: {np.max(features):.2f}")
    print(f"  整体标准差: {np.std(features):.2f}")
    
    print(f"\n特征统计（RobustScaler归一化后）：")
    print(f"  整体均值: {np.mean(features_scaled):.2f}")
    print(f"  整体中位数: {np.median(features_scaled):.2f}")
    print(f"  整体最小值: {np.min(features_scaled):.2f}")
    print(f"  整体最大值: {np.max(features_scaled):.2f}")
    print(f"  整体标准差: {np.std(features_scaled):.2f}")
    
    # 检查特征值范围是否合理
    print(f"\n归一化后特征值范围检查：")
    print(f"  99% 分位数上限: {np.percentile(features_scaled, 99):.2f}")
    print(f"  1% 分位数下限: {np.percentile(features_scaled, 1):.2f}")
    print(f"  超过 [-10, 10] 范围的特征值比例: {(np.mean((features_scaled < -10) | (features_scaled > 10)) * 100):.1f}%")
    
    # 使用HDBSCAN进行聚类
    print("\n=== 开始HDBSCAN聚类分析 ===")
    
    # HDBSCAN参数设置
    min_cluster_size = 5  # 簇的最小样本数
    min_samples = 3       # 每个点成为核心点所需的最小邻域样本数
    
    # 执行HDBSCAN聚类
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # 使用最大稳定性选择
    )
    
    hdb_labels = hdb.fit_predict(features_scaled)
    
    # 统计有效簇数（-1表示噪声点）
    unique_labels = np.unique(hdb_labels)
    valid_clusters = [label for label in unique_labels if label != -1]
    num_clusters = len(valid_clusters)
    num_noise = np.sum(hdb_labels == -1)
    
    print(f"HDBSCAN初始聚类: 有效簇数={num_clusters}, 噪声点={num_noise}")
    
    # 如果没有有效簇或簇数太少，使用K-means作为备选
    if num_clusters < 2:
        print("HDBSCAN聚类效果不佳，使用K-means作为备选")
        
        # 确定K值
        if fixed_k is not None:
            n_clusters = fixed_k
        else:
            n_clusters = 4  # 默认K值
        
        print(f"使用K-means，K={n_clusters}")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        
        cluster_labels = kmeans_labels
        num_clusters = n_clusters
        print(f"K-means聚类完成，簇数={num_clusters}")
    else:
        # 将噪声点分配到最近的簇
        print(f"最终聚类数量: {num_clusters}")
        print(f"将 {num_noise} 个噪声点分配到最近的簇...")
        
        # 计算每个簇的中心
        cluster_centers = []
        for label in valid_clusters:
            cluster_points = features_scaled[hdb_labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = np.array(cluster_centers)
        
        # 为噪声点找到最近的簇中心
        noise_indices = np.where(hdb_labels == -1)[0]
        for idx in noise_indices:
            point = features_scaled[idx]
            distances = np.linalg.norm(cluster_centers - point, axis=1)
            closest_cluster = valid_clusters[np.argmin(distances)]
            hdb_labels[idx] = closest_cluster
        
        cluster_labels = hdb_labels
        
        # 重新映射簇标签为0-based
        unique_labels = np.unique(cluster_labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        cluster_labels = np.array([label_map[label] for label in cluster_labels])
        num_clusters = len(unique_labels)
        
        print(f"调整后聚类分布: {np.bincount(cluster_labels)}")
    
    # 生成聚类可视化
    print("\n=== 生成聚类可视化图表 ===")
    
    # 创建输出目录
    output_dir = "output/clustering_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制聚类指标图表
    plt.figure(figsize=(12, 6))
    
    # 绘制聚类结果可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('聚类结果可视化 (PCA降维)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'), dpi=150)
    plt.close()
    print("聚类可视化图表已保存到: output/clustering_visualization/cluster_visualization.png")
    
    # 绘制聚类指标
    plt.figure(figsize=(10, 6))
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    print(f"\n=== 聚类结果验证 ===")
    print(f"轮廓系数: {silhouette_avg:.4f}")
    
    # 生成整合聚类可视化
    plt.figure(figsize=(15, 10))
    
    # 绘制前4个特征的分布
    for i in range(min(4, features.shape[1])):
        plt.subplot(2, 2, i+1)
        for cluster_id in range(num_clusters):
            cluster_data = features[:, i][cluster_labels == cluster_id]
            plt.hist(cluster_data, bins=30, alpha=0.5, label=f'Cluster {cluster_id}')
        plt.title(f'Feature {i+1} Distribution')
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'integrated_clustering_visualization.png'), dpi=150)
    plt.close()
    print("整合聚类可视化已保存到: output/clustering_visualization/integrated_clustering_visualization.png")
    
    # 收集聚类信息
    clustering_info = {
        'num_clusters': num_clusters,
        'cluster_distribution': np.bincount(cluster_labels).tolist(),
        'silhouette_score': silhouette_avg,
        'features_scaled': features_scaled.tolist(),
        'cluster_labels': cluster_labels.tolist()
    }
    
    return cluster_labels, clustering_info


def generate_cluster_visualizations(features: np.ndarray, cluster_labels: np.ndarray):
    """
    生成聚类可视化图表
    
    Args:
        features: 特征数组
        cluster_labels: 聚类标签
    """
    output_dir = "output/clustering_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制聚类指标图表
    plt.figure(figsize=(12, 8))
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(features, cluster_labels)
    
    # 绘制轮廓系数
    plt.subplot(2, 2, 1)
    plt.bar(['Silhouette Score'], [silhouette_avg], color='skyblue')
    plt.ylim(0, 1)
    plt.title(f'Silhouette Score: {silhouette_avg:.4f}')
    plt.grid(True)
    
    # 绘制聚类分布
    plt.subplot(2, 2, 2)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    plt.bar([f'Cluster {i}' for i in unique], counts, color='lightgreen')
    plt.title('Cluster Distribution')
    plt.ylabel('Number of Samples')
    plt.grid(True)
    
    # 绘制PCA降维图
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_metrics.png'), dpi=150)
    plt.close()
    
    print("聚类指标图表已保存到: output/clustering_visualization/clustering_metrics.png")


def analyze_network_states(cluster_labels: np.ndarray, datasets: Dict[str, List[Dict]]) -> Dict:
    """
    分析网络状态ID分布
    
    Args:
        cluster_labels: 聚类标签
        datasets: 数据集字典
        
    Returns:
        network_state_stats: 网络状态统计信息
    """
    print("\n=== 网络状态ID分布统计 ===")
    print(f"使用10D特征聚类，K={len(np.unique(cluster_labels))}")
    
    # 分配cluster_labels到各个数据集
    idx = 0
    state_id_stats = {}
    
    for dataset_name, dataset in datasets.items():
        state_counts = {}
        total_samples = 0
        
        for meta in dataset:
            if idx < len(cluster_labels):
                state_id = cluster_labels[idx]
                meta['network_state_id'] = state_id
                
                # 统计状态ID分布
                if state_id in state_counts:
                    state_counts[state_id] += 1
                else:
                    state_counts[state_id] = 1
                
                total_samples += 1
                idx += 1
        
        state_id_stats[dataset_name] = {
            'total_samples': total_samples,
            'state_counts': state_counts
        }
    
    # 打印状态ID分布
    print("\n各数据集状态ID分布:")
    print()
    
    for dataset_name, stats in state_id_stats.items():
        print(f"{dataset_name.upper()}集:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  状态ID分布:")
        
        sorted_states = sorted(stats['state_counts'].keys())
        for state_id in sorted_states:
            count = stats['state_counts'][state_id]
            percentage = (count / stats['total_samples']) * 100 if stats['total_samples'] > 0 else 0
            print(f"    状态ID {state_id}: {count}个样本 ({percentage:.2f}%)")
        print()
    
    return state_id_stats


def main():
    """
    聚类模块主函数，用于独立测试
    """
    print("# 弱网络行为聚类分析")
    print("="*60)
    
    # 这里可以添加独立测试代码
    print("聚类模块已加载，可用于独立执行网络行为聚类分析")


if __name__ == "__main__":
    main()
