#!/usr/bin/env python3

"""
10维特征聚类脚本
使用KMeans对10维特征进行聚类，可视化不同K值的情况
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
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


def cluster_with_different_k(features, k_range=range(2, 11)):
    """使用不同的K值进行聚类
    
    Args:
        features: 10维特征数组
        k_range: K值范围
        
    Returns:
        聚类结果字典，键为K值，值为包含聚类标签和评估指标的字典
    """
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    for k in k_range:
        print(f"处理K={k}...")
        
        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # 计算聚类质量指标
        silhouette = silhouette_score(features_scaled, labels)
        davies_bouldin = davies_bouldin_score(features_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
        
        # 保存结果
        results[k] = {
            'labels': labels,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'model': kmeans,
            'scaler': scaler
        }
    
    return results


def plot_metrics(results, output_dir):
    """绘制聚类质量指标随K值变化的曲线
    
    Args:
        results: 聚类结果字典
        output_dir: 输出目录
    """
    k_values = list(results.keys())
    
    # 提取指标
    silhouette_scores = [results[k]['silhouette'] for k in k_values]
    davies_bouldin_scores = [results[k]['davies_bouldin'] for k in k_values]
    calinski_harabasz_scores = [results[k]['calinski_harabasz'] for k in k_values]
    
    # 绘制指标曲线
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # 轮廓系数
    axs[0].plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8)
    axs[0].set_title('Silhouette Score vs K', fontsize=14)
    axs[0].set_xlabel('K', fontsize=12)
    axs[0].set_ylabel('Silhouette Score', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    
    # Davies-Bouldin指数
    axs[1].plot(k_values, davies_bouldin_scores, marker='o', linewidth=2, markersize=8)
    axs[1].set_title('Davies-Bouldin Score vs K', fontsize=14)
    axs[1].set_xlabel('K', fontsize=12)
    axs[1].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz指数
    axs[2].plot(k_values, calinski_harabasz_scores, marker='o', linewidth=2, markersize=8)
    axs[2].set_title('Calinski-Harabasz Score vs K', fontsize=14)
    axs[2].set_xlabel('K', fontsize=12)
    axs[2].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axs[2].grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / 'metrics_vs_k.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"聚类质量指标曲线已保存到: {output_path}")


def visualize_clusters(features, results, output_dir):
    """可视化不同K值的聚类结果
    
    Args:
        features: 10维特征数组
        results: 聚类结果字典
        output_dir: 输出目录
    """
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    Z = tsne.fit_transform(features_scaled)
    
    # 为每个K值生成可视化
    for k, result in results.items():
        print(f"可视化K={k}的结果...")
        labels = result['labels']
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.title(f't-SNE of 10D Features with K={k}', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter, label='Cluster Label', ticks=range(k))
        plt.grid(True, alpha=0.3)
        
        # 添加聚类质量指标
        metrics_text = f"Silhouette Score: {result['silhouette']:.3f}\n"
        metrics_text += f"Davies-Bouldin: {result['davies_bouldin']:.3f}\n"
        metrics_text += f"Calinski-Harabasz: {result['calinski_harabasz']:.3f}"
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 保存图像
        output_path = output_dir / f'cluster_k{k}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"K={k}的可视化结果已保存到: {output_path}")


def main():
    # 读取数据集
    datasets_dir = Path("output/datasets")
    
    # 读取训练集
    train_file = datasets_dir / "train.jsonl"
    
    print("提取10维特征...")
    all_features = []
    with open(train_file, "r") as f:
        for line in f:
            data = json.loads(line)
            window_data = data["window"]
            features = extract_enhanced_features(window_data)
            all_features.append(features)
    
    # 转换为numpy数组
    features_array = np.array(all_features)
    print(f"总样本数: {features_array.shape[0]}")
    print(f"特征维度: {features_array.shape[1]}")
    
    # 执行聚类
    k_range = range(2, 11)
    print(f"执行K={list(k_range)}的聚类...")
    results = cluster_with_different_k(features_array, k_range)
    
    # 创建输出目录
    output_dir = Path("output/clustering_10d")
    output_dir.mkdir(exist_ok=True)
    
    # 绘制聚类质量指标曲线
    print("绘制聚类质量指标曲线...")
    plot_metrics(results, output_dir)
    
    # 可视化聚类结果
    print("可视化聚类结果...")
    visualize_clusters(features_array, results, output_dir)
    
    # 保存聚类结果
    print("保存聚类结果...")
    for k, result in results.items():
        # 保存聚类标签
        np.save(output_dir / f"cluster_labels_k{k}.npy", result['labels'])
    
    print("\n聚类完成！")
    
    # 打印最佳K值建议
    print("\n最佳K值建议：")
    
    # 轮廓系数最大的K值
    best_silhouette_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"基于轮廓系数: K={best_silhouette_k} (分数: {results[best_silhouette_k]['silhouette']:.3f})")
    
    # Davies-Bouldin指数最小的K值
    best_davies_bouldin_k = min(results.keys(), key=lambda k: results[k]['davies_bouldin'])
    print(f"基于Davies-Bouldin指数: K={best_davies_bouldin_k} (分数: {results[best_davies_bouldin_k]['davies_bouldin']:.3f})")
    
    # Calinski-Harabasz指数最大的K值
    best_calinski_harabasz_k = max(results.keys(), key=lambda k: results[k]['calinski_harabasz'])
    print(f"基于Calinski-Harabasz指数: K={best_calinski_harabasz_k} (分数: {results[best_calinski_harabasz_k]['calinski_harabasz']:.3f})")


if __name__ == "__main__":
    main()