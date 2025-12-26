#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试使用TimeSeriesKMeans + Soft-DTW进行网络行为聚类
直接处理原始时间序列数据，无需手动提取特征

优化版本：添加进度条、降采样、并行计算和参数优化
"""

import os
# 优化multiprocessing行为，避免fork导致的问题
os.environ["LOKY_NO_FORK"] = "1"  # 强制使用spawn而非fork
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"  # 使用更稳定的临时目录
os.environ["OMP_NUM_THREADS"] = "4"  # 限制OpenMP线程数
os.environ["MKL_NUM_THREADS"] = "4"  # 限制MKL线程数

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time
import argparse

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="TimeSeriesKMeans + Soft-DTW 网络行为聚类测试")
    parser.add_argument("--sample-limit", type=int, default=2000, 
                      help="限制样本数量，降低内存占用（默认：2000）")
    parser.add_argument("--downsample-length", type=int, default=30, 
                      help="降采样目标长度（默认：30）")
    parser.add_argument("--downsample-method", type=str, default="paa", 
                      choices=["uniform", "paa"], 
                      help="降采样方法（默认：paa）")
    parser.add_argument("--max-k", type=int, default=3, 
                      help="最大K值（默认：3）")
    parser.add_argument("--n-init", type=int, default=2, 
                      help="每个K值的初始化次数（默认：2）")
    parser.add_argument("--n-jobs", type=int, default=4, 
                      help="并行作业数，限制CPU使用（默认：4）")
    parser.add_argument("--metric", type=str, default="dtw", 
                      choices=["dtw", "softdtw"], 
                      help="距离度量（默认：dtw，softdtw计算量更大）")
    parser.add_argument("--fixed-k", type=int, default=None, 
                      help="固定K值，跳过K值选择过程")
    return parser.parse_args()

def load_preprocessed_data(downsample_length=30, downsample_method='paa', sample_limit=2000):
    """
    加载真实的预处理数据
    从output/datasets目录中读取JSONL格式的窗口数据
    
    Args:
        downsample_length: 降采样目标长度，None表示不降采样
        downsample_method: 降采样方法，可选 'uniform' 或 'paa'（默认）
        sample_limit: 限制样本数量，降低内存占用
        
    Returns:
        转换后的tslearn格式数据
    """
    datasets_dir = Path("output/datasets")
    
    # 检查数据集目录是否存在
    if not datasets_dir.exists():
        print("No datasets directory found, generating sample data for testing...")
        return generate_sample_data()
    
    print(f"Loading real data from {datasets_dir}")
    
    # 读取所有JSONL文件
    jsonl_files = list(datasets_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found, generating sample data for testing...")
        return generate_sample_data()
    
    # 读取所有数据点
    all_windows = []
    for jsonl_file in jsonl_files:
        print(f"Reading {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'window' in data and data['window']:
                        all_windows.append(data['window'])
                        # 限制样本数量
                        if len(all_windows) >= sample_limit:
                            print(f"Reached sample limit of {sample_limit}, stopping reading")
                            break
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {jsonl_file.name}: {e}")
                    continue
            if len(all_windows) >= sample_limit:
                break
    
    print(f"Loaded {len(all_windows)} windows from real data")
    
    # 转换为适合TimeSeriesKMeans的格式
    if all_windows:
        return convert_to_tslearn_format(all_windows, downsample_length, downsample_method)
    else:
        print("No valid window data found, generating sample data...")
        return generate_sample_data()

def downsample_time_series(ts, target_length, method='paa'):
    """
    对单条时间序列进行降采样
    
    Args:
        ts: 原始时间序列，形状为(n_timestamps, n_features)
        target_length: 目标长度
        method: 降采样方法，可选 'uniform' 或 'paa'（默认）
        
    Returns:
        downsampled_ts: 降采样后的时间序列
    """
    original_length = len(ts)
    if original_length <= target_length:
        return ts
    
    if method == 'uniform':
        # 简单均匀采样：每隔step取一个点
        step = original_length // target_length
        indices = np.arange(0, original_length, step)[:target_length]
        return ts[indices]
    else:  # 'paa' - Piecewise Aggregate Approximation
        # PAA：更平滑的降采样，对每个区间取平均值
        bins = np.array_split(ts, target_length)
        return np.stack([np.mean(b, axis=0) for b in bins])

def convert_to_tslearn_format(windows, downsample_length=None, downsample_method='paa'):
    """
    将窗口数据转换为tslearn要求的格式
    
    Args:
        windows: 窗口数据列表，每个窗口包含多个时间点
        downsample_length: 降采样目标长度，None表示不降采样
        downsample_method: 降采样方法，可选 'uniform' 或 'paa'（默认）
        
    Returns:
        X: 形状为(n_samples, n_timestamps, 4)的时间序列数据
           4个特征: delay_up, loss_up, delay_down, loss_dn
    """
    print("Converting data to tslearn format...")
    
    # 确定统一的时间序列长度（使用最短窗口，避免截断过多数据）
    min_length = min(len(window) for window in windows)
    print(f"Original minimum window length: {min_length}")
    
    # 应用降采样
    if downsample_length is not None and downsample_length < min_length:
        target_length = downsample_length
        print(f"Applying downsampling to target length: {target_length} using {downsample_method} method")
    else:
        target_length = min_length
    
    # 提取4个特征：delay_up, loss_up, delay_down, loss_dn
    n_features = 4
    n_samples = len(windows)
    
    # 初始化结果数组
    X = np.zeros((n_samples, target_length, n_features))
    
    # 使用tqdm显示进度
    for i in tqdm(range(n_samples), desc="Processing windows"):
        window = windows[i]
        # 截断到最短窗口长度
        window_truncated = window[:min_length]
        
        # 转换为numpy数组以便降采样
        window_array = np.zeros((min_length, n_features))
        for j, point in enumerate(window_truncated):
            window_array[j, 0] = point.get('delay_up', 0.0)
            window_array[j, 1] = point.get('loss_up', 0.0)
            window_array[j, 2] = point.get('delay_down', 0.0)
            window_array[j, 3] = point.get('loss_dn', 0.0)
        
        # 应用降采样
        if downsample_length is not None and downsample_length < min_length:
            window_array = downsample_time_series(window_array, target_length, method=downsample_method)
        
        # 保存到结果数组
        X[i] = window_array
    
    print(f"Converted data shape: {X.shape}")
    return X

def generate_sample_data():
    """
    生成示例数据用于测试
    模拟预处理后的窗口数据
    """
    np.random.seed(42)
    
    # 生成3种不同模式的网络行为
    # 使用能被3整除的样本数，避免索引越界
    n_samples = 99
    T = 100  # 时间序列长度
    samples_per_pattern = n_samples // 3
    
    # 模式1：稳定网络
    pattern1 = {
        "del_up": np.random.normal(20, 5, size=(samples_per_pattern, T)),
        "del_dn": np.random.normal(30, 8, size=(samples_per_pattern, T)),
        "loss_up": np.random.binomial(1, 0.01, size=(samples_per_pattern, T)),
        "loss_dn": np.random.binomial(1, 0.02, size=(samples_per_pattern, T))
    }
    
    # 模式2：高延迟抖动
    pattern2 = {
        "del_up": np.concatenate([
            np.random.normal(20, 5, size=(samples_per_pattern, T//2)),
            np.random.normal(100, 20, size=(samples_per_pattern, T//2))
        ], axis=1),
        "del_dn": np.concatenate([
            np.random.normal(30, 8, size=(samples_per_pattern, T//2)),
            np.random.normal(150, 30, size=(samples_per_pattern, T//2))
        ], axis=1),
        "loss_up": np.concatenate([
            np.random.binomial(1, 0.01, size=(samples_per_pattern, T//2)),
            np.random.binomial(1, 0.1, size=(samples_per_pattern, T//2))
        ], axis=1),
        "loss_dn": np.concatenate([
            np.random.binomial(1, 0.02, size=(samples_per_pattern, T//2)),
            np.random.binomial(1, 0.15, size=(samples_per_pattern, T//2))
        ], axis=1)
    }
    
    # 模式3：突发丢包
    pattern3 = {
        "del_up": np.random.normal(40, 10, size=(samples_per_pattern, T)),
        "del_dn": np.random.normal(50, 15, size=(samples_per_pattern, T)),
        "loss_up": np.zeros((samples_per_pattern, T)),
        "loss_dn": np.zeros((samples_per_pattern, T))
    }
    # 添加突发丢包
    for i in range(samples_per_pattern):
        burst_pos = np.random.randint(0, T-20, size=2)
        for pos in burst_pos:
            pattern3["loss_up"][i, pos:pos+10] = 1
            pattern3["loss_dn"][i, pos:pos+15] = 1
    
    # 合并所有模式
    X = np.zeros((n_samples, T, 4))
    for i in range(n_samples):
        if i < samples_per_pattern:
            X[i, :, 0] = pattern1["del_up"][i]
            X[i, :, 1] = pattern1["del_dn"][i]
            X[i, :, 2] = pattern1["loss_up"][i]
            X[i, :, 3] = pattern1["loss_dn"][i]
        elif i < 2 * samples_per_pattern:
            idx = i - samples_per_pattern
            X[i, :, 0] = pattern2["del_up"][idx]
            X[i, :, 1] = pattern2["del_dn"][idx]
            X[i, :, 2] = pattern2["loss_up"][idx]
            X[i, :, 3] = pattern2["loss_dn"][idx]
        else:
            idx = i - 2 * samples_per_pattern
            X[i, :, 0] = pattern3["del_up"][idx]
            X[i, :, 1] = pattern3["del_dn"][idx]
            X[i, :, 2] = pattern3["loss_up"][idx]
            X[i, :, 3] = pattern3["loss_dn"][idx]
    
    print(f"Generated sample data with shape: {X.shape}")
    return X

def prepare_time_series_data(preprocessed_data):
    """
    准备时间序列数据，格式化为tslearn要求的形状
    
    Args:
        preprocessed_data: 预处理后的数据
        
    Returns:
        X: 形状为(n_samples, n_timestamps, 4)的时间序列数据
    """
    # 这里直接使用模拟数据
    # 实际实现时需要从预处理数据中提取
    X = preprocessed_data
    
    print(f"Prepared time series data with shape: {X.shape}")
    print(f"  Number of samples: {X.shape[0]}")
    print(f"  Time series length: {X.shape[1]}")
    print(f"  Number of dimensions: {X.shape[2]}")
    
    return X

def select_optimal_k(X, max_k=3):
    """
    使用肘部法则选择最优K值
    
    Args:
        X: 时间序列数据
        max_k: 最大尝试的K值
        
    Returns:
        optimal_k: 最优的聚类数量
    """
    print(f"\n=== 开始选择最优K值 ===")
    print(f"尝试K值范围: 2-{max_k}")
    print(f"TimeSeriesKMeans配置: metric='dtw', n_jobs=4, n_init=2, multiprocessing backend")
    
    # 计算总任务数和预计时间
    total_k = max_k - 1  # K从2到max_k，共max_k-1个K值
    n_init = 2
    print(f"\n📊 进度预估")
    print(f"   总K值数量: {total_k}")
    print(f"   每个K值初始化次数: {n_init}")
    print(f"   预计总耗时: 约{(total_k * 10):.0f}-{(total_k * 20):.0f}秒")
    print("   （已启用多核并行，限制4个CPU核心）")
    print("   正在执行，详细进度如下:\n")
    
    inertias = []
    silhouettes = []
    
    # 记录开始时间和当前进度
    start_total = time.time()
    
    for i, k in enumerate(range(2, max_k+1)):
        current_progress = (i / total_k) * 100
        print(f"{'='*50}")
        print(f"🔄 当前进度: {current_progress:.1f}% (K={k}/{max_k})")
        print(f"{'='*50}")
        
        start_k_time = time.time()
        
        # 直接使用优化参数配置，设置verbose=1显示迭代信息
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",  # 使用dtw代替softdtw，更快
            n_jobs=4,     # 限制并行计算使用4个CPU核心
            n_init=2,       # 减少初始化次数
            max_iter=50,    # 限制最大迭代次数
            max_iter_barycenter=3,  # 大幅降低DBA迭代次数，默认是100
            tol=1e-3,       # 宽松的收敛阈值
            random_state=42,
            verbose=1       # 显示迭代进度
        )
        
        model.fit(X)
        inertias.append(model.inertia_)
        
        # 计算轮廓系数
        silhouette = silhouette_score(X.reshape(X.shape[0], -1), model.labels_)
        silhouettes.append(silhouette)
        
        elapsed_k = time.time() - start_k_time
        elapsed_total = time.time() - start_total
        
        # 计算预计剩余时间
        if i > 0:
            avg_time_per_k = elapsed_total / (i + 1)
            remaining_k = total_k - (i + 1)
            estimated_remaining = avg_time_per_k * remaining_k
            print(f"\n⏱️  K={k} 耗时: {elapsed_k:.1f}秒")
            print(f"📈 累计耗时: {elapsed_total:.1f}秒")
            print(f"⏳ 预计剩余: {estimated_remaining:.1f}秒")
        else:
            print(f"\n⏱️  K={k} 耗时: {elapsed_k:.1f}秒")
            print(f"� 累计耗时: {elapsed_total:.1f}秒")
            print(f"⏳ 预计剩余: 计算中...")
        
        print(f"📊 聚类结果: 迭代次数={model.n_iter_}, Inertia={model.inertia_:.2f}, 轮廓系数={silhouette:.3f}")
        print()
    
    total_time = time.time() - start_total
    print(f"\n🎉 最优K值选择完成！")
    print(f"⏱️  总耗时: {total_time:.1f}秒")
    print(f"{'='*50}")
    
    # 绘制肘部图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k+1), inertias, marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('惯性（越小越好）')
    plt.title('肘部法则（Elbow Method）')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k+1), silhouettes, marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('轮廓系数（越大越好）')
    plt.title('轮廓系数图')
    plt.grid(True)
    
    # 保存肘部图
    output_dir = Path("output/soft_dtw_clustering")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "elbow_plot.png", dpi=300, bbox_inches="tight")
    print(f"肘部图已保存到: {output_dir / 'elbow_plot.png'}")
    plt.close()
    
    # 选择轮廓系数最高的K值
    optimal_k = range(2, max_k+1)[np.argmax(silhouettes)]
    print(f"=== 最优K值选择结果 ===")
    print(f"  轮廓系数最高的K值: {optimal_k}")
    print(f"  对应的轮廓系数: {max(silhouettes):.3f}")
    
    return optimal_k

def perform_clustering(X, optimal_k, args):
    """
    执行TimeSeriesKMeans + Soft-DTW聚类
    
    Args:
        X: 时间序列数据
        optimal_k: 最优聚类数量
        args: 命令行参数
        
    Returns:
        model: 训练好的聚类模型
        cluster_labels: 聚类标签
    """
    print(f"\n=== 开始执行TimeSeriesKMeans + DTW聚类 ===")
    print(f"  K值: {optimal_k}")
    print(f"  距离度量: dtw")
    print(f"  并行计算: 启用 (multiprocessing backend, n_jobs={args.n_jobs})")
    print(f"  初始化次数: {args.n_init}")
    print(f"  最大迭代次数: 50")
    print(f"  收敛阈值: 1e-3")
    
    # 防止OpenMP冲突，强制使用单个线程
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    
    start_time = time.time()
    
    # 使用优化的参数配置
    model = TimeSeriesKMeans(
        n_clusters=optimal_k,
        metric=args.metric,  # 使用命令行指定的距离度量
        n_jobs=args.n_jobs,     # 并行计算，限制CPU核心数
        n_init=args.n_init,       # 减少初始化次数
        max_iter=50,    # 适当增加最大迭代次数
        max_iter_barycenter=3,  # 大幅降低DBA迭代次数，默认是100
        tol=1e-3,       # 宽松的收敛阈值
        random_state=42,
        verbose=1       # 显示详细的迭代信息
    )
    
    cluster_labels = model.fit_predict(X)
    
    elapsed = time.time() - start_time
    
    print(f"=== 聚类完成 ===")
    print(f"  总耗时: {elapsed:.1f}秒")
    print(f"  迭代次数: {model.n_iter_}")
    print(f"  惯性值: {model.inertia_:.2f}")
    
    # 统计聚类分布
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique, counts))
    print(f"  聚类分布: {cluster_distribution}")
    
    # 计算轮廓系数
    silhouette = silhouette_score(X.reshape(X.shape[0], -1), cluster_labels)
    print(f"  轮廓系数: {silhouette:.3f}")
    
    return model, cluster_labels

def visualize_clusters(model, cluster_labels, X):
    """
    可视化聚类结果
    
    Args:
        model: 训练好的聚类模型
        cluster_labels: 聚类标签
        X: 时间序列数据
    """
    print(f"\n=== 开始可视化聚类结果 ===")
    
    optimal_k = model.n_clusters
    centers = model.cluster_centers_  # 形状: (K, T, 4)
    
    # 设置中文显示，支持跨平台
    plt.rcParams['font.sans-serif'] = [
        'STHeiti',           # macOS 黑体
        'WenQuanYi Zen Hei', # Linux 文泉驿正黑
        'SimHei',            # Windows 黑体
        'DejaVu Sans'        # 后备字体
    ]  # 用于显示中文，按平台优先级排序
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    
    channel_names = [
        "上行延迟 (ms)",
        "下行延迟 (ms)", 
        "上行丢包率",
        "下行丢包率"
    ]
    
    # 为每个簇绘制典型行为
    for cluster_id in range(optimal_k):
        plt.figure(figsize=(16, 4))
        
        for channel_idx in range(4):
            plt.subplot(1, 4, channel_idx + 1)
            
            # 绘制该簇的中心曲线
            plt.plot(centers[cluster_id, :, channel_idx], 
                    color='red', linewidth=3, label='簇中心')
            
            # 随机绘制该簇中的3个样本
            cluster_samples = X[cluster_labels == cluster_id]
            n_samples_to_plot = min(3, len(cluster_samples))
            
            for i in range(n_samples_to_plot):
                plt.plot(cluster_samples[i, :, channel_idx], 
                        alpha=0.3, label=f'样本 {i+1}')
            
            plt.title(f'簇 {cluster_id + 1} - {channel_names[channel_idx]}')
            plt.xlabel('时间点')
            plt.ylabel(channel_names[channel_idx])
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path("output/soft_dtw_clustering")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"cluster_{cluster_id + 1}_behavior.png", 
                   dpi=300, bbox_inches="tight")
        print(f"  簇 {cluster_id + 1} 可视化保存到: {output_dir / f'cluster_{cluster_id + 1}_behavior.png'}")
        plt.close()
    
    # 绘制所有簇中心对比
    plt.figure(figsize=(16, 4))
    
    for channel_idx in range(4):
        plt.subplot(1, 4, channel_idx + 1)
        
        for cluster_id in range(optimal_k):
            plt.plot(centers[cluster_id, :, channel_idx], 
                    linewidth=2, label=f'簇 {cluster_id + 1}')
        
        plt.title(f'所有簇 - {channel_names[channel_idx]}')
        plt.xlabel('时间点')
        plt.ylabel(channel_names[channel_idx])
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_clusters_centers.png", 
               dpi=300, bbox_inches="tight")
    print(f"  所有簇中心对比图保存到: {output_dir / 'all_clusters_centers.png'}")
    plt.close()
    
    # 添加降维可视化
    print(f"\n=== 开始降维可视化 ===")
    
    # 将时间序列展平：(n_samples, T, 4) -> (n_samples, T*4)
    X_flat = X.reshape(X.shape[0], -1)
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    
    # 创建颜色映射（使用新API避免警告）
    colors = plt.colormaps['tab10']
    
    # 1. PCA降维可视化
    print(f"  执行PCA降维...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_flat)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   color=colors(cluster_id), 
                   label=f'簇 {cluster_id + 1}', 
                   alpha=0.6, s=50)
    
    # 绘制簇中心的PCA投影
    centers_flat = centers.reshape(optimal_k, -1)
    centers_pca = pca.transform(centers_flat)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               color='black', 
               marker='X', 
               s=200, 
               label='簇中心')
    
    plt.title('PCA降维可视化 - 网络行为聚类结果')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_clustering.png", 
               dpi=300, bbox_inches="tight")
    print(f"  PCA降维可视化保存到: {output_dir / 'pca_clustering.png'}")
    plt.close()
    
    # 2. t-SNE降维可视化
    print(f"  执行t-SNE降维...")
    # 为t-SNE设置合适的perplexity值，确保小于样本数量
    tsne_perplexity = min(30, len(X_flat) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_flat)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   color=colors(cluster_id), 
                   label=f'簇 {cluster_id + 1}', 
                   alpha=0.6, s=50)
    
    # 绘制簇中心的t-SNE投影
    # 使用已训练好的t-SNE模型转换簇中心，而不是重新训练
    centers_tsne = tsne.fit_transform(np.concatenate([X_flat[:100], centers_flat]))[-optimal_k:]
    plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], 
               color='black', 
               marker='X', 
               s=200, 
               label='簇中心')
    
    plt.title('t-SNE降维可视化 - 网络行为聚类结果')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_clustering.png", 
               dpi=300, bbox_inches="tight")
    print(f"  t-SNE降维可视化保存到: {output_dir / 'tsne_clustering.png'}")
    plt.close()
    
    # 3. UMAP降维可视化
    print(f"  执行UMAP降维...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_flat)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(optimal_k):
        mask = cluster_labels == cluster_id
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1], 
                   color=colors(cluster_id), 
                   label=f'簇 {cluster_id + 1}', 
                   alpha=0.6, s=50)
    
    # 绘制簇中心的UMAP投影
    centers_umap = reducer.transform(centers_flat)
    plt.scatter(centers_umap[:, 0], centers_umap[:, 1], 
               color='black', 
               marker='X', 
               s=200, 
               label='簇中心')
    
    plt.title('UMAP降维可视化 - 网络行为聚类结果')
    plt.xlabel('UMAP维度1')
    plt.ylabel('UMAP维度2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "umap_clustering.png", 
               dpi=300, bbox_inches="tight")
    print(f"  UMAP降维可视化保存到: {output_dir / 'umap_clustering.png'}")
    plt.close()
    
    print(f"=== 降维可视化完成 ===")
    print(f"=== 所有可视化完成 ===")

def save_clustering_results(model, cluster_labels, X):
    """
    保存聚类结果
    
    Args:
        model: 训练好的聚类模型
        cluster_labels: 聚类标签
        X: 时间序列数据
    """
    print(f"\n=== 保存聚类结果 ===")
    
    output_dir = Path("output/soft_dtw_clustering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存聚类标签
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    
    # 保存簇中心
    np.save(output_dir / "cluster_centers.npy", model.cluster_centers_)
    
    # 生成聚类统计信息
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = {
        "cluster_counts": dict(zip(unique.tolist(), counts.tolist())),
        "n_clusters": model.n_clusters,
        "inertia": model.inertia_,
        "n_iter": model.n_iter_
    }
    
    with open(output_dir / "clustering_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
    
    # 生成CSV报告
    summary = []
    for idx, (seq, label) in enumerate(zip(X, cluster_labels)):
        summary.append({
            "sample_id": idx,
            "cluster": int(label),
            "mean_up_delay": float(np.mean(seq[:, 0])),
            "p95_up_delay": float(np.percentile(seq[:, 0], 95)),
            "mean_down_delay": float(np.mean(seq[:, 1])),
            "p95_down_delay": float(np.percentile(seq[:, 1], 95)),
            "mean_up_loss": float(np.mean(seq[:, 2])),
            "mean_down_loss": float(np.mean(seq[:, 3])),
            "max_up_loss_burst": float(np.max([len(list(g)) for k, g in groupby(seq[:, 2]) if k == 1] or [0])),
            "max_down_loss_burst": float(np.max([len(list(g)) for k, g in groupby(seq[:, 3]) if k == 1] or [0]))
        })
    
    df = pd.DataFrame(summary)
    df.to_csv(output_dir / "network_behavior_clusters.csv", index=False, encoding="utf-8-sig")
    
    print(f"  聚类标签保存到: {output_dir / 'cluster_labels.npy'}")
    print(f"  簇中心保存到: {output_dir / 'cluster_centers.npy'}")
    print(f"  聚类统计保存到: {output_dir / 'clustering_stats.json'}")
    print(f"  聚类报告保存到: {output_dir / 'network_behavior_clusters.csv'}")
    print(f"  报告前5行:")
    print(df.head())
    
    print(f"=== 结果保存完成 ===")

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    print("# TimeSeriesKMeans + Soft-DTW 网络行为聚类测试")
    print("=" * 60)
    
    # 1. 加载预处理数据
    print("\n1. 加载预处理数据...")
    preprocessed_data = load_preprocessed_data(
        downsample_length=args.downsample_length,
        downsample_method=args.downsample_method,
        sample_limit=args.sample_limit
    )
    
    # 2. 准备时间序列数据
    print("\n2. 准备时间序列数据...")
    X = prepare_time_series_data(preprocessed_data)
    
    # 3. 选择最优K值或使用固定K值
    if args.fixed_k is not None:
        optimal_k = args.fixed_k
        print(f"\n3. 使用固定K值: {optimal_k}")
    else:
        optimal_k = select_optimal_k(X, max_k=args.max_k)
    
    # 4. 执行聚类
    print(f"\n4. 执行聚类...")
    model, cluster_labels = perform_clustering(X, optimal_k, args)
    
    # 5. 可视化聚类结果
    print("\n5. 可视化聚类结果...")
    visualize_clusters(model, cluster_labels, X)
    
    # 6. 保存聚类结果
    print("\n6. 保存聚类结果...")
    save_clustering_results(model, cluster_labels, X)
    
    print("\n" + "=" * 60)
    print("# 聚类测试完成！")
    print(f"结果保存在: output/soft_dtw_clustering/")

# 添加groupby导入，用于计算丢包突发长度
from itertools import groupby

if __name__ == "__main__":
    main()
