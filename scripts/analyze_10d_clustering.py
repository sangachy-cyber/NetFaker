#!/usr/bin/env python3
"""
使用10D特征进行聚类分析，并统计train/test/val集的分布

该脚本直接处理数据，使用10D特征进行聚类，生成state_id，并统计不同数据集的分布情况。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置中文字体支持，确保中文能正常显示
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['PingFang SC', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.labelsize': 12,
    'axes.titlesize': 14
})

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _max_consecutive(arr):
    """计算数组中连续True的最大长度
    
    Args:
        arr: 布尔数组
        
    Returns:
        int: 最大连续True的长度
    """
    if not arr.any():
        return 0
    
    # 计算连续True的长度
    consecutive_counts = []
    count = 0
    for val in arr:
        if val:
            count += 1
        else:
            if count > 0:
                consecutive_counts.append(count)
                count = 0
    if count > 0:
        consecutive_counts.append(count)
    
    return max(consecutive_counts) if consecutive_counts else 0


def _compute_features(window_df):
    """计算窗口的特征（用于聚类）
    
    Args:
        window_df: 窗口DataFrame
        
    Returns:
        20维特征数组 [raw_mean_delay_up, p95_up, p1_up, std_up, trend_slope_up, 
                      autocorr_lag5_up, loss_up, max_consec_loss_up, 
                      max_burst_up, n_switches_up,
                      raw_mean_delay_down, p95_down, p1_down, std_down, trend_slope_down,
                      autocorr_lag5_down, loss_down, max_consec_loss_down,
                      max_burst_down, n_switches_down]
    """
    # 使用上下行数据
    # 优先使用原始的delay_up和delay_down列，如果不存在则使用del_up和del_dn列
    if "delay_up" in window_df.columns:
        del_up = window_df["delay_up"].values
    else:
        del_up = window_df["del_up"].values
    if "delay_down" in window_df.columns:
        del_down = window_df["delay_down"].values
    else:
        del_down = window_df["del_dn"].values
    
    # 检查时延是否为负，如果为负则抛出异常
    if np.any(del_up < 0):
        raise ValueError(f"上行延迟数据中存在负值：{del_up[del_up < 0]}")
    if np.any(del_down < 0):
        raise ValueError(f"下行延迟数据中存在负值：{del_down[del_down < 0]}")
    
    # 检查原始延迟是否真的 <= 2000
    if np.max(del_up) > 2000:
        print(f"警告：发现上行延迟 >2000ms！max={np.max(del_up)}")
    if np.max(del_down) > 2000:
        print(f"警告：发现下行延迟 >2000ms！max={np.max(del_down)}")
    
    # 强制将延迟限制在 [0, 2000] ms
    del_up = np.clip(del_up, 0, 2000)
    del_down = np.clip(del_down, 0, 2000)
    
    # 检查时延平均值是否大于10ms，如果小于则记录警告
    avg_delay_up = np.mean(del_up)
    if avg_delay_up < 10:
        print(f"警告：上行延迟平均值小于10ms：{avg_delay_up:.2f}ms")
    avg_delay_down = np.mean(del_down)
    if avg_delay_down < 10:
        print(f"警告：下行延迟平均值小于10ms：{avg_delay_down:.2f}ms")
    
    loss_up = window_df["loss_up"].values
    loss_down = window_df["loss_dn"].values
    
    # 获取时间戳序列
    timestamps = window_df["timestamp"].values
    
    # 验证时间戳是否单调递增
    if not np.all(np.diff(timestamps) >= 0):
        print("警告：时间戳非单调！")
        timestamps = np.sort(timestamps)  # 或跳过该窗口
    
    # 辅助函数：计算延迟趋势斜率
    from scipy.stats import linregress
    def compute_trend_slope(delays, timestamps):
        """计算延迟趋势斜率
        
        Args:
            delays: 延迟序列（ms）
            timestamps: 时间戳序列（秒）
            
        Returns:
            趋势斜率（ms/s）
        """
        if len(delays) < 2:
            return 0.0
        # 使用真实时间差作为x轴，而不是采样点索引
        x = timestamps
        # 归一化x轴到[0, window_duration]，单位：秒
        x_normalized = x - x[0]
        slope, _, _, _, _ = linregress(x_normalized, delays)
        return slope
    
    # 辅助函数：计算延迟序列的自相关系数
    def compute_autocorr(delays, lag=5):
        """计算延迟序列的自相关系数"""
        if len(delays) <= lag:
            return 0.0
        return np.corrcoef(delays[:-lag], delays[lag:])[0, 1] if np.std(delays) > 0 else 0.0
    
    # 辅助函数：计算最大连续高延迟段
    def compute_max_burst(delays):
        """计算最大连续高延迟段（秒）"""
        if len(delays) == 0:
            return 0.0
        # 高延迟阈值：90%分位数
        high_delay_threshold = np.percentile(delays, 90)
        high_delay_mask = delays > high_delay_threshold
        return _max_consecutive(high_delay_mask) / 10.0  # 每10个点=1秒
    
    # 辅助函数：计算状态切换次数
    def compute_n_switches(delays, window_size=20, threshold_pct=95):
        """计算状态切换次数"""
        if len(delays) < window_size * 2:
            return 0
        # 计算滑动窗口方差
        import pandas as pd
        window_std = pd.Series(delays).rolling(window_size).std().fillna(0).values
        # 计算方差差异
        diff_std = np.abs(np.diff(window_std))
        # 设置阈值
        threshold = np.percentile(diff_std, threshold_pct)
        # 统计突变次数
        return (diff_std > threshold).sum()
    
    # === 上行特征计算 ===
    # 1. raw_mean_delay_up: 原始上行延迟序列的均值
    raw_mean_delay_up = np.mean(del_up)
    
    # 2. p95_up: 上行延迟的95%分位数
    p95_up = np.percentile(del_up, 95)
    
    # 3. p1_up: 上行延迟的1%分位数
    p1_up = np.percentile(del_up, 1)
    
    # 4. std_up: 上行延迟的标准差
    std_up = np.std(del_up)
    
    # 5. trend_slope_up: 上行延迟趋势斜率（ms/s）
    trend_slope_up = compute_trend_slope(del_up, timestamps)
    
    # 6. autocorr_lag5_up: 上行延迟的自相关系数（lag=5）
    autocorr_lag5_up = compute_autocorr(del_up)
    
    # 7. loss_up: 上行平均丢包率
    loss_up_mean = np.mean(loss_up)
    
    # 8. max_consec_loss_up: 上行最长连续丢包窗口数 → 转秒
    max_consec_loss_up = _max_consecutive(loss_up > 0) / 10.0  # 每10个点=1秒
    
    # 9. max_burst_up: 上行最长连续高延迟段（秒）
    max_burst_up = compute_max_burst(del_up)
    
    # 10. n_switches_up: 上行滑动窗口方差突变次数
    n_switches_up = compute_n_switches(del_up)
    
    # === 下行特征计算 ===
    # 11. raw_mean_delay_down: 原始下行延迟序列的均值
    raw_mean_delay_down = np.mean(del_down)
    
    # 12. p95_down: 下行延迟的95%分位数
    p95_down = np.percentile(del_down, 95)
    
    # 13. p1_down: 下行延迟的1%分位数
    p1_down = np.percentile(del_down, 1)
    
    # 14. std_down: 下行延迟的标准差
    std_down = np.std(del_down)
    
    # 15. trend_slope_down: 下行延迟趋势斜率（ms/s）
    trend_slope_down = compute_trend_slope(del_down, timestamps)
    
    # 16. autocorr_lag5_down: 下行延迟的自相关系数（lag=5）
    autocorr_lag5_down = compute_autocorr(del_down)
    
    # 17. loss_down: 下行平均丢包率
    loss_down_mean = np.mean(loss_down)
    
    # 18. max_consec_loss_down: 下行最长连续丢包窗口数 → 转秒
    max_consec_loss_down = _max_consecutive(loss_down > 0) / 10.0  # 每10个点=1秒
    
    # 19. max_burst_down: 下行最长连续高延迟段（秒）
    max_burst_down = compute_max_burst(del_down)
    
    # 20. n_switches_down: 下行滑动窗口方差突变次数
    n_switches_down = compute_n_switches(del_down)
    
    # 构建20维特征向量
    features = np.array([
        # 上行特征
        raw_mean_delay_up,
        p95_up,
        p1_up,
        std_up,
        trend_slope_up,
        autocorr_lag5_up,
        loss_up_mean,
        max_consec_loss_up,
        max_burst_up,
        n_switches_up,
        # 下行特征
        raw_mean_delay_down,
        p95_down,
        p1_down,
        std_down,
        trend_slope_down,
        autocorr_lag5_down,
        loss_down_mean,
        max_consec_loss_down,
        max_burst_down,
        n_switches_down
    ])
    
    # === 安全 Clip（防御性）===
    # 上行延迟类特征：raw_mean, p95, p1, std
    features[0:4] = np.clip(features[0:4], 0, 2000)
    # 上行趋势斜率：限制在±50 ms/s，符合真实网络
    features[4] = np.clip(features[4], -50, 50)
    # 上行自相关系数：限制在[-1, 1]，理论范围
    features[5] = np.clip(features[5], -1, 1)
    # 上行丢包率：限制在[0, 1]，理论范围
    features[6] = np.clip(features[6], 0, 1)
    
    # 下行延迟类特征：raw_mean, p95, p1, std
    features[10:14] = np.clip(features[10:14], 0, 2000)
    # 下行趋势斜率：限制在±50 ms/s，符合真实网络
    features[14] = np.clip(features[14], -50, 50)
    # 下行自相关系数：限制在[-1, 1]，理论范围
    features[15] = np.clip(features[15], -1, 1)
    # 下行丢包率：限制在[0, 1]，理论范围
    features[16] = np.clip(features[16], 0, 1)
    
    # === 非线性压缩（避免 NaN）===
    # 上行延迟统计量：非负，安全使用log1p
    features[0:4] = np.log1p(features[0:4])
    # 上行其他非负特征：loss, max_consec_loss, max_burst, n_switches
    features[6:10] = np.log1p(features[6:10])
    
    # 下行延迟统计量：非负，安全使用log1p
    features[10:14] = np.log1p(features[10:14])
    # 下行其他非负特征：loss, max_consec_loss, max_burst, n_switches
    features[16:20] = np.log1p(features[16:20])
    
    # 趋势斜率：带符号的log压缩，保留方向
    # 上行趋势斜率
    slope_up = features[4]
    features[4] = np.sign(slope_up) * np.log1p(np.abs(slope_up) + 1e-8)  # 加epsilon防止log(0)
    # 下行趋势斜率
    slope_down = features[14]
    features[14] = np.sign(slope_down) * np.log1p(np.abs(slope_down) + 1e-8)  # 加epsilon防止log(0)
    
    # 确保所有值都是有限的，替换NaN和无穷大
    features = np.where(np.isfinite(features), features, 0.0)
    
    return features


def load_processed_data():
    """加载处理后的数据
    
    Returns:
        (window_dfs, trace_ids, split_labels): 窗口DataFrame列表、trace_id列表、数据集划分标签列表
    """
    import pandas as pd
    
    # 从预处理后的JSONL文件加载数据
    data_dir = Path("output/datasets")
    if not data_dir.exists():
        raise FileNotFoundError(f"预处理后的数据目录不存在: {data_dir}")
    
    window_dfs = []
    trace_ids = []
    split_labels = []
    
    # 遍历所有数据集文件
    for split in ["train", "val", "test"]:
        file_path = data_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        print(f"加载{split}集数据: {file_path}")
        
        # 读取JSONL文件
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                window_df = pd.DataFrame(record["window"])
                window_dfs.append(window_df)
                trace_ids.append(record["trace_id"])
                split_labels.append(split)
    
    print(f"总共加载了 {len(window_dfs)} 个窗口")
    return window_dfs, trace_ids, split_labels


def _calculate_optimal_k(features_scaled, k_range=(2, 10)):
    """计算最优的聚类数量K
    
    Args:
        features_scaled: 标准化后的特征数组
        k_range: K值的范围，默认为2到10
    
    Returns:
        tuple: (best_k, inertias, silhouettes, davies_bouldins)
            best_k: 最优的K值
            inertias: 不同K值下的惯性列表
            silhouettes: 不同K值下的轮廓系数列表
            davies_bouldins: 不同K值下的Davies-Bouldin指数列表
    """
    print(f"\n计算最优K值（范围：{k_range[0]}-{k_range[1]}）...")
    
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    # 遍历K值范围，计算不同K值下的聚类指标
    for k in range(k_range[0], k_range[1]+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 计算惯性（inertia）
        inertias.append(kmeans.inertia_)
        
        # 计算轮廓系数（silhouette score）
        if k > 1:
            silhouette = silhouette_score(features_scaled, cluster_labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
        
        # 计算Davies-Bouldin指数
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        davies_bouldins.append(davies_bouldin)
        
        print(f"K={k}: 惯性={kmeans.inertia_:.2f}, 轮廓系数={silhouette:.4f}, Davies-Bouldin指数={davies_bouldin:.4f}")
    
    # 选择最优K值
    # 1. 轮廓系数最高的K值
    best_k_silhouette = k_range[0] + np.argmax(silhouettes)
    
    # 2. Davies-Bouldin指数最低的K值
    best_k_db = k_range[0] + np.argmin(davies_bouldins)
    
    # 3. 肘部法则（找到惯性下降速度明显变慢的点）
    # 计算相邻K值之间的惯性差值
    inertia_diffs = np.diff(inertias)
    # 计算差值的变化率
    inertia_diff_ratios = np.abs(np.diff(inertia_diffs))
    # 找到变化率最大的点，即为肘部
    if len(inertia_diff_ratios) > 0:
        best_k_elbow = k_range[0] + np.argmax(inertia_diff_ratios) + 1
    else:
        best_k_elbow = best_k_silhouette
    
    # 综合考虑，优先选择轮廓系数最高的K值
    best_k = best_k_silhouette
    
    print(f"\n最优K值选择：")
    print(f"- 轮廓系数最高: K={best_k_silhouette}")
    print(f"- Davies-Bouldin指数最低: K={best_k_db}")
    print(f"- 肘部法则: K={best_k_elbow}")
    print(f"\n综合选择最优K值: {best_k}")
    
    return best_k, inertias, silhouettes, davies_bouldins


def _plot_clustering_metrics(k_values, inertias, silhouettes, davies_bouldins, best_k, output_dir):
    """绘制聚类指标图表
    
    Args:
        k_values: K值列表
        inertias: 不同K值下的惯性列表
        silhouettes: 不同K值下的轮廓系数列表
        davies_bouldins: 不同K值下的Davies-Bouldin指数列表
        best_k: 最优K值
        output_dir: 输出目录
    """
    print(f"\n绘制聚类指标图表...")
    
    # 使用全局字体设置，无需重复配置
    
    # 创建一个包含3个子图的图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Elbow Method (Inertia)
    axes[0].plot(k_values, inertias, marker='o', linestyle='-', color='b')
    axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. Silhouette Score
    axes[1].plot(k_values, silhouettes, marker='o', linestyle='-', color='g')
    axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score')
    axes[1].grid(True)
    axes[1].legend()
    
    # 3. Davies-Bouldin Index
    axes[2].plot(k_values, davies_bouldins, marker='o', linestyle='-', color='purple')
    axes[2].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title('Davies-Bouldin Index')
    axes[2].grid(True)
    axes[2].legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'clustering_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"聚类指标图表已保存到: {output_path}")
    plt.close()


def _plot_cluster_visualization(features_scaled, cluster_labels, best_k, output_dir):
    """绘制聚类结果的降维可视化
    
    Args:
        features_scaled: 标准化后的特征数组
        cluster_labels: 聚类标签数组
        best_k: 最优K值
        output_dir: 输出目录
    """
    print(f"\n绘制聚类结果可视化...")
    
    # 使用全局字体设置，无需重复配置
    
    # 创建一个包含2个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. PCA Dimensionality Reduction
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    ax1.set_xlabel('PCA Dimension 1')
    ax1.set_ylabel('PCA Dimension 2')
    ax1.set_title(f'PCA Visualization (K={best_k})')
    plt.colorbar(scatter1, ax=ax1, label='Cluster Label')
    ax1.grid(True)
    
    # 2. t-SNE Dimensionality Reduction
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_result = tsne.fit_transform(features_scaled)
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title(f't-SNE Visualization (K={best_k})')
    plt.colorbar(scatter2, ax=ax2, label='Cluster Label')
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / 'cluster_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"聚类可视化图表已保存到: {output_path}")
    plt.close()


def cluster_10d_features(window_dfs, k=None, k_range=(2, 10), output_dir=None):
    """使用10D特征进行聚类
    
    Args:
        window_dfs: 窗口DataFrame列表
        k: 聚类的K值，如果为None则自动确定最优K值
        k_range: 自动确定K值时的范围
        output_dir: 输出目录，用于保存可视化图表
    
    Returns:
        (cluster_labels, scaler, k): 聚类标签数组、用于标准化的Scaler对象和实际使用的K值
    """
    # 计算10D特征
    print("\n计算10D特征...")
    features = []
    
    # 遍历所有窗口，计算每个窗口的10D特征
    for window_df in window_dfs:
        window_features = _compute_features(window_df)
        features.append(window_features)
    
    features = np.array(features)
    print(f"特征形状: {features.shape}")
    

    
    # 校验：确保特征是基于原始延迟数据，而不是分位数归一化后的数据
    print(f"\n=== 特征校验 ===")
    print(f"特征统计（归一化前）：")
    print(f"  均值: {np.mean(features):.2f}")
    print(f"  中位数: {np.median(features):.2f}")
    print(f"  最小值: {np.min(features):.2f}")
    print(f"  最大值: {np.max(features):.2f}")
    print(f"  标准差: {np.std(features):.2f}")
    
    # 标准化特征（使用RobustScaler，抗异常值）
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"\n特征统计（RobustScaler归一化后）：")
    print(f"  均值: {np.mean(features_scaled):.2f}")
    print(f"  中位数: {np.median(features_scaled):.2f}")
    print(f"  最小值: {np.min(features_scaled):.2f}")
    print(f"  最大值: {np.max(features_scaled):.2f}")
    print(f"  标准差: {np.std(features_scaled):.2f}")
    
    # 检查各维度归一化后分布
    print(f"\n=== 各维度归一化后分布检查 ===")
    for i in range(features_scaled.shape[1]):
        col = features_scaled[:, i]
        q25, q75 = np.percentile(col, 25), np.percentile(col, 75)
        iqr = q75 - q25
        print(f"特征 {i}: min={col.min():.2f}, max={col.max():.2f}, "
              f"median={np.median(col):.2f}, IQR={iqr:.2f}")
    
    # 自动确定最优K值
    if k is None:
        best_k, inertias, silhouettes, davies_bouldins = _calculate_optimal_k(features_scaled, k_range)
        k = best_k
        
        # 生成K值列表
        k_values = list(range(k_range[0], k_range[1]+1))
        
        # 绘制聚类指标图表
        if output_dir:
            _plot_clustering_metrics(k_values, inertias, silhouettes, davies_bouldins, best_k, output_dir)
    
    # 执行KMeans聚类
    print(f"\n执行KMeans聚类，K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # 计算聚类质量指标（仅当样本数大于1且聚类数大于1时）
    if len(features_scaled) > 1 and k > 1:
        silhouette = silhouette_score(features_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_labels)
        
        print(f"聚类质量指标:")
        print(f"- 轮廓系数: {silhouette:.4f}")
        print(f"- Davies-Bouldin指数: {davies_bouldin:.4f}")
        print(f"- Calinski-Harabasz指数: {calinski_harabasz:.4f}")
    else:
        print("样本数或聚类数不足，无法计算聚类质量指标")
    
    # 绘制聚类结果可视化
    if output_dir:
        _plot_cluster_visualization(features_scaled, cluster_labels, k, output_dir)
    
    return cluster_labels, scaler, k


def analyze_distribution(cluster_labels, split_labels, k):
    """分析不同数据集的聚类分布
    
    Args:
        cluster_labels: 聚类标签数组
        split_labels: 数据集划分标签数组
        k: 聚类的K值
    
    Returns:
        dict: 不同数据集的聚类分布统计
    """
    print("\n=== 数据集分布统计 ===")
    
    # 确保输入是一维数组
    cluster_labels = np.array(cluster_labels).flatten()
    split_labels = np.array(split_labels).flatten()
    
    # 统计每个数据集的样本数
    unique_splits = np.unique(split_labels)
    distribution = {}
    
    for split in unique_splits:
        split_mask = split_labels == split
        split_labels_cluster = cluster_labels[split_mask]
        total_samples = len(split_labels_cluster)
        
        # 统计每个state_id的样本数
        # 使用np.unique代替np.bincount，处理可能的非连续标签
        unique_states, state_counts = np.unique(split_labels_cluster, return_counts=True)
        state_dist = {}
        
        for state_id, count in zip(unique_states, state_counts):
            percentage = (count / total_samples) * 100
            state_dist[int(state_id)] = {
                "count": int(count),
                "percentage": percentage
            }
        
        distribution[split] = {
            "total_samples": total_samples,
            "state_distribution": state_dist
        }
        
        # 打印当前数据集的分布
        print(f"\n{split.upper()}集:")
        print(f"  总样本数: {total_samples}")
        print(f"  状态ID分布:")
        for state_id in sorted(state_dist.keys()):
            stat = state_dist[state_id]
            print(f"    状态ID {state_id}: {stat['count']}个样本 ({stat['percentage']:.2f}%)")
    
    return distribution


def main(args):
    """主函数"""
    print("=== 10D特征聚类分析 ===")
    
    # 1. 加载数据
    window_dfs, trace_ids, split_labels = load_processed_data()
    
    # 2. 创建输出目录
    output_dir = Path("output/10d_clustering_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 使用10D特征进行聚类
    # 如果指定了--auto-k，则自动确定最优K值
    if args.auto_k:
        # 解析k_range参数
        k_range = tuple(map(int, args.k_range.split(',')))
        cluster_labels, scaler, k = cluster_10d_features(window_dfs, k=None, k_range=k_range, output_dir=output_dir)
    else:
        # 使用指定的K值
        cluster_labels, scaler, k = cluster_10d_features(window_dfs, k=args.k, output_dir=output_dir)
    
    # 4. 分析分布
    distribution = analyze_distribution(cluster_labels, split_labels, k)
    
    # 保存聚类结果
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    np.save(output_dir / "split_labels.npy", split_labels)
    
    # 保存分布统计
    with open(output_dir / "distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_dir}")
    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用10D特征进行聚类分析")
    parser.add_argument('--k', type=int, default=6, help="聚类的K值")
    parser.add_argument('--auto-k', action='store_true', help="自动确定最优K值")
    parser.add_argument('--k-range', type=str, default="2,10", help="自动确定K值的范围，格式为'low,high'")
    
    args = parser.parse_args()
    
    main(args)
