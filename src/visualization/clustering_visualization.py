#!/usr/bin/env python3
"""聚类可视化模块，用于生成聚类结果的可视化图表"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path


class ClusteringVisualizer:
    """聚类可视化器，用于生成聚类结果的可视化图表"""
    
    def __init__(self, output_dir: Path):
        """初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体支持 - 优化字体配置，确保中文正常显示
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.constrained_layout.use'] = True
    
    def plot_clustering_metrics(self, k_range, inertias, silhouettes, davies_bouldins, best_k):
        """绘制聚类指标图表
        
        Args:
            k_range: K值范围
            inertias: 不同K值下的惯性列表
            silhouettes: 不同K值下的轮廓系数列表
            davies_bouldins: 不同K值下的Davies-Bouldin指数列表
            best_k: 最优K值
        """
        print("\n绘制聚类指标图表...")
        
        # 创建一个包含3个子图的图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 肘部法则图（惯性）
        axes[0].plot(k_range, inertias, marker='o', linestyle='-', color='b')
        axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'最优K={best_k}')
        axes[0].set_xlabel('K值')
        axes[0].set_ylabel('惯性（Inertia）')
        axes[0].set_title('肘部法则（Elbow Method）')
        axes[0].grid(True)
        axes[0].legend()
        
        # 2. 轮廓系数图
        axes[1].plot(k_range, silhouettes, marker='o', linestyle='-', color='g')
        axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'最优K={best_k}')
        axes[1].set_xlabel('K值')
        axes[1].set_ylabel('轮廓系数（Silhouette Score）')
        axes[1].set_title('轮廓系数图')
        axes[1].grid(True)
        axes[1].legend()
        
        # 3. Davies-Bouldin指数图
        axes[2].plot(k_range, davies_bouldins, marker='o', linestyle='-', color='purple')
        axes[2].axvline(x=best_k, color='r', linestyle='--', label=f'最优K={best_k}')
        axes[2].set_xlabel('K值')
        axes[2].set_ylabel('Davies-Bouldin指数')
        axes[2].set_title('Davies-Bouldin指数图')
        axes[2].grid(True)
        axes[2].legend()
        
        # 保存图表
        output_path = self.output_dir / 'clustering_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"聚类指标图表已保存到: {output_path}")
        plt.close()
    
    def plot_cluster_visualization(self, features_scaled, cluster_labels, best_k):
        """绘制聚类结果的降维可视化
        
        Args:
            features_scaled: 标准化后的特征数组
            cluster_labels: 聚类标签数组
            best_k: 最优K值
        """
        print("\n绘制聚类结果可视化...")
        
        # 创建一个包含2个子图的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 使用tab10颜色映射，与整合可视化保持一致
        cmap = plt.get_cmap('tab10', best_k)
        
        # 1. PCA降维可视化
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features_scaled)
        scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap=cmap, s=50, alpha=0.6)
        ax1.set_xlabel('PCA维度1')
        ax1.set_ylabel('PCA维度2')
        ax1.set_title(f'PCA降维可视化（K={best_k}）')
        plt.colorbar(scatter1, ax=ax1, label='聚类标签')
        ax1.grid(True)
        
        # 2. t-SNE降维可视化
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        tsne_result = tsne.fit_transform(features_scaled)
        scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap=cmap, s=50, alpha=0.6)
        ax2.set_xlabel('t-SNE维度1')
        ax2.set_ylabel('t-SNE维度2')
        ax2.set_title(f't-SNE降维可视化（K={best_k}）')
        plt.colorbar(scatter2, ax=ax2, label='聚类标签')
        ax2.grid(True)
        
        # 保存图表
        output_path = self.output_dir / 'cluster_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"聚类可视化图表已保存到: {output_path}")
        plt.close()
    
    def generate_clustering_visualizations(self, features, cluster_labels, k_range, inertias, silhouettes, davies_bouldins, best_k):
        """生成所有聚类可视化图表
        
        Args:
            features: 特征数组
            cluster_labels: 聚类标签数组
            k_range: K值范围
            inertias: 不同K值下的惯性列表
            silhouettes: 不同K值下的轮廓系数列表
            davies_bouldins: 不同K值下的Davies-Bouldin指数列表
            best_k: 最优K值
        """
        print("\n=== 生成聚类可视化图表 ===")
        
        # 绘制聚类指标图表
        self.plot_clustering_metrics(k_range, inertias, silhouettes, davies_bouldins, best_k)
        
        # 绘制聚类结果可视化
        self.plot_cluster_visualization(features, cluster_labels, best_k)
        
        print("\n=== 聚类可视化完成 ===")
    
    def generate_integrated_visualization(self, features, cluster_labels, best_k, window_data, qt_up, qt_down):
        """生成整合的聚类可视化，包含降维图、典型样本图和统计信息
        
        Args:
            features: 特征数组
            cluster_labels: 聚类标签数组
            best_k: 最优K值
            window_data: 窗口数据列表
            qt_up: 上行QuantileTransformer模型
            qt_down: 下行QuantileTransformer模型
        """
        import pandas as pd
        
        print("\n=== 生成整合聚类可视化 ===")
        
        # 选择典型案例（每类两个，优先包含下行丢包）
        typical_cases = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # 找出当前聚类的所有窗口
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # 收集所有窗口，优先选择包含下行丢包的窗口
            cases_for_cluster = []
            all_cases = []
            
            for idx in cluster_indices:
                if idx < len(window_data):
                    window_df = pd.DataFrame(window_data[idx]["window"])
                    if len(window_df) > 0:
                        case_info = {
                            "window": window_data[idx],
                            "cluster_id": cluster_id,
                            "has_loss_dn": np.any(window_df["loss_dn"].values > 0)
                        }
                        all_cases.append(case_info)
            
            # 首先选择所有包含下行丢包的窗口
            loss_dn_cases = [case for case in all_cases if case["has_loss_dn"]]
            # 然后选择不包含下行丢包的窗口
            no_loss_cases = [case for case in all_cases if not case["has_loss_dn"]]
            
            # 合并列表，优先保留包含下行丢包的窗口
            combined_cases = loss_dn_cases + no_loss_cases
            
            # 随机选择4个窗口，如果不足则使用所有可用窗口
            import random
            random.shuffle(combined_cases)
            selected_cases = combined_cases[:4]
            
            # 如果仍然不足4个，使用所有窗口
            if len(selected_cases) > 0:
                typical_cases[cluster_id] = selected_cases
        
        # 创建一个大图，容纳聚类图和每个聚类的四个典型案例（每类两行，每行两个案例）
        num_clusters = len(unique_clusters)
        total_rows = 1 + num_clusters * 2  # 1行聚类图 + 每类2行典型案例
        fig = plt.figure(figsize=(18, 20 + num_clusters * 6))
        
        # 第1行：显示聚类降维图和统计信息（横跨2列）
        ax_cluster = fig.add_subplot(total_rows, 1, 1)
        self._plot_integrated_cluster_visualization(ax_cluster, features, cluster_labels, best_k)
        
        # 绘制典型案例：每类两行，每行两个案例
        current_row = 2
        for cluster_id, cases_for_cluster in typical_cases.items():
            # 第1行：案例1和案例2
            if current_row <= total_rows:
                ax1 = fig.add_subplot(total_rows, 2, (current_row-1)*2 + 1)
                self._plot_typical_case_integrated(ax1, cases_for_cluster[0], qt_up, qt_down, cluster_id, best_k, case_idx=1)
                
                if len(cases_for_cluster) > 1:
                    ax2 = fig.add_subplot(total_rows, 2, (current_row-1)*2 + 2)
                    self._plot_typical_case_integrated(ax2, cases_for_cluster[1], qt_up, qt_down, cluster_id, best_k, case_idx=2)
            
            # 第2行：案例3和案例4
            if current_row + 1 <= total_rows:
                if len(cases_for_cluster) > 2:
                    ax3 = fig.add_subplot(total_rows, 2, current_row*2 + 1)
                    self._plot_typical_case_integrated(ax3, cases_for_cluster[2], qt_up, qt_down, cluster_id, best_k, case_idx=3)
                
                if len(cases_for_cluster) > 3:
                    ax4 = fig.add_subplot(total_rows, 2, current_row*2 + 2)
                    self._plot_typical_case_integrated(ax4, cases_for_cluster[3], qt_up, qt_down, cluster_id, best_k, case_idx=4)
            
            # 增加行数
            current_row += 2
        
        # 保存整合图表
        output_path = self.output_dir / 'integrated_clustering_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"整合聚类可视化已保存到: {output_path}")
        plt.close()
        
        print("\n=== 整合聚类可视化完成 ===")
    
    def _plot_integrated_cluster_visualization(self, ax, features_scaled, cluster_labels, best_k):
        """绘制整合的聚类降维图
        
        Args:
            ax: matplotlib轴对象
            features_scaled: 标准化后的特征数组
            cluster_labels: 聚类标签数组
            best_k: 最优K值
        """
        # 创建一个包含2个子图的图形
        ax1 = ax.inset_axes([0, 0.5, 0.5, 0.5])  # PCA图
        ax2 = ax.inset_axes([0.5, 0.5, 0.5, 0.5])  # t-SNE图
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'聚类结果可视化（K={best_k}）', fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
         
        # 使用更美观的配色方案 - 更换为tab10配色
        cmap = plt.get_cmap('tab10', best_k)  # 改用tab10配色，更加清晰美观
        
        # 1. PCA降维可视化
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(features_scaled)
        scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap=cmap, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax1.set_xlabel('PCA维度1')
        ax1.set_ylabel('PCA维度2')
        ax1.set_title(f'PCA降维可视化（K={best_k}）')
        ax1.grid(True, alpha=0.3)
        
        # 2. t-SNE降维可视化
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        tsne_result = tsne.fit_transform(features_scaled)
        scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap=cmap, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax2.set_xlabel('t-SNE维度1')
        ax2.set_ylabel('t-SNE维度2')
        ax2.set_title(f't-SNE降维可视化（K={best_k}）')
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条和统计信息
        cbar_ax = ax.inset_axes([0.1, 0.05, 0.3, 0.3])
        cbar = plt.colorbar(scatter2, cax=cbar_ax, orientation='vertical', label='聚类标签')
        cbar_ax.set_title('聚类标签')
        
        # 计算聚类统计信息
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        cluster_stats = {cluster_id: count for cluster_id, count in zip(unique_clusters, counts)}
        
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(features_scaled, cluster_labels)
        
        # 添加统计信息
        stats_ax = ax.inset_axes([0.5, 0.05, 0.4, 0.3])
        stats_ax.set_xticks([])
        stats_ax.set_yticks([])
        stats_ax.spines['top'].set_visible(False)
        stats_ax.spines['right'].set_visible(False)
        stats_ax.spines['bottom'].set_visible(False)
        stats_ax.spines['left'].set_visible(False)
        
        # 绘制统计信息
        stats_text = f"聚类统计信息\n\n"
        stats_text += f"最佳K值: {best_k}\n"
        stats_text += f"轮廓系数: {silhouette:.4f}\n\n"
        stats_text += "样本分布:\n"
        for cluster_id in sorted(unique_clusters):
            count = cluster_stats[cluster_id]
            percentage = (count / len(cluster_labels)) * 100
            stats_text += f"类别 {cluster_id}: {count}个 ({percentage:.1f}%)\n"
        
        stats_ax.text(0.05, 0.95, stats_text, fontsize=12, verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        stats_ax.set_title('聚类统计')
    
    def _plot_typical_case_integrated(self, ax, case_data, qt_up, qt_down, cluster_id, best_k, case_idx=1):
        """绘制典型案例的时延和卡顿情况（整合版）
        
        Args:
            ax: matplotlib轴对象
            case_data: 典型案例数据
            qt_up: 上行QuantileTransformer模型
            qt_down: 下行QuantileTransformer模型
            cluster_id: 聚类ID
            best_k: 最优K值
            case_idx: 案例索引，用于区分同一类的不同案例
        """
        # 提取窗口数据
        window_df = pd.DataFrame(case_data["window"]["window"])
        # 只取前100个点
        window_df = window_df.head(100)
        
        # 反归一化函数
        def inverse_transform_delay(delays, qt_model, mean=0.0, std=1.0):
            # 1. 先进行z-score反变换
            delays_zscore_inv = delays * std + mean
            # 2. 确保输入是二维数组
            delays_2d = delays_zscore_inv.reshape(-1, 1)
            # 3. 反QuantileTransformer变换
            inverse_delays = qt_model.inverse_transform(delays_2d)
            # 4. 转换回一维数组
            return inverse_delays.flatten()
        
        # 反归一化时延数据
        del_up = window_df["del_up"].values
        del_dn = window_df["del_dn"].values
        loss_up = window_df["loss_up"].values
        loss_dn = window_df["loss_dn"].values
        
        inverse_del_up = inverse_transform_delay(del_up, qt_up)
        inverse_del_dn = inverse_transform_delay(del_dn, qt_down)
        
        # 使用tab10颜色映射，与聚类图保持一致
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('tab10', best_k)
        cluster_color = cmap(cluster_id % best_k)
        
        # 绘制上行和下行时延
        ax.plot(inverse_del_up, label="上行时延 (ms)", color=cluster_color, linewidth=2)
        ax.plot(inverse_del_dn, label="下行时延 (ms)", color=cluster_color, linewidth=2, linestyle='--')
        
        # 添加卡顿标记（丢包时）
        loss_up_indices = np.where(loss_up > 0)[0]
        loss_dn_indices = np.where(loss_dn > 0)[0]
        
        if len(loss_up_indices) > 0:
            ax.scatter(loss_up_indices, inverse_del_up[loss_up_indices], 
                      color="red", marker="x", s=100, label="上行丢包")
        if len(loss_dn_indices) > 0:
            ax.scatter(loss_dn_indices, inverse_del_dn[loss_dn_indices], 
                      color="purple", marker="x", s=100, label="下行丢包")
        
        # 设置统一的坐标范围
        ax.set_xlabel("时间点 (100ms间隔)")
        ax.set_ylabel("时延 (ms)")
        
        # 统一时延坐标范围：0-2000ms，调整半对数刻度使0-100ms分布更合理
        ax.set_ylim(0, 2000)
        ax.set_yscale('symlog', linthresh=100, linscale=0.8)  # 0-100ms线性刻度，100ms以上对数刻度，增大linscale使小值区域更宽松
        # 添加自定义y轴刻度：0、100、200、1000、2000毫秒
        ax.set_yticks([0, 100, 200, 1000, 2000])
        ax.set_yticklabels(['0', '100', '200', '1000', '2000'])  # 确保显示正确的刻度标签
        
        # 添加次要y轴显示丢包率
        ax_loss = ax.twinx()
        # 绘制丢包率
        ax_loss.plot(loss_up, label="上行丢包率", color="red", linestyle=':', linewidth=1.5, alpha=0.6)
        ax_loss.plot(loss_dn, label="下行丢包率", color="purple", linestyle=':', linewidth=1.5, alpha=0.6)
        # 统一丢包率坐标范围
        ax_loss.set_ylim(-0.01, 1.01)
        ax_loss.set_ylabel("丢包率")
        
        # 合并图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_loss.get_legend_handles_labels()
        ax_loss.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # 设置标题
        if case_idx == 1:
            ax.set_title(f"类别 {cluster_id} - 典型案例 A", fontsize=14, color=cluster_color)
        else:
            ax.set_title(f"类别 {cluster_id} - 典型案例 B", fontsize=14, color=cluster_color)
        
        ax.grid(True, alpha=0.3)
        
        # 在图中添加类别颜色标记
        ax_inset = ax.inset_axes([0.8, 0.8, 0.15, 0.15])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['bottom'].set_visible(False)
        ax_inset.spines['left'].set_visible(False)
        ax_inset.set_facecolor(cluster_color)
        ax_inset.text(0.5, 0.5, f"类别 {cluster_id}", transform=ax_inset.transAxes, 
                     ha='center', va='center', fontsize=12, color='white', fontweight='bold')
