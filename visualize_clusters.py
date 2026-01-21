import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns  # 新增这一行
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 支持通过环境变量指定输出目录，用于 compare_strategies.py
PLOT_DIR = os.getenv("CLUSTER_PLOT_DIR", "cluster_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ================== 你原有的函数（保持不变）==================
def plot_representative_samples(
    analysis_dir="clustering_analysis", output_dir="cluster_plots"
):
    """读取 samples_*.csv 并生成时间序列图"""
    os.makedirs(output_dir, exist_ok=True)
    label_types = ["S0", "S1_light", "S1_anomaly"]
    
    # 优化中文显示和分辨率
    plt.rcParams.update({
        'font.family': 'Arial Unicode MS',  # 支持中文
        'font.size': 10,  # 增大基础字体
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,  # 提高默认分辨率
    })
    
    for label in label_types:
        csv_path = os.path.join(analysis_dir, f"samples_{label}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 跳过 {label}: 文件不存在")
            continue
        df = pd.read_csv(csv_path)
        sample_ids = df["sample_id"].unique()
        if len(sample_ids) == 0:
            continue
        n_samples = min(len(sample_ids), 3)
        # 为每个样本创建 2x1 子图：上行和下行，增大尺寸
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples), dpi=300)
        if n_samples == 1:
            axes = [axes]
        fig.suptitle(f"Cluster: {label}", fontsize=18, fontweight="bold")
        for i, sid in enumerate(sample_ids[:3]):
            sample_df = df[df["sample_id"] == sid]
            
            # 绘制上行数据 (UL)
            ax_ul = axes[i][0]
            ax_ul.plot(
                sample_df["time_step"],
                sample_df["UL_delay_ms"],
                color="tab:blue",
                label="UL Delay (ms)",
                linewidth=1.5,  # 加粗线条
                marker='.',  # 添加标记点
                markersize=3,
            )
            ax_ul.set_ylabel("Delay (ms)", color="tab:blue", fontweight="bold")
            ax_ul.tick_params(axis="y", labelcolor="tab:blue")
            ax_ul.grid(True, alpha=0.3)  # 添加网格线
            
            ax2_ul = ax_ul.twinx()
            ax2_ul.plot(
                sample_df["time_step"],
                sample_df["UL_loss%"],
                color="tab:red",
                linestyle="--",
                label="UL Loss (%)",
                alpha=0.8,
                linewidth=1.5,  # 加粗线条
                marker='x',  # 添加标记点
                markersize=3,
            )
            ax2_ul.set_ylabel("Loss (%)", color="tab:red", fontweight="bold")
            ax2_ul.tick_params(axis="y", labelcolor="tab:red")
            ax2_ul.set_ylim(0, max(5, sample_df["UL_loss%"].max() * 1.1))
            ax_ul.set_title(f"Sample {sid} - UL", fontsize=12, fontweight="bold")  # 使用英文标题避免字体问题
            
            lines1, labels1 = ax_ul.get_legend_handles_labels()
            lines2, labels2 = ax2_ul.get_legend_handles_labels()
            ax_ul.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.8)
            
            # 绘制下行数据 (DL)
            ax_dl = axes[i][1]
            ax_dl.plot(
                sample_df["time_step"],
                sample_df["DL_delay_ms"],
                color="tab:green",
                label="DL Delay (ms)",
                linewidth=1.5,  # 加粗线条
                marker='.',  # 添加标记点
                markersize=3,
            )
            ax_dl.set_ylabel("Delay (ms)", color="tab:green", fontweight="bold")
            ax_dl.tick_params(axis="y", labelcolor="tab:green")
            ax_dl.grid(True, alpha=0.3)  # 添加网格线
            
            ax2_dl = ax_dl.twinx()
            ax2_dl.plot(
                sample_df["time_step"],
                sample_df["DL_loss%"],
                color="tab:purple",
                linestyle="--",
                label="DL Loss (%)",
                alpha=0.8,
                linewidth=1.5,  # 加粗线条
                marker='x',  # 添加标记点
                markersize=3,
            )
            ax2_dl.set_ylabel("Loss (%)", color="tab:purple", fontweight="bold")
            ax2_dl.tick_params(axis="y", labelcolor="tab:purple")
            ax2_dl.set_ylim(0, max(5, sample_df["DL_loss%"].max() * 1.1))
            ax_dl.set_title(f"Sample {sid} - DL", fontsize=12, fontweight="bold")  # 使用英文标题避免字体问题
            
            lines3, labels3 = ax_dl.get_legend_handles_labels()
            lines4, labels4 = ax2_dl.get_legend_handles_labels()
            ax_dl.legend(lines3 + lines4, labels3 + labels4, loc="upper right", framealpha=0.8)
            
            # 设置 x 轴标签
            if i == n_samples - 1:
                ax_ul.set_xlabel("Time Step", fontweight="bold")
                ax_dl.set_xlabel("Time Step", fontweight="bold")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 只生成 PNG 格式，适合屏幕查看
        output_path_png = os.path.join(output_dir, f"{label}_samples.png")
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成: {output_path_png}")


def plot_summary_comparison(analysis_dir="clustering_analysis", output_dir="cluster_plots"):
    """生成一个对比图：S0 / S1_light / S1_anomaly 各取1个窗口"""
    label_types = ["S0", "S1_light", "S1_anomaly"]
    data = {}
    for label in label_types:
        csv_path = os.path.join(analysis_dir, f"samples_{label}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            sample_df = df[df["sample_id"] == df["sample_id"].min()]
            data[label] = sample_df
    if not data:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Cluster Comparison: One Representative Window per Class", fontsize=16)
    colors = {"S0": "green", "S1_light": "orange", "S1_anomaly": "red"}
    for i, (label, df) in enumerate(data.items()):
        axes[i].plot(
            df["time_step"],
            df["UL_delay_ms"],
            color=colors[label],
            label=f"{label} UL Delay",
            linewidth=1.5,
        )
        axes[i].plot(
            df["time_step"],
            df["DL_delay_ms"],
            color=colors[label],
            linestyle=":",
            label=f"{label} DL Delay",
            linewidth=1.5,
        )
        axes[i].set_ylabel("Delay (ms)")
        axes[i].set_title(label, fontweight="bold", color=colors[label])
        axes[i].legend(loc="upper right")
        if i == 2:
            axes[i].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, "cluster_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 已生成对比图: {output_path}")


# ================== 新增：UMAP 聚类结构可视化 ==================
def inverse_log_transform(log_vals):
    return np.exp(log_vals) - 1.0


def extract_gmm_features(windows):
    """提取与GMM训练时相同的4D特征：上下行延迟的均值和标准差
    
    参数:
        windows: 形状为(N, T, 4)的数组，包含N个窗口，每个窗口有T个时间步
    
    返回:
        形状为(N, 4)的特征数组，每个样本包含4个特征：
        [上行延迟均值, 上行延迟标准差, 下行延迟均值, 下行延迟标准差]
    """
    features = []
    for win in windows:
        d1_log, p1, d2_log, p2 = win.T
        d1 = inverse_log_transform(d1_log)  # 转换回原始延迟
        d2 = inverse_log_transform(d2_log)
        feat = [
            np.mean(d1),
            np.std(d1),
            np.mean(d2),
            np.std(d2),
        ]
        features.append(feat)
    return np.array(features)


def extract_features(windows):
    features = []
    for win in windows:
        d1_log, p1, d2_log, p2 = win.T
        d1 = inverse_log_transform(d1_log)
        d2 = inverse_log_transform(d2_log)
        feat = [
            np.mean(d1),
            np.std(d1),
            np.mean(p1),
            np.sum(p1 > 0.05),
            np.mean(d2),
            np.std(d2),
            np.mean(p2),
            np.sum(p2 > 0.05),
        ]
        features.append(feat)
    return np.array(features)


def plot_umap_clustering(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    model_path="dataset/cluster_models_gmm_three_class.pkl",
    output_dir="cluster_plots",
    sample_limit=1000,  # 每种类型最多使用1000个窗口进行可视化，None表示不限制
):
    """新增：绘制 UMAP 聚类结构图
    
    参数:
        sample_limit: 每种类型最多使用的窗口数量，默认1000，None表示不限制
    """
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        with open(model_path, "rb") as f:
            cluster_model = pickle.load(f)

        train_windows = dataset["train"]["windows"]
        train_labels = dataset["train"]["labels"]

        # 样本限制逻辑
        if sample_limit is not None:
            print(f"🔄 应用样本限制：每种类型最多 {sample_limit} 个窗口")
            
            # 按标签分组
            label_groups = {}
            for i, label in enumerate(train_labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(i)
            
            # 对每组进行采样
            sampled_indices = []
            for label, indices in label_groups.items():
                if len(indices) > sample_limit:
                    # 随机采样sample_limit个索引
                    np.random.seed(42)  # 固定随机种子，确保结果可复现
                    sampled_indices.extend(np.random.choice(indices, size=sample_limit, replace=False))
                else:
                    sampled_indices.extend(indices)
            
            # 对采样后的索引进行排序，保持原始顺序
            sampled_indices.sort()
            
            # 应用采样
            train_windows = train_windows[sampled_indices]
            train_labels = [train_labels[i] for i in sampled_indices]
            
            print(f"📊 采样后窗口数: {len(train_windows)}")
            print(f"   标签统计: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

        # 提取特征并标准化
        X = extract_features(train_windows)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # UMAP 降维
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=30, min_dist=0.1
        )
        embedding = reducer.fit_transform(X_scaled)

        # 优化图表设置
        plt.rcParams.update({
            'font.family': 'Arial Unicode MS',  # 支持中文
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })

        # 图1: S0 vs S1
        s0_s1_labels = ["S1" if label != "S0" else "S0" for label in train_labels]
        plt.figure(figsize=(10, 8), dpi=300)
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=s0_s1_labels,
            palette=["#1f77b4", "#ff7f0e"],  # 更鲜明的颜色
            s=50,  # 增大点大小
            alpha=0.7,  # 提高不透明度
            linewidth=0.5,  # 添加点边框
            edgecolor='w',
        )
        plt.title("First-stage: S0 vs S1", fontsize=18, fontweight="bold")
        plt.xlabel("UMAP Component 1", fontweight="bold")
        plt.ylabel("UMAP Component 2", fontweight="bold")
        plt.legend(title="Class", fontsize=12, title_fontsize=14, markerscale=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "umap_s0_vs_s1.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 图2: 最终三类
        plt.figure(figsize=(10, 8), dpi=300)
        palette = {"S0": "#1f77b4", "S1_light": "#ff7f0e", "S1_anomaly": "#d62728"}  # 更鲜明的颜色
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=train_labels,
            palette=palette,
            s=50,  # 增大点大小
            alpha=0.7,  # 提高不透明度
            linewidth=0.5,  # 添加点边框
            edgecolor='w',
        )
        plt.title("Final Labels", fontsize=18, fontweight="bold")
        plt.xlabel("UMAP Component 1", fontweight="bold")
        plt.ylabel("UMAP Component 2", fontweight="bold")
        plt.legend(title="Class", fontsize=12, title_fontsize=14, markerscale=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "umap_final_labels.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 图3: S1 子簇（合并前） - 仅适用于两阶段策略
        s1_indices = [i for i, lbl in enumerate(train_labels) if lbl != "S0"]
        if s1_indices:
            # 检查模型类型，只对两阶段模型绘制子簇图
            if isinstance(cluster_model, dict) and "stage2" in cluster_model:
                s1_windows = [train_windows[i] for i in s1_indices]
                X2 = extract_features(s1_windows)
                X2_scaled = cluster_model["stage2"]["scaler"].transform(X2)
                sublabels = cluster_model["stage2"]["model"].predict(X2_scaled)

                plt.figure(figsize=(10, 8), dpi=300)
                sns.scatterplot(
                    x=embedding[s1_indices, 0],
                    y=embedding[s1_indices, 1],
                    hue=sublabels,
                    palette="tab10",
                    s=50,  # 增大点大小
                    alpha=0.7,  # 提高不透明度
                    linewidth=0.5,  # 添加点边框
                    edgecolor='w',
                )
                plt.title("S1 Sub-clusters (Before Merging)", fontsize=18, fontweight="bold")
                plt.xlabel("UMAP Component 1", fontweight="bold")
                plt.ylabel("UMAP Component 2", fontweight="bold")
                plt.legend(title="Sub-cluster", fontsize=12, title_fontsize=14, markerscale=2)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "umap_s1_subclusters.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                print("  ⚠️ 跳过 S1 子簇图：仅适用于两阶段聚类模型")

        print("✅ 已生成 UMAP 聚类图:")
        print(f"  - {output_dir}/umap_s0_vs_s1.png")
        print(f"  - {output_dir}/umap_s1_subclusters.png")
        print(f"  - {output_dir}/umap_final_labels.png")

    except Exception as e:
        print(f"⚠️ UMAP 可视化失败（跳过）: {e}")

# ========== 新增：分析 GMM 模糊类型 ==========
def analyze_gmm_fuzzy_types(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    model_path="dataset/cluster_models_gmm_three_class.pkl",
    output_dir="cluster_plots",
    prob_threshold=0.25,
):
    """分析 GMM 聚类中的模糊样本，并按模糊类型分组"""
    import pickle
    import numpy as np
    from collections import Counter
    from sklearn.preprocessing import StandardScaler

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    with open(model_path, "rb") as f:
        gmm_model = pickle.load(f)  # 可能是直接的GMM或字典结构

    windows = dataset["train"]["windows"]  # shape: (N, T, 4)
    true_labels = np.array(dataset["train"]["labels"])

    # === 关键修复：必须使用与训练时相同的特征提取 ===
    features = extract_gmm_features(windows)  # 使用与GMM训练时相同的4D特征！
    
    # 标准化特征用于后续UMAP可视化
    scaler_for_umap = StandardScaler()
    features_scaled = scaler_for_umap.fit_transform(features)

    # 处理不同类型的模型结构
    posterior_probs = None
    semantic_labels = []
    
    if isinstance(gmm_model, dict):
        # 两阶段聚类模型结构
        print(f"📦 检测到两阶段模型结构，包含: {list(gmm_model.keys())}")
        
        # 第一阶段：S0 vs S1
        if "stage1" in gmm_model and hasattr(gmm_model["stage1"]["model"], "predict_proba"):
            # 使用第一阶段的模型和scaler
            scaler1 = gmm_model["stage1"]["scaler"]
            model1 = gmm_model["stage1"]["model"]
            features_scaled1 = scaler1.transform(features)
            probs1 = model1.predict_proba(features_scaled1)
            
            # 第二阶段：S1_light vs S1_anomaly
            if "stage2" in gmm_model and hasattr(gmm_model["stage2"]["model"], "predict_proba"):
                scaler2 = gmm_model["stage2"]["scaler"]
                model2 = gmm_model["stage2"]["model"]
                
                # 对S1样本应用第二阶段模型
                s1_mask = probs1[:, 1] > 0.5  # 假设1是S1类
                features_s1 = features[s1_mask]
                features_scaled2 = scaler2.transform(features_s1)
                probs2 = model2.predict_proba(features_scaled2)
                
                # 组合概率：3类概率
                posterior_probs = np.zeros((len(features), 3))
                posterior_probs[:, 0] = probs1[:, 0]  # S0概率
                
                # 对于S0样本，S1_light和S1_anomaly概率设为低
                posterior_probs[~s1_mask, 1] = 0.1
                posterior_probs[~s1_mask, 2] = 0.1
                
                # 对于S1样本，根据第二阶段概率分配
                posterior_probs[s1_mask, 1] = probs1[s1_mask, 1] * probs2[:, 0]  # S1_light
                posterior_probs[s1_mask, 2] = probs1[s1_mask, 1] * probs2[:, 1]  # S1_anomaly
        
        # 标准化后验概率，确保每一行和为1
        if posterior_probs is not None:
            row_sums = posterior_probs.sum(axis=1, keepdims=True)
            posterior_probs = posterior_probs / row_sums
            
            # 生成语义标签
            for i in range(len(posterior_probs)):
                if posterior_probs[i, 0] > 0.5:
                    semantic_labels.append("S0")
                elif posterior_probs[i, 1] > posterior_probs[i, 2]:
                    semantic_labels.append("S1_light")
                else:
                    semantic_labels.append("S1_anomaly")
    else:
        # 直接的GMM模型（三类或四类）
        print("📦 检测到直接GMM模型")
        
        # 使用与训练时相同的标准化
        scaler = StandardScaler()
        features_scaled_for_model = scaler.fit_transform(features)  # 由于未保存训练时的scaler，这里重新拟合
        
        # 获取后验概率
        if hasattr(gmm_model, "predict_proba"):
            posterior_probs = gmm_model.predict_proba(features_scaled_for_model)
        else:
            raise ValueError("模型不支持predict_proba方法")
        
        # 生成语义标签 - true_labels已经是语义标签
        semantic_labels = list(true_labels)  # 直接使用已有的语义标签
    
    # 确保后验概率已正确生成
    if posterior_probs is None:
        raise ValueError("无法生成后验概率，请检查模型结构")

    fuzzy_types = []
    n_components = posterior_probs.shape[1]  # 获取GMM的组件数
    
    for p in posterior_probs:
        active_idx = tuple(np.where(p > prob_threshold)[0])
        if len(active_idx) <= 1:
            fuzzy_types.append("clear")
        elif len(active_idx) == 2:
            # 两模糊类型
            names = tuple(sorted(["S0", "S1_light", "S1_anomaly", "S2"][i] for i in active_idx))
            fuzzy_types.append(f"{names[0]}-{names[1]}")
        elif len(active_idx) == 3:
            # 三模糊类型
            fuzzy_types.append("all_three")
        elif len(active_idx) >= 4:
            # 四模糊类型（针对四类GMM）
            fuzzy_types.append("all_four")
        else:
            fuzzy_types.append("clear")

    counter = Counter(fuzzy_types)
    total_samples = len(fuzzy_types)
    fuzzy_samples = total_samples - counter.get("clear", 0)
    
    print("\n🔍 Fuzzy Type Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Clear samples: {counter.get('clear', 0)} ({counter.get('clear', 0)/total_samples*100:.1f}%)")
    print(f"  Fuzzy samples: {fuzzy_samples} ({fuzzy_samples/total_samples*100:.1f}%)")
    print(f"  GMM components: {n_components}")
    
    if fuzzy_samples > 0:
        print("  Fuzzy type breakdown:")
        for typ, cnt in sorted(counter.items()):
            if typ != "clear":
                print(f"    - {typ}: {cnt} samples ({cnt/total_samples*100:.1f}%)")
    else:
        print("  No fuzzy samples found!")
    
    print(f"  Probability threshold: {prob_threshold:.2f}")

    return features_scaled, posterior_probs, semantic_labels, fuzzy_types, counter

# ========== 新增：多方法可视化 ==========
def plot_pca_clustering(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    output_dir="cluster_plots",
    sample_limit=1000,
):
    """使用PCA可视化聚类结果"""
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        windows = dataset["train"]["windows"]
        labels = dataset["train"]["labels"]
        
        # 样本限制
        if sample_limit is not None:
            label_groups = {}
            for i, label in enumerate(labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(i)
            
            sampled_indices = []
            for label, indices in label_groups.items():
                if len(indices) > sample_limit:
                    np.random.seed(42)
                    sampled_indices.extend(np.random.choice(indices, size=sample_limit, replace=False))
                else:
                    sampled_indices.extend(indices)
            
            sampled_indices.sort()
            windows = windows[sampled_indices]
            labels = [labels[i] for i in sampled_indices]
        
        # 提取特征并标准化
        features = extract_features(windows)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA 降维
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(features_scaled)
        
        # 可视化
        plt.figure(figsize=(10, 8), dpi=300)
        palette = {"S0": "#1f77b4", "S1_light": "#2ca02c", "S1_anomaly": "#d62728"}
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels,
            palette=palette,
            s=50,
            alpha=0.7,
            linewidth=0.5,
            edgecolor='w',
        )
        plt.title("PCA Clustering Results", fontsize=18, fontweight="bold")
        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontweight="bold")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontweight="bold")
        plt.legend(title="Class", fontsize=12, title_fontsize=14, markerscale=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "pca_clustering.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成 PCA 聚类图: {output_path}")
        
    except Exception as e:
        print(f"⚠️ PCA 可视化失败（跳过）: {e}")


def plot_tsne_clustering(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    output_dir="cluster_plots",
    sample_limit=1000,
):
    """使用t-SNE可视化聚类结果"""
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        windows = dataset["train"]["windows"]
        labels = dataset["train"]["labels"]
        
        # 样本限制
        if sample_limit is not None:
            label_groups = {}
            for i, label in enumerate(labels):
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(i)
            
            sampled_indices = []
            for label, indices in label_groups.items():
                if len(indices) > sample_limit:
                    np.random.seed(42)
                    sampled_indices.extend(np.random.choice(indices, size=sample_limit, replace=False))
                else:
                    sampled_indices.extend(indices)
            
            sampled_indices.sort()
            windows = windows[sampled_indices]
            labels = [labels[i] for i in sampled_indices]
        
        # 提取特征并标准化
        features = extract_features(windows)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
        embedding = tsne.fit_transform(features_scaled)
        
        # 可视化
        plt.figure(figsize=(10, 8), dpi=300)
        palette = {"S0": "#1f77b4", "S1_light": "#2ca02c", "S1_anomaly": "#d62728"}
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=labels,
            palette=palette,
            s=50,
            alpha=0.7,
            linewidth=0.5,
            edgecolor='w',
        )
        plt.title("t-SNE Clustering Results", fontsize=18, fontweight="bold")
        plt.xlabel("t-SNE Component 1", fontweight="bold")
        plt.ylabel("t-SNE Component 2", fontweight="bold")
        plt.legend(title="Class", fontsize=12, title_fontsize=14, markerscale=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "tsne_clustering.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成 t-SNE 聚类图: {output_path}")
        
    except Exception as e:
        print(f"⚠️ t-SNE 可视化失败（跳过）: {e}")


def plot_pairplot(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    output_dir="cluster_plots",
    sample_limit=500,
):
    """绘制特征散点图矩阵"""
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        windows = dataset["train"]["windows"]
        labels = dataset["train"]["labels"]
        
        # 样本限制
        if sample_limit is not None and len(windows) > sample_limit:
            np.random.seed(42)
            sampled_indices = np.random.choice(len(windows), size=sample_limit, replace=False)
            windows = windows[sampled_indices]
            labels = [labels[i] for i in sampled_indices]
        
        # 提取特征
        features = extract_features(windows)
        
        # 创建DataFrame
        feature_names = [
            "UL_Delay_Mean", "UL_Delay_Std", "UL_Loss_Mean", "UL_Loss_Count",
            "DL_Delay_Mean", "DL_Delay_Std", "DL_Loss_Mean", "DL_Loss_Count"
        ]
        
        df = pd.DataFrame(features, columns=feature_names)
        df["Label"] = labels
        
        # 绘制散点图矩阵
        g = sns.pairplot(df, hue="Label", palette={"S0": "#1f77b4", "S1_light": "#2ca02c", "S1_anomaly": "#d62728"},
                        diag_kind="kde", markers=".", plot_kws={"s": 30, "alpha": 0.7})
        
        g.fig.suptitle("Feature Pairplot", fontsize=18, fontweight="bold", y=1.02)
        g.fig.set_size_inches(15, 15)
        
        output_path = os.path.join(output_dir, "pairplot.png")
        g.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成特征散点图矩阵: {output_path}")
        
    except Exception as e:
        print(f"⚠️ 散点图矩阵可视化失败（跳过）: {e}")


def plot_density_plots(
    dataset_path="dataset/train_dataset_gmm_three_class.pkl",
    output_dir="cluster_plots",
):
    """绘制特征密度图"""
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        
        windows = dataset["train"]["windows"]
        labels = dataset["train"]["labels"]
        
        # 提取特征
        features = extract_features(windows)
        
        # 创建DataFrame
        feature_names = [
            "UL_Delay_Mean", "UL_Delay_Std", "UL_Loss_Mean", "UL_Loss_Count",
            "DL_Delay_Mean", "DL_Delay_Std", "DL_Loss_Mean", "DL_Loss_Count"
        ]
        
        df = pd.DataFrame(features, columns=feature_names)
        df["Label"] = labels
        
        # 绘制密度图
        fig, axes = plt.subplots(4, 2, figsize=(15, 20), dpi=300)
        axes = axes.ravel()
        
        for i, feature in enumerate(feature_names):
            for label in ["S0", "S1_light", "S1_anomaly"]:
                subset = df[df["Label"] == label]
                sns.kdeplot(subset[feature], ax=axes[i], label=label, linewidth=2)
            
            axes[i].set_title(f"Density of {feature}", fontsize=14, fontweight="bold")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Density")
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "density_plots.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已生成特征密度图: {output_path}")
        
    except Exception as e:
        print(f"⚠️ 密度图可视化失败（跳过）: {e}")


def plot_tsne_with_fuzzy_types(output_dir="cluster_plots", dataset_path=None, model_path=None):
    """在t-SNE图上绘制模糊类型"""
    if dataset_path is None:
        dataset_path = "dataset/train_dataset_gmm_three_class.pkl"
    if model_path is None:
        model_path = "dataset/cluster_models_gmm_three_class.pkl"
    
    features, probs, labels, fuzzy_types, _ = analyze_gmm_fuzzy_types(
        dataset_path=dataset_path, 
        model_path=model_path, 
        output_dir=output_dir
    )
    
    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(features)
    
    # 准备绘图标签 - 清晰类也区分三种类型
    plot_labels = []
    for ft, lbl in zip(fuzzy_types, labels):
        if ft == "clear":
            plot_labels.append(f"Clear: {lbl}")
        elif ft == "all_three":
            plot_labels.append("Fuzzy: All Three")
        elif ft == "all_four":
            plot_labels.append("Fuzzy: All Four")
        else:
            plot_labels.append(f"Fuzzy: {ft}")
    
    # 合并稀有模糊类型（避免图例太乱）
    unique_types = list(set(plot_labels))
    if len(unique_types) > 10:  # 增加阈值，因为四类GMM可能产生更多模糊类型
        # 只保留清晰类和主要模糊类型，其余归为 "Other Fuzzy"
        major_types = ["Clear: S0", "Clear: S1_light", "Clear: S1_anomaly"]
        plot_labels = [
            lbl if lbl in major_types or lbl.startswith("Fuzzy: S0-S1") else "Fuzzy: Other" for lbl in plot_labels
        ]
    
    # 使用动态生成的调色板，支持所有模糊类型
    fixed_colors = {
        "Clear: S0": "#1f77b4",  # 蓝色
        "Clear: S1_light": "#2ca02c",  # 绿色
        "Clear: S1_anomaly": "#d62728",  # 红色
    }
    
    # 收集所有标签并生成调色板
    all_labels = sorted(set(plot_labels))
    palette = {}
    
    # 为清晰类分配固定颜色
    for label in all_labels:
        if label in fixed_colors:
            palette[label] = fixed_colors[label]
    
    # 为模糊类分配颜色
    fuzzy_labels = [label for label in all_labels if label.startswith("Fuzzy:")]
    # 使用tab10调色板的剩余颜色
    fuzzy_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i, label in enumerate(fuzzy_labels):
        if label not in palette:
            palette[label] = fuzzy_colors[i % len(fuzzy_colors)]
    
    plt.figure(figsize=(12, 9), dpi=300)
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=plot_labels,
        palette=palette,
        s=30,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("t-SNE: Clear vs. Fuzzy Regions (GMM Posterior Analysis)", fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE Component 1", fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontweight="bold")
    plt.legend(title="Region Type", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "tsne_fuzzy_types.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved fuzzy-type t-SNE: {output_path}")


def plot_pca_with_fuzzy_types(output_dir="cluster_plots", dataset_path=None, model_path=None):
    """在PCA图上绘制模糊类型"""
    if dataset_path is None:
        dataset_path = "dataset/train_dataset_gmm_three_class.pkl"
    if model_path is None:
        model_path = "dataset/cluster_models_gmm_three_class.pkl"
    
    features, probs, labels, fuzzy_types, _ = analyze_gmm_fuzzy_types(
        dataset_path=dataset_path, 
        model_path=model_path, 
        output_dir=output_dir
    )
    
    # PCA 降维
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(features)
    
    # 准备绘图标签 - 清晰类也区分三种类型
    plot_labels = []
    for ft, lbl in zip(fuzzy_types, labels):
        if ft == "clear":
            plot_labels.append(f"Clear: {lbl}")
        elif ft == "all_three":
            plot_labels.append("Fuzzy: All Three")
        elif ft == "all_four":
            plot_labels.append("Fuzzy: All Four")
        else:
            plot_labels.append(f"Fuzzy: {ft}")
    
    # 合并稀有模糊类型（避免图例太乱）
    unique_types = list(set(plot_labels))
    if len(unique_types) > 10:  # 增加阈值，因为四类GMM可能产生更多模糊类型
        # 只保留清晰类和主要模糊类型，其余归为 "Other Fuzzy"
        major_types = ["Clear: S0", "Clear: S1_light", "Clear: S1_anomaly"]
        plot_labels = [
            lbl if lbl in major_types or lbl.startswith("Fuzzy: S0-S1") else "Fuzzy: Other" for lbl in plot_labels
        ]
    
    # 使用动态生成的调色板，支持所有模糊类型
    fixed_colors = {
        "Clear: S0": "#1f77b4",  # 蓝色
        "Clear: S1_light": "#2ca02c",  # 绿色
        "Clear: S1_anomaly": "#d62728",  # 红色
    }
    
    # 收集所有标签并生成调色板
    all_labels = sorted(set(plot_labels))
    palette = {}
    
    # 为清晰类分配固定颜色
    for label in all_labels:
        if label in fixed_colors:
            palette[label] = fixed_colors[label]
    
    # 为模糊类分配颜色
    fuzzy_labels = [label for label in all_labels if label.startswith("Fuzzy:")]
    # 使用tab10调色板的剩余颜色
    fuzzy_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i, label in enumerate(fuzzy_labels):
        if label not in palette:
            palette[label] = fuzzy_colors[i % len(fuzzy_colors)]
    
    plt.figure(figsize=(12, 9), dpi=300)
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=plot_labels,
        palette=palette,
        s=30,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("PCA: Clear vs. Fuzzy Regions (GMM Posterior Analysis)", fontsize=16, fontweight="bold")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontweight="bold")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontweight="bold")
    plt.legend(title="Region Type", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "pca_fuzzy_types.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved fuzzy-type PCA: {output_path}")


def plot_umap_with_fuzzy_types(output_dir="cluster_plots", dataset_path=None, model_path=None):
    """在 UMAP 上绘制模糊类型，突出边界区域"""
    if dataset_path is None:
        dataset_path = "dataset/train_dataset_gmm_three_class.pkl"
    if model_path is None:
        model_path = "dataset/cluster_models_gmm_three_class.pkl"
        
    features, probs, labels, fuzzy_types, _ = analyze_gmm_fuzzy_types(
        dataset_path=dataset_path, 
        model_path=model_path, 
        output_dir=output_dir
    )

    # 直接使用 analyze_gmm_fuzzy_types 返回的标准化特征，不需要再次提取
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    # 准备绘图标签 - 清晰类也区分三种类型
    plot_labels = []
    for ft, lbl in zip(fuzzy_types, labels):
        if ft == "clear":
            # 清晰类：区分三种类型
            plot_labels.append(f"Clear: {lbl}")
        elif ft == "all_three":
            plot_labels.append("Fuzzy: All Three")
        elif ft == "all_four":
            plot_labels.append("Fuzzy: All Four")
        else:
            plot_labels.append(f"Fuzzy: {ft}")

    # 合并稀有模糊类型（避免图例太乱）
    unique_types = list(set(plot_labels))
    if len(unique_types) > 10:  # 增加阈值，因为四类GMM可能产生更多模糊类型
        # 只保留清晰类和主要模糊类型，其余归为 "Other Fuzzy"
        major_types = ["Clear: S0", "Clear: S1_light", "Clear: S1_anomaly"]
        plot_labels = [
            lbl if lbl in major_types or lbl.startswith("Fuzzy: S0-S1") else "Fuzzy: Other" for lbl in plot_labels
        ]

    plt.rcParams.update({
        'font.family': 'Arial Unicode MS',
        'font.size': 10,
        'figure.dpi': 300
    })

    plt.figure(figsize=(12, 9))
    
    # 使用颜色循环而非固定调色板，支持所有模糊类型
    # 为清晰类设置固定颜色，模糊类使用颜色循环
    fixed_colors = {
        "Clear: S0": "#1f77b4",  # 蓝色
        "Clear: S1_light": "#2ca02c",  # 绿色
        "Clear: S1_anomaly": "#d62728",  # 红色
    }
    
    # 收集所有标签并生成调色板
    all_labels = sorted(set(plot_labels))
    palette = {}
    
    # 为清晰类分配固定颜色
    for label in all_labels:
        if label in fixed_colors:
            palette[label] = fixed_colors[label]
    
    # 为模糊类分配颜色
    fuzzy_labels = [label for label in all_labels if label.startswith("Fuzzy:")]
    # 使用tab10调色板的剩余颜色
    fuzzy_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i, label in enumerate(fuzzy_labels):
        if label not in palette:
            palette[label] = fuzzy_colors[i % len(fuzzy_colors)]
    
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=plot_labels,
        palette=palette,
        s=30,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("UMAP: Clear vs. Fuzzy Regions (GMM Posterior Analysis)", fontsize=16, fontweight="bold")
    plt.xlabel("UMAP Component 1", fontweight="bold")
    plt.ylabel("UMAP Component 2", fontweight="bold")
    plt.legend(title="Region Type", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "umap_fuzzy_types.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved fuzzy-type UMAP: {output_path}")

# ================== 主函数 ==================
if __name__ == "__main__":
    plot_representative_samples(output_dir=PLOT_DIR)
    print("\n🎉 所有聚类可视化已完成！")
    print("📁 输出目录: cluster_plots/")

    # 新增：调用 UMAP 可视化
    print("\n🔍 生成 UMAP 聚类结构图...")
    plot_umap_clustering(output_dir=PLOT_DIR, sample_limit=None)

    # 新增：多方法聚类可视化
    print("\n🔍 生成 PCA 聚类结构图...")
    plot_pca_clustering(output_dir=PLOT_DIR, sample_limit=None)
    
    print("\n🔍 生成 t-SNE 聚类结构图...")
    plot_tsne_clustering(output_dir=PLOT_DIR, sample_limit=1000)  # t-SNE计算量大，限制样本数
    
    print("\n🔍 生成特征散点图矩阵...")
    plot_pairplot(output_dir=PLOT_DIR, sample_limit=500)  # 散点图矩阵样本数限制
    
    print("\n🔍 生成特征密度图...")
    plot_density_plots(output_dir=PLOT_DIR)

    # ========== 新增：模糊类型分析 ==========
    print("\n🌫️  分析 GMM 模糊类型...")
    plot_umap_with_fuzzy_types(output_dir=PLOT_DIR)
    plot_pca_with_fuzzy_types(output_dir=PLOT_DIR)
    plot_tsne_with_fuzzy_types(output_dir=PLOT_DIR)
    
    print("\n🎉 所有聚类可视化已完成！")
    print("📁 输出目录: cluster_plots/")
