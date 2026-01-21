#!/usr/bin/env python3
"""
批量分析聚类策略 —— 适配你的 build_train_dataset.py 输出
从 dataset/train_dataset.pkl 中读取数据
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 配置
# -----------------------------
STRATEGIES = [
    "gmm_three_class",
    "auto_gmm_bic",
    "hierarchical",
    "kmeans_two_stage",
    "isolation_forest_two_stage",
    "dbscan",
]
DATASET_DIR = "dataset"  # ← 你的 OUTPUT_DIR
PLOT_DIR = "cluster_analysis_batch"
os.makedirs(PLOT_DIR, exist_ok=True)


def inverse_log_transform(x):
    return np.expm1(x)


def extract_enhanced_features(windows):
    """从窗口中提取增强特征"""
    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    total_delay = (UL_delay + DL_delay) / 2.0
    total_loss = (UL_loss + DL_loss) / 2.0

    features = {
        "mean_delay": np.mean(total_delay, axis=1),
        "delay_jitter": np.std(total_delay, axis=1),
        "mean_loss_pct": np.mean(total_loss, axis=1) * 100,
        "burst_freq_pct": np.mean(total_loss > 0.05, axis=1) * 100,
    }
    return pd.DataFrame(features)


def load_data_and_labels(strategy):
    """从 train_dataset.pkl 加载 windows + labels"""
    pkl_path = os.path.join(DATASET_DIR, f"train_dataset_{strategy}.pkl")
    if not os.path.exists(pkl_path):
        # 兼容你当前的命名（无后缀）
        pkl_path = os.path.join(DATASET_DIR, "train_dataset.pkl")
        if not os.path.exists(pkl_path):
            print(f"⚠️ 跳过 {strategy}：文件不存在")
            return None, None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 合并 train + test
    windows = np.concatenate(
        [data["train"]["windows"], data["test"]["windows"]], axis=0
    )
    labels = data["train"]["labels"] + data["test"]["labels"]

    return windows, labels


def analyze_single_strategy(strategy):
    print(f"\n🔍 分析策略: {strategy}")

    windows, labels = load_data_and_labels(strategy)
    if windows is None:
        return None

    df_features = extract_enhanced_features(windows)
    df_features["label"] = labels
    df_features["strategy"] = strategy

    # 保存 CSV
    csv_path = os.path.join(PLOT_DIR, f"clustered_data_{strategy}.csv")
    df_features.to_csv(csv_path, index=False)

    # t-SNE
    feature_cols = ["mean_delay", "delay_jitter", "mean_loss_pct", "burst_freq_pct"]
    X = df_features[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    df_features["tsne_x"] = X_tsne[:, 0]
    df_features["tsne_y"] = X_tsne[:, 1]

    # 单图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_features,
        x="tsne_x",
        y="tsne_y",
        hue="label",
        palette=["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"],
        s=8,
        alpha=0.6,
    )
    plt.title(f"{strategy}", fontsize=14)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"tsne_{strategy}.png"), dpi=150)
    plt.close()

    print(f"✅ 完成 {strategy}")
    return df_features


def plot_comparison(all_dfs):
    n = len(all_dfs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (strategy, df) in enumerate(all_dfs.items()):
        ax = axes[idx]
        sns.scatterplot(
            data=df,
            x="tsne_x",
            y="tsne_y",
            hue="label",
            palette=["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"],
            s=6,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(strategy, fontsize=12)
        ax.legend().set_visible(False)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "tsne_comparison_all.png"), dpi=150)
    plt.close()


def generate_summary(all_dfs):
    rows = []
    for strategy, df in all_dfs.items():
        for label in df["label"].unique():
            sub = df[df["label"] == label]
            rows.append(
                {
                    "Strategy": strategy,
                    "Class": label,
                    "Count": len(sub),
                    "Mean_Delay": sub["mean_delay"].mean(),
                    "Delay_Jitter": sub["delay_jitter"].mean(),
                    "Mean_Loss_Pct": sub["mean_loss_pct"].mean(),
                    "Burst_Freq_Pct": sub["burst_freq_pct"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(PLOT_DIR, "summary_statistics.csv"), index=False
    )


def main():
    print("🚀 批量分析启动...")

    # 为每个策略生成数据（注意：你当前所有策略共用同一个 train_dataset.pkl）
    # 所以我们只加载一次，但按策略名保存（实际标签不同）
    all_dfs = {}
    for strategy in STRATEGIES:
        # 临时 hack：假设你已为每个策略单独运行并保存了 train_dataset.pkl
        # 更好的做法是修改 build_train_dataset.py 输出带策略名的文件
        df = analyze_single_strategy(strategy)
        if df is not None:
            all_dfs[strategy] = df

    if not all_dfs:
        print("❌ 无数据")
        return

    plot_comparison(all_dfs)
    generate_summary(all_dfs)
    print(f"\n🎉 结果保存至: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
