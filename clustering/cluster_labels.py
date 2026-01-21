# import numpy as np
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.ensemble import IsolationForest
# from sklearn.mixture import GaussianMixture
#
# from visualize_clusters import extract_features
#
#
# def inverse_log_transform(log_vals):
#     return np.exp(log_vals) - 1.0
#
# def extract_clustering_features(windows):
#     features = []
#     for win in windows:
#         d1_log, p1, d2_log, p2 = win.T
#         d1 = inverse_log_transform(d1_log)
#         d2 = inverse_log_transform(d2_log)
#         feat = [
#             np.mean(d1), np.std(d1),
#             np.mean(p1), np.sum(p1 > 0.05),
#             np.mean(d2), np.std(d2),
#             np.mean(p2), np.sum(p2 > 0.05)
#         ]
#         features.append(feat)
#     return np.array(features)
#
# # ==================== 策略1: 两阶段 KMeans (你的当前版本) ====================
# def strategy_kmeans_two_stage(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略1: 两阶段 KMeans")
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Stage 1: KMeans(2)
#     kmeans1 = KMeans(n_clusters=2, random_state=random_state, n_init=10)
#     stage1_labels = kmeans1.fit_predict(X_scaled)
#
#     # 假设簇0是正常（可通过质心延迟判断）
#     centroids = kmeans1.cluster_centers_
#     delay_means = []
#     for c in range(2):
#         mask = stage1_labels == c
#         if np.any(mask):
#             avg_delay = np.mean([
#                 np.mean(np.concatenate([inverse_log_transform(w[0]), inverse_log_transform(w[2])]))
#                 for w in train_windows[mask]
#             ])
#             delay_means.append((avg_delay, c))
#         else:
#             delay_means.append((np.inf, c))
#     delay_means.sort()
#     normal_cid = delay_means[0][1]  # 延迟最低的为 S0
#
#     s0_indices = np.where(stage1_labels == normal_cid)[0]
#     s1_indices = np.where(stage1_labels != normal_cid)[0]
#
#     print(f"  Stage1: S0={len(s0_indices)}, S1={len(s1_indices)}")
#
#     if len(s1_indices) == 0:
#         raise ValueError("No abnormal samples found!")
#
#     # Stage 2: S1 内部 KMeans(5)
#     s1_windows = train_windows[s1_indices]
#     X2 = extract_clustering_features(s1_windows)
#     scaler2 = StandardScaler()
#     X2_scaled = scaler2.fit_transform(X2)
#
#     kmeans2 = KMeans(n_clusters=5, random_state=random_state, n_init=10)
#     labels2 = kmeans2.fit_predict(X2_scaled)
#
#     # 合并逻辑（简化版：只保留最紧凑的1-2个）
#     cluster_metrics = []
#     for c in range(5):
#         mask = labels2 == c
#         if mask.sum() == 0:
#             avg_dist = float('inf')
#         elif mask.sum() == 1:
#             avg_dist = 0.0
#         else:
#             centroid = X2_scaled[mask].mean(axis=0)
#             avg_dist = np.linalg.norm(X2_scaled[mask] - centroid, axis=1).mean()
#         cluster_metrics.append((avg_dist, c))
#     cluster_metrics.sort()
#
#     min_avg_dist = cluster_metrics[0][0]
#     threshold = max(min_avg_dist * 2.0, 1.0)
#     light_clusters = {c for avg_dist, c in cluster_metrics if avg_dist <= threshold}
#
#     # 构建最终标签
#     final_labels = ["S0"] * len(train_windows)
#     for idx, s1_idx in enumerate(s1_indices):
#         final_labels[s1_idx] = "S1_light" if labels2[idx] in light_clusters else "S1_anomaly"
#
#     cluster_models = {
#         "strategy": "kmeans_two_stage",
#         "stage1": {"model": kmeans1, "scaler": scaler, "normal_cid": normal_cid},
#         "stage2": {"model": kmeans2, "scaler": scaler2, "light_clusters": light_clusters}
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 策略2: Isolation Forest + KMeans ====================
# def strategy_isolation_forest_two_stage(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略2: Isolation Forest + KMeans")
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Stage 1: Isolation Forest
#     iso = IsolationForest(contamination=0.4, random_state=random_state)  # 可调
#     stage1_pred = iso.fit_predict(X_scaled)  # 1=normal, -1=abnormal
#
#     s0_indices = np.where(stage1_pred == 1)[0]
#     s1_indices = np.where(stage1_pred == -1)[0]
#
#     print(f"  Stage1: S0={len(s0_indices)}, S1={len(s1_indices)}")
#
#     if len(s1_indices) == 0:
#         raise ValueError("No abnormal samples found!")
#
#     # Stage 2: 同策略1
#     s1_windows = train_windows[s1_indices]
#     X2 = extract_clustering_features(s1_windows)
#     scaler2 = StandardScaler()
#     X2_scaled = scaler2.fit_transform(X2)
#
#     kmeans2 = KMeans(n_clusters=5, random_state=random_state, n_init=10)
#     labels2 = kmeans2.fit_predict(X2_scaled)
#
#     cluster_metrics = []
#     for c in range(5):
#         mask = labels2 == c
#         if mask.sum() == 0:
#             avg_dist = float('inf')
#         elif mask.sum() == 1:
#             avg_dist = 0.0
#         else:
#             centroid = X2_scaled[mask].mean(axis=0)
#             avg_dist = np.linalg.norm(X2_scaled[mask] - centroid, axis=1).mean()
#         cluster_metrics.append((avg_dist, c))
#     cluster_metrics.sort()
#
#     min_avg_dist = cluster_metrics[0][0]
#     threshold = max(min_avg_dist * 2.0, 1.0)
#     light_clusters = {c for avg_dist, c in cluster_metrics if avg_dist <= threshold}
#
#     final_labels = ["S0"] * len(train_windows)
#     for idx, s1_idx in enumerate(s1_indices):
#         final_labels[s1_idx] = "S1_light" if labels2[idx] in light_clusters else "S1_anomaly"
#
#     cluster_models = {
#         "strategy": "isolation_forest_two_stage",
#         "stage1": {"model": iso, "scaler": scaler},
#         "stage2": {"model": kmeans2, "scaler": scaler2, "light_clusters": light_clusters}
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 策略3: GMM 一步三分类 ====================
# def strategy_gmm_three_class(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略3: GMM 一步三分类")
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # GMM
#     gmm = GaussianMixture(n_components=3, random_state=random_state, covariance_type='full')
#     labels = gmm.fit_predict(X_scaled)
#
#     # 根据异常程度排序
#     cluster_scores = []
#     for c in range(3):
#         mask = labels == c
#         if not np.any(mask):
#             score = float('inf')
#         else:
#             delays = []
#             losses = []
#             for w in train_windows[mask]:
#                 d1 = inverse_log_transform(w[0])
#                 d2 = inverse_log_transform(w[2])
#                 delays.extend([d1, d2])
#                 losses.extend([w[1], w[3]])
#             avg_delay = np.mean(delays)
#             avg_loss = np.mean(losses)
#             score = avg_delay * 0.7 + avg_loss * 100 * 0.3  # 加权异常分
#         cluster_scores.append((score, c))
#
#     cluster_scores.sort()  # 升序：S0 < S1_light < S1_anomaly
#     label_map = {}
#     for rank, (score, cid) in enumerate(cluster_scores):
#         name = ["S0", "S1_light", "S1_anomaly"][rank]
#         label_map[cid] = name
#
#     final_labels = [label_map[l] for l in labels]
#
#     cluster_models = {
#         "strategy": "gmm_three_class",
#         "model": gmm,
#         "scaler": scaler,
#         "label_map": label_map
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 公共保存函数 ====================
def _save_results(train_windows, final_labels, cluster_models, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 保存样本
    save_representative_samples(train_windows, final_labels, output_dir)

    # 保存分布
    unique, counts = np.unique(final_labels, return_counts=True)
    stats = dict(zip(unique, counts))
    with open(os.path.join(output_dir, "label_distribution.txt"), "w") as f:
        f.write("Final Label Distribution:\n")
        for label, count in stats.items():
            f.write(f"{label}: {count}\n")
    print(f"\n📊 最终标签分布: {stats}")
#
# # ==================== 预测函数（用于推理）====================
# def predict_labels_with_model(windows, cluster_models):
#     """根据模型类型预测"""
#     strategy = cluster_models["strategy"]
#     X = extract_clustering_features(windows)
#
#     if strategy == "gmm_three_class":
#         X_scaled = cluster_models["scaler"].transform(X)
#         labels = cluster_models["model"].predict(X_scaled)
#         return [cluster_models["label_map"][l] for l in labels]
#
#     else:  # two-stage
#         X_scaled = cluster_models["stage1"]["scaler"].transform(X)
#         if strategy == "kmeans_two_stage":
#             stage1_labels = cluster_models["stage1"]["model"].predict(X_scaled)
#             normal_cid = cluster_models["stage1"]["normal_cid"]
#             is_s0 = (stage1_labels == normal_cid)
#         else:  # isolation forest
#             stage1_pred = cluster_models["stage1"]["model"].predict(X_scaled)
#             is_s0 = (stage1_pred == 1)
#
#         final_labels = ["S0"] * len(windows)
#         s1_indices = np.where(~is_s0)[0]
#         if len(s1_indices) > 0:
#             s1_windows = [windows[i] for i in s1_indices]
#             X2 = extract_clustering_features(s1_windows)
#             X2_scaled = cluster_models["stage2"]["scaler"].transform(X2)
#             labels2 = cluster_models["stage2"]["model"].predict(X2_scaled)
#             light_clusters = cluster_models["stage2"]["light_clusters"]
#             for idx, s1_idx in enumerate(s1_indices):
#                 final_labels[s1_idx] = "S1_light" if labels2[idx] in light_clusters else "S1_anomaly"
#         return final_labels
#
# # ==================== 方案3：自动 GMM + BIC 选 K ====================
# def strategy_auto_gmm_bic(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略3: 自动 GMM + BIC 选 K")
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 自动选择最优 K
#     best_bic = np.inf
#     best_gmm = None
#     best_k = 2
#     max_k = 5
#
#     for k in range(2, max_k + 1):
#         try:
#             gmm = GaussianMixture(
#                 n_components=k,
#                 random_state=random_state,
#                 covariance_type='full',
#                 max_iter=200
#             )
#             gmm.fit(X_scaled)
#             bic = gmm.bic(X_scaled)
#             if bic < best_bic:
#                 best_bic = bic
#                 best_gmm = gmm
#                 best_k = k
#         except Exception as e:
#             print(f"⚠️ GMM failed for K={k}: {e}")
#             continue
#
#     print(f"🔍 自动选择 K={best_k} (BIC={best_bic:.2f})")
#     labels = best_gmm.predict(X_scaled)
#
#     # 根据异常程度排序
#     cluster_scores = []
#     for c in range(best_k):
#         mask = labels == c
#         if not np.any(mask):
#             score = float('inf')
#         else:
#             delays = []
#             losses = []
#             for w in train_windows[mask]:
#                 d1 = inverse_log_transform(w[0])
#                 d2 = inverse_log_transform(w[2])
#                 delays.extend([d1, d2])
#                 losses.extend([w[1], w[3]])
#             avg_delay = np.mean(delays)
#             avg_loss = np.mean(losses)
#             score = avg_delay * 0.7 + avg_loss * 100 * 0.3  # 加权异常分
#         cluster_scores.append((score, c))
#
#     cluster_scores.sort()  # 升序：S0 < S1_light < S1_anomaly
#
#     # 映射到语义标签
#     final_labels = []
#     for l in labels:
#         # 找到该类别的排名
#         rank = next(i for i, (score, cid) in enumerate(cluster_scores) if cid == l)
#         # 根据排名分配语义标签
#         if best_k == 2:
#             # 两分类：S0 和 S1_light（无严重异常）
#             final_labels.append("S0" if rank == 0 else "S1_light")
#         else:
#             # 多分类：S0, S1_light, S1_anomaly...
#             if rank == 0:
#                 final_labels.append("S0")
#             elif rank == best_k - 1:
#                 final_labels.append("S1_anomaly")
#             else:
#                 final_labels.append("S1_light")
#
#     cluster_models = {
#         "strategy": "auto_gmm_bic",
#         "model": best_gmm,
#         "scaler": scaler,
#         "best_k": best_k,
#         "cluster_scores": cluster_scores
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 方案4：层次聚类 + 距离阈值 ====================
# def strategy_hierarchical(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略4: 层次聚类 + 距离阈值")
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 使用层次聚类，自动确定簇数
#     clusterer = AgglomerativeClustering(
#         n_clusters=None, distance_threshold=100.0, linkage="ward", compute_full_tree=True
#     )
#     labels_numeric = clusterer.fit_predict(X_scaled)
#     n_clusters = len(np.unique(labels_numeric))
#     print(f"🔍 自动确定簇数: {n_clusters}")
#
#     # 根据异常程度排序
#     cluster_scores = []
#     for c in range(n_clusters):
#         mask = labels_numeric == c
#         if not np.any(mask):
#             score = float('inf')
#         else:
#             delays = []
#             losses = []
#             for w in train_windows[mask]:
#                 d1 = inverse_log_transform(w[0])
#                 d2 = inverse_log_transform(w[2])
#                 delays.extend([d1, d2])
#                 losses.extend([w[1], w[3]])
#             avg_delay = np.mean(delays)
#             avg_loss = np.mean(losses)
#             score = avg_delay * 0.7 + avg_loss * 100 * 0.3  # 加权异常分
#         cluster_scores.append((score, c))
#
#     cluster_scores.sort()  # 升序：S0 < S1_light < S1_anomaly
#
#     # 映射到语义标签
#     final_labels = []
#     for l in labels_numeric:
#         # 找到该类别的排名
#         rank = next(i for i, (score, cid) in enumerate(cluster_scores) if cid == l)
#         # 根据排名分配语义标签
#         if n_clusters == 1:
#             final_labels.append("S0")
#         elif n_clusters == 2:
#             # 两分类：S0 和 S1_light（无严重异常）
#             final_labels.append("S0" if rank == 0 else "S1_light")
#         else:
#             # 多分类：S0, S1_light, S1_anomaly...
#             if rank == 0:
#                 final_labels.append("S0")
#             elif rank == n_clusters - 1:
#                 final_labels.append("S1_anomaly")
#             else:
#                 final_labels.append("S1_light")
#
#     cluster_models = {
#         "strategy": "hierarchical",
#         "model": clusterer,
#         "scaler": scaler,
#         "n_clusters": n_clusters,
#         "cluster_scores": cluster_scores
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 方案5：DBSCAN ====================
# def strategy_dbscan(train_windows, random_state=42, output_dir="clustering_analysis"):
#     print("🔄 使用策略5: DBSCAN 基于密度聚类")
#     from sklearn.cluster import DBSCAN
#
#     X = extract_clustering_features(train_windows)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # 使用 DBSCAN 聚类
#     dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
#     labels_numeric = dbscan.fit_predict(X_scaled)
#
#     # 计算簇数（忽略噪声点 -1）
#     unique_labels = np.unique(labels_numeric)
#     n_clusters = len([l for l in unique_labels if l != -1])
#     n_noise = np.sum(labels_numeric == -1)
#     print(f"🔍 DBSCAN 结果: {n_clusters} 个簇, {n_noise} 个噪声点")
#
#     # 计算每个簇的异常程度
#     cluster_scores = []
#     noise_indices = np.where(labels_numeric == -1)[0]
#
#     # 处理正常簇
#     for c in range(n_clusters):
#         mask = labels_numeric == c
#         if not np.any(mask):
#             continue
#
#         delays = []
#         losses = []
#         for w in train_windows[mask]:
#             d1 = inverse_log_transform(w[0])
#             d2 = inverse_log_transform(w[2])
#             delays.extend([d1, d2])
#             losses.extend([w[1], w[3]])
#         avg_delay = np.mean(delays)
#         avg_loss = np.mean(losses)
#         score = avg_delay * 0.7 + avg_loss * 100 * 0.3  # 加权异常分
#         cluster_scores.append((score, c))
#
#     cluster_scores.sort()  # 升序：S0 < S1_light < S1_anomaly
#
#     # 映射到语义标签
#     final_labels = []
#     for l in labels_numeric:
#         if l == -1:
#             # 噪声点作为严重异常
#             final_labels.append("S1_anomaly")
#         else:
#             # 找到该类别的排名
#             rank = next(i for i, (score, cid) in enumerate(cluster_scores) if cid == l)
#             # 根据排名分配语义标签
#             if n_clusters == 1:
#                 final_labels.append("S0")
#             else:
#                 if rank == 0:
#                     final_labels.append("S0")
#                 elif rank == n_clusters - 1:
#                     final_labels.append("S1_anomaly")
#                 else:
#                     final_labels.append("S1_light")
#
#     cluster_models = {
#         "strategy": "dbscan",
#         "model": dbscan,
#         "scaler": scaler,
#         "n_clusters": n_clusters,
#         "n_noise": n_noise
#     }
#
#     _save_results(train_windows, final_labels, cluster_models, output_dir)
#     return final_labels, cluster_models
#
# # ==================== 新的主入口 ====================
# def two_stage_clustering(train_windows, strategy="kmeans_two_stage", random_state=42, output_dir="clustering_analysis"):
#     """
#     统一入口，支持多种策略：
#       - "kmeans_two_stage": 两阶段 KMeans
#       - "isolation_forest_two_stage": Isolation Forest + KMeans
#       - "gmm_three_class": 固定三类 GMM
#       - "auto_gmm_bic": 自动 GMM + BIC 选 K
#       - "hierarchical": 层次聚类 + 距离阈值
#       - "dbscan": DBSCAN 基于密度聚类
#     """
#     if strategy == "kmeans_two_stage":
#         return strategy_kmeans_two_stage(train_windows, random_state, output_dir)
#     elif strategy == "isolation_forest_two_stage":
#         return strategy_isolation_forest_two_stage(train_windows, random_state, output_dir)
#     elif strategy == "gmm_three_class":
#         return strategy_gmm_three_class(train_windows, random_state, output_dir)
#     elif strategy == "auto_gmm_bic":
#         return strategy_auto_gmm_bic(train_windows, random_state, output_dir)
#     elif strategy == "hierarchical":
#         return strategy_hierarchical(train_windows, random_state, output_dir)
#     elif strategy == "dbscan":
#         return strategy_dbscan(train_windows, random_state, output_dir)
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")
#
# # 保留原有函数名兼容性（可选）
# # def two_stage_clustering(...): ...  # 已定义
#
def save_representative_samples(windows, labels, output_dir, n_samples=10):
    """
    保存代表性样本，严格匹配 visualize_clusters.py 所需的列名：
        sample_id, time_step,
        UL_delay_ms, UL_loss%,
        DL_delay_ms, DL_loss%
    """
    import random
    import pandas as pd
    import os

    random.seed(42)
    label_set = sorted(set(labels))

    for label in label_set:
        indices = [i for i, l in enumerate(labels) if l == label]
        if not indices:
            continue

        selected_indices = random.sample(indices, min(n_samples, len(indices)))
        all_rows = []

        for sample_id, idx in enumerate(selected_indices):
            win = windows[idx]  # shape: (T, 4) → [d1_log, p1, d2_log, p2]
            T = win.shape[0]
            for t in range(T):
                d1_log, p1, d2_log, p2 = win[t]
                UL_delay_ms = inverse_log_transform(d1_log)
                DL_delay_ms = inverse_log_transform(d2_log)
                UL_loss_pct = p1 * 100.0  # 转换为百分比（如果原数据是比率）
                DL_loss_pct = p2 * 100.0

                all_rows.append(
                    {
                        "sample_id": sample_id,
                        "time_step": t,
                        "UL_delay_ms": UL_delay_ms,
                        "UL_loss%": UL_loss_pct,  # ← 注意这里是 %
                        "DL_delay_ms": DL_delay_ms,
                        "DL_loss%": DL_loss_pct,  # ← 注意这里是 %
                    }
                )

        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(output_dir, f"samples_{label}.csv")
        df.to_csv(csv_path, index=False)
        print(f"🔍 人工审核样本已保存至:\n  - {csv_path}")
#
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# import numpy as np
#
#
# def auto_hierarchical_clustering(features, max_distance=100.0):
#     """
#     自动层次聚类：基于簇间距离阈值确定类别数
#     :param features: (n_samples, n_features) 特征矩阵（如 [mean_delay, loss_pct]）
#     :param max_distance: 合并停止的距离阈值（需根据数据尺度调整）
#     :return: labels (list of str like "S0", "S1", ...)
#     """
#     # 计算 linkage matrix（用于树状图）
#     linked = linkage(features, method="ward")  # ward 最小化方差
#
#     # 自动确定簇数：找到距离 > max_distance 的切割点
#     distances = linked[:, 2]
#     # 找最大的 gap
#     gaps = np.diff(distances)
#     if len(gaps) == 0:
#         n_clusters = 1
#     else:
#         best_gap_idx = np.argmax(gaps)
#         # 切割点在 best_gap_idx + 1
#         n_clusters = len(distances) - best_gap_idx
#
#     # 或者更简单：直接用距离阈值
#     clusterer = AgglomerativeClustering(
#         n_clusters=None, distance_threshold=max_distance, linkage="ward"
#     )
#     labels_numeric = clusterer.fit_predict(features)
#
#     # 转换为语义标签 S0, S1, S2...
#     unique_labels = sorted(np.unique(labels_numeric))
#     label_map = {lbl: f"S{i}" for i, lbl in enumerate(unique_labels)}
#     labels_str = [label_map[l] for l in labels_numeric]
#
#     return labels_str, clusterer.n_clusters_
#
# # 移除重复的函数定义，保留完整实现

"""
聚类策略定义模块
输入: windows (N, T, 4) —— [UL_delay_log, UL_loss, DL_delay_log, DL_loss]
输出: labels (list of str), e.g., ["S0", "S1_light", ...]
"""

import numpy as np

# -----------------------------
# 工具函数
# -----------------------------

def inverse_log_transform(x):
    """还原对数变换：log(1+x) -> x"""
    return np.expm1(x)

def smart_label_mapping(features_2d, labels_numeric, strategy_name=""):
    """
    根据特征自动分配语义标签：
    - features_2d: shape (N, 2), 列为 [mean_delay, mean_loss_pct]
    - labels_numeric: 聚类原始标签（整数）
    返回: ["S0", "S1_light", "S1_anomaly", ...]
    """
    unique_labels = np.unique(labels_numeric)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return ["S0"] * len(labels_numeric)

    # 计算每簇的平均延迟和丢包
    stats = []
    for lbl in unique_labels:
        mask = (labels_numeric == lbl)
        mean_delay = np.mean(features_2d[mask, 0])
        mean_loss = np.mean(features_2d[mask, 1])
        stats.append((lbl, mean_delay, mean_loss))

    # 按平均延迟升序排序
    stats.sort(key=lambda x: x[1])

    label_map = {}

    if n_clusters == 2:
        low_lbl, low_delay, low_loss = stats[0]
        high_lbl, high_delay, high_loss = stats[1]

        # 如果高簇丢包 > 95%，视为 anomaly；否则是 light
        if high_loss > 95.0:
            label_map = {low_lbl: "S0", high_lbl: "S1_anomaly"}
        else:
            label_map = {low_lbl: "S0", high_lbl: "S1_light"}

    else:  # n_clusters >= 3
        # 最低延迟 → S0
        label_map[stats[0][0]] = "S0"

        # 找出最可能是 anomaly 的簇：优先看丢包，其次看延迟
        loss_values = [s[2] for s in stats]
        delay_values = [s[1] for s in stats]
        max_loss_idx = int(np.argmax(loss_values))
        max_delay_idx = int(np.argmax(delay_values))

        # 如果最高丢包 > 50%，就用它；否则用最高延迟
        if loss_values[max_loss_idx] > 50.0:
            anomaly_lbl = stats[max_loss_idx][0]
        else:
            anomaly_lbl = stats[max_delay_idx][0]

        label_map[anomaly_lbl] = "S1_anomaly"

        # 剩余所有簇 → S1_light
        for lbl, _, _ in stats:
            if lbl not in label_map:
                label_map[lbl] = "S1_light"

    final_labels = [label_map[l] for l in labels_numeric]
    print(f"🧠 {strategy_name} 自动标签映射: {label_map}")
    return final_labels


# -----------------------------
# 聚类策略实现
# -----------------------------

def strategy_kmeans_two_stage(windows):
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    total_delay = (UL_delay + DL_delay) / 2.0
    total_loss = (UL_loss + DL_loss) / 2.0

    features = np.column_stack([
        np.mean(total_delay, axis=1),
        np.mean(total_loss, axis=1) * 100,
    ])

    # 第一阶段：K=2
    kmeans = KMeans(n_clusters=2, random_state=42)
    stage1_labels = kmeans.fit_predict(features)

    # 找出“正常”簇（平均延迟更低）
    mean_delays = [features[stage1_labels == i, 0].mean() for i in [0, 1]]
    normal_cluster = np.argmin(mean_delays)
    abnormal_mask = (stage1_labels != normal_cluster)

    final_labels = ["S0"] * len(windows)

    if np.any(abnormal_mask):
        # 第二阶段：在异常簇中用 Isolation Forest 找极端点
        abnormal_features = features[abnormal_mask]
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso_labels = iso.fit_predict(abnormal_features)  # 1=正常, -1=异常

        idx = 0
        for i, is_abnormal in enumerate(abnormal_mask):
            if is_abnormal:
                if iso_labels[idx] == -1:
                    final_labels[i] = "S1_anomaly"
                else:
                    final_labels[i] = "S1_light"
                idx += 1

    return final_labels, None


def strategy_isolation_forest_two_stage(windows):
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    total_delay = (UL_delay + DL_delay) / 2.0
    total_loss = (UL_loss + DL_loss) / 2.0

    features = np.column_stack([
        np.mean(total_delay, axis=1),
        np.mean(total_loss, axis=1) * 100,
    ])

    # 第一阶段：Isolation Forest 找大异常
    iso1 = IsolationForest(contamination=0.05, random_state=42)
    outlier_mask = (iso1.fit_predict(features) == -1)

    final_labels = ["S1_anomaly" if m else "S0" for m in outlier_mask]

    # 第二阶段：对非异常部分做 K=2 细分
    normal_mask = ~outlier_mask
    if np.sum(normal_mask) > 10:
        normal_features = features[normal_mask]
        kmeans = KMeans(n_clusters=2, random_state=42)
        sub_labels = kmeans.fit_predict(normal_features)

        # 按延迟分配 S0 / S1_light
        mean_delays = [normal_features[sub_labels == i, 0].mean() for i in [0, 1]]
        better_cluster = np.argmin(mean_delays)  # 更低延迟的是 S0

        idx = 0
        for i, is_normal in enumerate(normal_mask):
            if is_normal:
                if sub_labels[idx] == better_cluster:
                    final_labels[i] = "S0"
                else:
                    final_labels[i] = "S1_light"
                idx += 1

    return final_labels, None


def strategy_gmm_three_class(windows):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    features_4d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_4d)

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    labels_numeric = gmm.fit_predict(features_scaled)

    final_labels = smart_label_mapping(features_2d, labels_numeric, "gmm_three_class")
    return final_labels, gmm


def strategy_gmm_four_class(windows):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    features_4d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_4d)

    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    labels_numeric = gmm.fit_predict(features_scaled)

    final_labels = smart_label_mapping(features_2d, labels_numeric, "gmm_four_class")
    return final_labels, gmm


def strategy_auto_gmm_bic(windows):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    # 用于映射的 2D 特征
    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    # 用于聚类的 8D 特征
    features_8d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(UL_loss, axis=1) * 100,
        np.mean(UL_loss > 0.05, axis=1) * 100,
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
        np.mean(DL_loss, axis=1) * 100,
        np.mean(DL_loss > 0.05, axis=1) * 100,
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_8d)

    best_bic = np.inf
    best_gmm = None
    best_k = 2

    for k in range(2, 6):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200)
            gmm.fit(features_scaled)
            bic = gmm.bic(features_scaled)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        except Exception:
            continue

    print(f"🔍 auto_gmm_bic 选择 K={best_k} (BIC={best_bic:.0f})")
    labels_numeric = best_gmm.predict(features_scaled)
    final_labels = smart_label_mapping(features_2d, labels_numeric, "auto_gmm_bic")
    return final_labels, best_gmm


def strategy_hierarchical(windows):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    features_4d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_4d)

    # 自动选 K：通过距离阈值
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.5,
        linkage='ward'
    )
    labels_numeric = clustering.fit_predict(features_scaled)

    # 限制最多 3 类（避免碎片化）
    if len(np.unique(labels_numeric)) > 3:
        # 回退到 K=3
        clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
        labels_numeric = clustering.fit_predict(features_scaled)

    final_labels = smart_label_mapping(features_2d, labels_numeric, "hierarchical")
    return final_labels, clustering


def strategy_dbscan(windows):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    features_4d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_4d)

    dbscan = DBSCAN(eps=0.8, min_samples=20)
    labels_numeric = dbscan.fit_predict(features_scaled)

    final_labels = smart_label_mapping(features_2d, labels_numeric, "dbscan")
    return final_labels, dbscan


def strategy_hdbscan_auto(windows):
    try:
        import hdbscan
    except ImportError:
        raise ImportError("请安装: pip install hdbscan")

    from sklearn.preprocessing import StandardScaler

    UL_delay = inverse_log_transform(windows[:, :, 0])
    UL_loss = windows[:, :, 1]
    DL_delay = inverse_log_transform(windows[:, :, 2])
    DL_loss = windows[:, :, 3]

    features_2d = np.column_stack([
        (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
        (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
    ])

    features_6d = np.column_stack([
        np.mean(UL_delay, axis=1),
        np.std(UL_delay, axis=1),
        np.mean(UL_loss, axis=1) * 100,
        np.mean(DL_delay, axis=1),
        np.std(DL_delay, axis=1),
        np.mean(DL_loss, axis=1) * 100,
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_6d)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(50, len(windows) // 100),  # 至少50或1%
        min_samples=10,
        metric='euclidean'
    )
    labels_numeric = clusterer.fit_predict(features_scaled)

    final_labels = smart_label_mapping(features_2d, labels_numeric, "hdbscan_auto")
    return final_labels, clusterer


# -----------------------------
# 策略注册表
# -----------------------------

STRATEGY_MAP = {
    "kmeans_two_stage": strategy_kmeans_two_stage,
    "isolation_forest_two_stage": strategy_isolation_forest_two_stage,
    "gmm_three_class": strategy_gmm_three_class,
    "gmm_four_class": strategy_gmm_four_class,
    "auto_gmm_bic": strategy_auto_gmm_bic,
    "hierarchical": strategy_hierarchical,
    "dbscan": strategy_dbscan,
    "hdbscan_auto": strategy_hdbscan_auto,
}

# -----------------------------
# 统一接口函数（供 build_train_dataset.py 调用）
# -----------------------------


def two_stage_clustering(
    windows, strategy="gmm_three_class", random_state=42, output_dir="clustering_analysis"
):
    """
    统一聚类入口
    :param windows: (N, T, 4) array
    :param strategy: str, one of STRATEGY_MAP keys
    :param output_dir: 输出目录，用于保存样本和分布
    :return: labels (list of str), model (fitted clusterer or None)
    """
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(STRATEGY_MAP.keys())}"
        )

    np.random.seed(random_state)
    label_func = STRATEGY_MAP[strategy]
    labels, model = label_func(windows)
    
    # 保存样本和分布
    _save_results(windows, labels, model, output_dir)
    
    return labels, model


def predict_labels_with_model(windows, model, strategy="gmm_three_class"):
    """
    使用已训练模型预测新窗口标签
    注意：目前仅部分策略支持（如 GMM），其他返回 None
    """
    if strategy == "gmm_three_class" and hasattr(model, "predict"):
        from sklearn.preprocessing import StandardScaler

        UL_delay = inverse_log_transform(windows[:, :, 0])
        DL_delay = inverse_log_transform(windows[:, :, 2])
        features_4d = np.column_stack(
            [
                np.mean(UL_delay, axis=1),
                np.std(UL_delay, axis=1),
                np.mean(DL_delay, axis=1),
                np.std(DL_delay, axis=1),
            ]
        )
        scaler = StandardScaler()
        # 注意：这里需要保存训练时的 scaler！当前简化处理
        features_scaled = scaler.fit_transform(features_4d)
        labels_numeric = model.predict(features_scaled)

        # 重新提取 2D 特征用于映射（临时）
        UL_loss = windows[:, :, 1]
        DL_loss = windows[:, :, 3]
        features_2d = np.column_stack(
            [
                (np.mean(UL_delay, axis=1) + np.mean(DL_delay, axis=1)) / 2,
                (np.mean(UL_loss, axis=1) + np.mean(DL_loss, axis=1)) * 100,
            ]
        )
        return smart_label_mapping(features_2d, labels_numeric, "gmm_three_class")

    else:
        # 其他策略暂不支持在线预测（需重跑整个流程）
        raise NotImplementedError(
            f"Prediction not implemented for strategy: {strategy}"
        )