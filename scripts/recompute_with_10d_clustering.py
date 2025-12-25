#!/usr/bin/env python3
"""
使用10D特征聚类结果重新生成条件向量

该脚本使用10D特征的聚类结果作为新的state_id，重新生成用于模型训练和推理的条件向量。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import stage9_recompute_condition_vectors, WindowMetaRenamed


def load_clustering_labels(k):
    """加载10D特征聚类结果标签
    
    Args:
        k (int): 聚类的K值
    
    Returns:
        np.ndarray: 聚类标签数组
    """
    cluster_labels_path = f"output/clustering_10d/cluster_labels_k{k}.npy"
    if not os.path.exists(cluster_labels_path):
        raise FileNotFoundError(f"聚类结果文件不存在: {cluster_labels_path}")
    
    cluster_labels = np.load(cluster_labels_path)
    print(f"加载聚类标签: {cluster_labels_path}")
    print(f"聚类标签形状: {cluster_labels.shape}")
    
    return cluster_labels


def load_renamed_datasets():
    """加载重命名后的数据集
    
    Returns:
        dict: 包含train, val, test的数据集字典
    """
    import pandas as pd
    datasets_dir = Path("output/datasets")
    
    datasets = {}
    for dataset_name in ["train", "val", "test"]:
        file_path = datasets_dir / f"{dataset_name}.jsonl"
        
        dataset = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                # 转换为pandas DataFrame
                window_df = pd.DataFrame(data["window"])
                
                meta: WindowMetaRenamed = {
                    "window": window_df,
                    "is_first": False,  # 所有保存的样本都是keep=True，即not is_first
                    "start_time": data["start_time"],
                    "trace_id": data["trace_id"]
                }
                
                dataset.append(meta)
        
        datasets[dataset_name] = dataset
    
    return datasets


def load_assets():
    """加载资产文件
    
    Returns:
        dict: 包含qt_up, qt_down等资产的字典
    """
    import joblib
    assets_dir = Path("output/assets")
    
    assets = {
        "qt_up": joblib.load(assets_dir / "qt_up.pkl"),
        "qt_down": joblib.load(assets_dir / "qt_down.pkl")
    }
    
    return assets


def create_network_state_map(cluster_labels):
    """创建网络状态映射
    
    Args:
        cluster_labels (np.ndarray): 聚类标签数组
    
    Returns:
        dict: network_state_map，键为trace_id，值为聚类标签
    """
    # 注意：这里我们假设每个trace_id对应一个聚类标签
    # 实际情况可能需要根据原始数据进行调整
    # 这里我们简单地为每个trace分配一个唯一的ID，并映射到对应的聚类标签
    network_state_map = {}
    for i, label in enumerate(cluster_labels):
        # 假设trace_id格式为"trace_0", "trace_1", ...
        trace_id = f"trace_{i}"
        network_state_map[trace_id] = int(label)
    
    return network_state_map


def main(k=3):
    """主函数
    
    Args:
        k (int): 聚类的K值
    """
    print(f"使用10D特征聚类结果重新生成条件向量，K={k}...")
    
    # 1. 加载聚类标签
    cluster_labels = load_clustering_labels(k)
    
    # 2. 创建网络状态映射
    network_state_map = create_network_state_map(cluster_labels)
    print(f"创建了 {len(network_state_map)} 个trace的状态映射")
    
    # 3. 加载重命名后的数据集
    print("加载重命名后的数据集...")
    renamed_datasets = load_renamed_datasets()
    print(f"数据集大小: train={len(renamed_datasets['train'])}, val={len(renamed_datasets['val'])}, test={len(renamed_datasets['test'])}")
    
    # 4. 加载资产文件
    print("加载资产文件...")
    assets = load_assets()
    
    # 5. 重新计算条件向量
    print("重新计算条件向量...")
    final_datasets, extra_assets = stage9_recompute_condition_vectors(
        renamed_datasets, assets, network_state_map
    )
    assets.update(extra_assets)
    
    # 6. 保存结果
    print("保存结果...")
    from src.preprocessing import stage10_save_artifacts
    out_dir = Path(f"output/clustered_10d_k{k}")
    out_dir.mkdir(exist_ok=True, parents=True)
    stage10_save_artifacts(final_datasets, assets, out_dir, dtype=np.float32)
    
    # 7. 更新schema
    schema = {
        "pipeline_version": "v1.4",
        "columns": ["timestamp", "del_up", "del_dn", "loss_up", "loss_dn"],
        "window_size": 100,
        "freq_hz": 10,
        "normalized": True,
        "condition_vector_dim": 23,
        "condition_vector_structure": {
            "global_features": list(range(13)),
            "local_features": list(range(13, 23)),
        },
        "network_state_id_source": f"10D features clustering (KMeans, K={k})",
        "network_state_id_encoding": "integer category stored as float (e.g., 2.0)",
        "level2_normalization": "Z-score normalization applied to 22 float dimensions (indices 0–10 and 12–22), excluding the integer-encoded network_state_id at index 11.",
        "split_strategy": "Entire traces are assigned to a single split to prevent data leakage.",
        "normalization": {
            "del_up/del_dn": "QuantileTransformer(output_distribution='normal'), fitted on train set",
            "loss_up/loss_dn": "Clipped to [0.0, 1.0], no transformation applied",
        },
    }
    
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # 8. 保存聚类信息
    clustering_info = {
        "cluster_features": "10D",
        "optimal_k": k,
        "state_id_stats": assets.get("state_id_stats", {})
    }
    
    with open(meta_dir / "clustering_info.json", "w", encoding="utf-8") as f:
        json.dump(clustering_info, f, ensure_ascii=False, indent=2)
    
    print("处理完成！")
    print(f"结果保存到: {out_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用10D特征聚类结果重新生成条件向量")
    parser.add_argument('--k', type=int, default=3, help="聚类的K值，默认使用轮廓系数最佳的K=3")
    
    args = parser.parse_args()
    
    main(k=args.k)
