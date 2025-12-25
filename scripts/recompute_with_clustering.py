#!/usr/bin/env python3

"""
使用聚类结果重新生成条件向量脚本
"""

import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path

from src.preprocessing import stage9_recompute_condition_vectors, WindowMetaRenamed


def load_clustering_results():
    """加载聚类结果
    
    Returns:
        dict: network_state_map，键为trace_id，值为聚类标签
    """
    clustering_dir = Path("output/clustering")
    network_state_map_path = clustering_dir / "network_state_map.json"
    
    with open(network_state_map_path, "r") as f:
        network_state_map = json.load(f)
    
    return network_state_map


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


def main():
    print("加载聚类结果...")
    network_state_map = load_clustering_results()
    print(f"加载了 {len(network_state_map)} 个trace的聚类结果")
    
    print("加载重命名后的数据集...")
    renamed_datasets = load_renamed_datasets()
    print(f"数据集大小: train={len(renamed_datasets['train'])}, val={len(renamed_datasets['val'])}, test={len(renamed_datasets['test'])}")
    
    print("加载资产文件...")
    assets = load_assets()
    
    print("重新计算条件向量...")
    final_datasets, extra_assets = stage9_recompute_condition_vectors(
        renamed_datasets, assets, network_state_map
    )
    assets.update(extra_assets)
    
    print("保存结果...")
    from src.preprocessing import stage10_save_artifacts
    out_dir = Path("output/clustered")
    out_dir.mkdir(exist_ok=True)
    stage10_save_artifacts(final_datasets, assets, out_dir, dtype=np.float32)
    
    # 更新schema
    schema = {
        "pipeline_version": "v1.3",
        "columns": ["timestamp", "del_up", "del_dn", "loss_up", "loss_dn"],
        "window_size": 100,
        "freq_hz": 10,
        "normalized": True,
        "condition_vector_dim": 23,
        "condition_vector_structure": {
            "global_features": list(range(13)),
            "local_features": list(range(13, 23)),
        },
        "network_state_id_source": "6D features clustering (KMeans, K=6)",
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
    
    print("处理完成！")
    print(f"结果保存到: {out_dir}")


if __name__ == "__main__":
    main()