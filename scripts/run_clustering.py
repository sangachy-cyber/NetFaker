#!/usr/bin/env python3
"""
聚类模块入口脚本。
基于训练集窗口数据，通过聚类定义网络状态（state_id），并为测试集分配一致语义的状态标签。

默认策略：使用 GMM（高斯混合模型）聚 3 类，输出概率增强的 state_id。
"""

import argparse
from netfaker.simcore.clustering import ClusterRunner


def main():
    parser = argparse.ArgumentParser(description="HoloWAN 聚类与状态生成模块")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gmm",
        choices=["gmm"],
        help="聚类算法（默认：gmm）"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        help="聚类数量（默认：3）"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/datasets/train.parquet",
        help="训练集路径（默认：data/datasets/train.parquet）"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/datasets/test.parquet",
        help="测试集路径（默认：data/datasets/test.parquet）"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="纯净状态的置信度阈值（默认：0.85）"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="生成可视化（默认：True）"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="禁用可视化"
    )
    parser.add_argument(
        "--assign-test-states",
        action="store_true",
        default=True,
        help="为测试集分配状态（默认：True）"
    )
    
    args = parser.parse_args()
    
    print(f"=== HoloWAN 聚类与状态生成模块 ===")
    print(f"算法: {args.algorithm}")
    print(f"聚类数量: {args.n_components}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"训练集: {args.train_path}")
    print(f"测试集: {args.test_path}")
    print(f"可视化: {args.visualize}")
    print(f"为测试集分配状态: {args.assign_test_states}")
    print()
    
    # 初始化并运行聚类执行器
    runner = ClusterRunner(
        algorithm=args.algorithm,
        n_components=args.n_components,
        confidence_threshold=args.confidence_threshold,
        visualize=args.visualize
    )
    
    try:
        stats = runner.run(args.train_path, args.test_path)
        print("=== 聚类完成 ===")
        print(f"训练集状态分布: {stats['train']['state_distribution']}")
        print(f"训练集平均置信度: {stats['train']['mean_proba']:.4f}")
        print(f"测试集状态分布: {stats['test']['state_distribution']}")
        print(f"测试集平均置信度: {stats['test']['mean_proba']:.4f}")
        
        if "gmm" in stats:
            print(f"GMM BIC: {stats['gmm']['bic']:.2f}")
            print(f"GMM AIC: {stats['gmm']['aic']:.2f}")
        
        print()
        print(f"输出文件:")
        print(f"  - 训练集带状态: data/clusters/train_with_state.parquet")
        print(f"  - 测试集带状态: data/clusters/test_with_state.parquet")
        print(f"  - 模型文件: data/clusters/gmm_model.joblib")
        print(f"  - 特征缩放器: data/clusters/feature_scaler.joblib")
        print(f"  - 状态元数据: data/clusters/state_metadata.json")
        print(f"  - 实验日志: logs/clustering_{stats['timestamp']}.json")
        
        if "visualization" in stats:
            print(f"  - 可视化:")
            if "umap" in stats["visualization"]:
                print(f"    - UMAP 图: {stats['visualization']['umap']}")
            if "tsne" in stats["visualization"]:
                print(f"    - t-SNE 图: {stats['visualization']['tsne']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()