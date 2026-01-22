"""可视化模块，用于生成网络状态聚类的降维可视化结果。

该模块提供了以下功能：
- UMAP 降维可视化
- t-SNE 降维可视化
- 训练集和测试集组合可视化
- 静态图像生成
- 交互式可视化（可选）

模块特性：
- 条件导入 UMAP，处理依赖问题
- 固定文件名，便于集成到报告中
- 支持混合状态标记
- 自定义颜色映射
"""

import json
import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE  # t-SNE 降维

# 尝试导入 UMAP，处理依赖问题
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("警告: UMAP依赖未安装，UMAP可视化功能将不可用。您可以使用 'pip install umap-learn' 安装。")


class Visualizer:
    """可视化器，用于生成网络状态聚类的降维可视化。

    该类提供了UMAP和t-SNE两种降维方法的可视化功能，
    支持静态图像生成和交互式可视化（可选）。

    典型使用场景：
    - 网络状态聚类结果可视化
    - 训练集和测试集组合分析
    - 聚类效果评估

    示例：
    ```python
    # 初始化可视化器
    visualizer = Visualizer("data/clusters/state_metadata.json")

    # 生成可视化
    viz_paths = visualizer.generate_visualization(
        features, state_info,
        output_dir="output/reports/cluster",
        title="Network State Clustering"
    )

    # 查看生成的文件
    print("UMAP可视化路径:", viz_paths.get("umap"))
    print("t-SNE可视化路径:", viz_paths.get("tsne"))
    ```
    """

    def __init__(self, state_metadata_path: str = "data/clusters/state_metadata.json"):
        """初始化可视化器。

        Args:
            state_metadata_path: 状态元数据文件路径，包含状态颜色信息

        Attributes:
            state_metadata_path: 状态元数据文件路径
            state_metadata: 加载的状态元数据字典
            color_map: 状态ID到颜色的映射字典
        """
        self.state_metadata_path = state_metadata_path
        self.state_metadata = self._load_state_metadata()
        self.color_map = self._create_color_map()

    def _load_state_metadata(self) -> Dict[str, Any]:
        """加载状态元数据文件。

        从指定路径加载状态元数据JSON文件，包含状态ID、名称和颜色信息。

        Returns:
            Dict[str, Any]: 状态元数据字典，包含状态列表和其他信息

        示例：
        ```python
        # 加载状态元数据
        metadata = visualizer._load_state_metadata()
        print("加载的状态数量:", len(metadata.get("states", [])))
        ```
        """
        if os.path.exists(self.state_metadata_path):
            with open(self.state_metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _create_color_map(self) -> Dict[int, str]:
        """创建状态颜色映射字典。

        从状态元数据中提取每个状态的颜色信息，创建状态ID到颜色代码的映射。

        Returns:
            Dict[int, str]: 颜色映射字典，键为state_id，值为颜色代码

        示例：
        ```python
        # 创建颜色映射
        color_map = visualizer._create_color_map()
        print("颜色映射:", color_map)
        ```
        """
        color_map = {}
        if "states" in self.state_metadata:
            for state in self.state_metadata["states"]:
                color_map[state["state_id"]] = state["color"]
        return color_map

    def generate_visualization(self, X: np.ndarray, state_info: Dict[str, np.ndarray],
                               output_dir: str = "output/reports/cluster",
                               title: str = "HoloWAN Network State Clustering") -> Dict[str, str]:
        """生成UMAP和t-SNE可视化。

        该方法执行以下步骤：
        1. 确保输出目录存在
        2. 如果UMAP可用，执行UMAP降维并生成可视化
        3. 执行t-SNE降维并生成可视化
        4. 返回生成的文件路径

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)
            state_info: 状态信息字典，包含以下字段：
                - state_id: 状态ID数组
                - is_pure: 是否为纯净状态的布尔数组
                - state_proba: 状态概率数组
                - state_name: 状态名称数组
            output_dir: 输出目录路径
            title: 可视化标题

        Returns:
            Dict[str, str]: 输出文件路径字典，包含：
                - umap: UMAP可视化文件路径（如果生成）
                - tsne: t-SNE可视化文件路径

        示例：
        ```python
        # 生成可视化
        viz_paths = visualizer.generate_visualization(
            features, state_info,
            output_dir="output/reports/cluster",
            title="Network State Clustering"
        )

        # 查看生成的文件路径
        print("UMAP可视化:", viz_paths.get("umap"))
        print("t-SNE可视化:", viz_paths.get("tsne"))
        ```
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        output_paths = {}

        # 执行 UMAP 降维并生成可视化
        if UMAP_AVAILABLE:
            umap_embedding = self._run_umap(X)
            umap_path = self._create_static_plot(
                umap_embedding, state_info, output_dir, f"{title} (UMAP)", "umap"
            )
            output_paths["umap"] = umap_path

        # 执行 t-SNE 降维并生成可视化
        tsne_embedding = self._run_tsne(X)
        tsne_path = self._create_static_plot(
            tsne_embedding, state_info, output_dir, f"{title} (t-SNE)", "tsne"
        )
        output_paths["tsne"] = tsne_path

        return output_paths

    def _run_umap(self, X: np.ndarray) -> np.ndarray:
        """执行UMAP降维。

        使用UMAP算法将高维特征矩阵降维到2维空间，用于可视化。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 降维后的嵌入，形状为 (n_samples, 2)

        示例：
        ```python
        # 执行UMAP降维
        embedding = visualizer._run_umap(features)
        print("UMAP嵌入形状:", embedding.shape)  # 输出 (n_samples, 2)
        ```
        """
        if not UMAP_AVAILABLE:
            # 如果 UMAP 不可用，使用随机嵌入作为 fallback
            return np.random.randn(X.shape[0], 2)

        try:
            # 调整 UMAP 参数以获得更好的聚类可视化效果
            umap = UMAP(
                n_components=2,
                n_neighbors=15,  # 增加邻居数量，平衡局部和全局结构
                min_dist=0.1,    # 增加最小距离，使聚类更清晰
                metric="euclidean",
                random_state=42
            )
            return umap.fit_transform(X)
        except Exception as e:
            # 如果UMAP运行失败，打印异常信息并使用随机嵌入作为 fallback
            print(f"UMAP可视化失败: {str(e)}")
            return np.random.randn(X.shape[0], 2)

    def _run_tsne(self, X: np.ndarray) -> np.ndarray:
        """执行t-SNE降维。

        使用t-SNE算法将高维特征矩阵降维到2维空间，用于可视化。

        Args:
            X: 特征矩阵，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 降维后的嵌入，形状为 (n_samples, 2)

        示例：
        ```python
        # 执行t-SNE降维
        embedding = visualizer._run_tsne(features)
        print("t-SNE嵌入形状:", embedding.shape)  # 输出 (n_samples, 2)
        ```
        """
        try:
            # 使用简化的 TSNE 参数，确保与当前 scikit-learn 版本兼容
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                random_state=42
            )
            return tsne.fit_transform(X).astype(np.float64)
        except Exception as e:
            # 如果t-SNE运行失败，打印异常信息并使用随机嵌入作为 fallback
            print(f"t-SNE可视化失败: {str(e)}")
            return np.random.randn(X.shape[0], 2)

    def _create_static_plot(self, embedding: np.ndarray, state_info: Dict[str, np.ndarray],
                           output_dir: str, title: str, method: str) -> str:
        """创建静态可视化图。

        生成降维结果的静态可视化图，包括：
        1. 不同状态使用不同颜色
        2. 混合状态添加边框标记
        3. 样本数量较少的状态增大点的大小
        4. 使用固定文件名保存

        Args:
            embedding: 降维嵌入，形状为 (n_samples, 2)
            state_info: 状态信息字典
            output_dir: 输出目录
            title: 可视化标题
            method: 降维方法名称（"umap" 或 "tsne"）

        Returns:
            str: 输出文件路径

        示例：
        ```python
        # 创建静态可视化图
        plot_path = visualizer._create_static_plot(
            embedding, state_info,
            "output/reports/cluster",
            "Network State Clustering",
            "umap"
        )
        print("可视化文件已保存到:", plot_path)
        ```
        """
        from src.netfaker.simcore.utils import setup_matplotlib_font
        
        # 设置Matplotlib字体，确保中文显示正常
        font_config = setup_matplotlib_font()
        
        plt.figure(figsize=(14, 12))

        # 获取状态信息
        state_ids = state_info["state_id"]
        is_pure = state_info["is_pure"]
        state_info["state_proba"]

        # 为每个状态创建散点，确保所有状态都能显示
        unique_state_ids = np.unique(state_ids)
        for state_id in unique_state_ids:
            mask = state_ids == state_id
            color = self.color_map.get(state_id, "#999999")

            # 计算透明度（用于后续扩展）
            # alphas = np.minimum(1.0, state_probas[mask] * 1.2)

            # 为不同状态使用不同的标记
            marker = "o"  # 默认圆形
            if state_id >= 3:  # 混合状态
                marker = "s"  # 正方形

            # 为样本数量较少的状态增加点的大小
            sample_count = np.sum(mask)
            size = 50
            if sample_count < 100:  # 样本数量较少的状态
                size = 80

            # 绘制点
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                alpha=0.7,
                s=size,
                marker=marker,
                label=f"{self._get_state_name(state_id)} ({sample_count})"
            )

            # 为混合状态添加灰色边框
            mixed_mask = mask & (~is_pure)
            if np.any(mixed_mask):
                plt.scatter(
                    embedding[mixed_mask, 0],
                    embedding[mixed_mask, 1],
                    c='none',
                    edgecolors='#666666',
                    linewidths=1.5,
                    s=size + 10,
                    marker=marker
                )

        plt.title(title, fontsize=16)
        plt.xlabel(f"{method.upper()} 1", fontsize=12)
        plt.ylabel(f"{method.upper()} 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()

        # 使用固定文件名，不再包含时间戳
        output_path = os.path.join(output_dir, f"clustering_gmm_{method}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _get_state_name(self, state_id: int) -> str:
        """根据状态ID获取状态名称。

        从状态元数据中查找对应状态ID的名称，如果未找到则返回默认名称。

        Args:
            state_id: 状态ID

        Returns:
            str: 状态名称

        示例：
        ```python
        # 获取状态名称
        state_name = visualizer._get_state_name(0)
        print("状态0的名称:", state_name)
        ```
        """
        if "states" in self.state_metadata:
            for state in self.state_metadata["states"]:
                if state["state_id"] == state_id:
                    return state["state_name"]
        return f"state_{state_id}"

    def _create_interactive_plot(self, embedding: np.ndarray, state_info: Dict[str, np.ndarray],
                                output_dir: str, title: str) -> str:
        """创建交互式可视化图（可选）。

        使用Plotly创建交互式UMAP可视化，支持悬停查看详情。

        Args:
            embedding: UMAP嵌入，形状为 (n_samples, 2)
            state_info: 状态信息字典
            output_dir: 输出目录
            title: 可视化标题

        Returns:
            str: 输出文件路径，如果Plotly未安装则返回空字符串

        示例：
        ```python
        # 创建交互式可视化
        interactive_path = visualizer._create_interactive_plot(
            embedding, state_info,
            "output/reports/cluster",
            "Interactive Network State Clustering"
        )
        if interactive_path:
            print("交互式可视化已保存到:", interactive_path)
        else:
            print("Plotly未安装，跳过交互式可视化")
        ```
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            # 创建数据框
            df = pd.DataFrame({
                "UMAP1": embedding[:, 0],
                "UMAP2": embedding[:, 1],
                "state_id": state_info["state_id"],
                "state_name": state_info["state_name"],
                "is_pure": state_info["is_pure"],
                "confidence": state_info["state_proba"]
            })

            # 创建颜色映射
            color_discrete_map = {
                state["state_name"]: state["color"]
                for state in self.state_metadata.get("states", [])
            }

            # 创建散点图
            fig = px.scatter(
                df,
                x="UMAP1",
                y="UMAP2",
                color="state_name",
                color_discrete_map=color_discrete_map,
                hover_data=["state_id", "state_name", "is_pure", "confidence"],
                title=title,
                opacity=0.7
            )

            # 为混合状态添加边框
            mixed_df = df[~df["is_pure"]]
            if not mixed_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=mixed_df["UMAP1"],
                        y=mixed_df["UMAP2"],
                        mode="markers",
                        marker={
                                "size": 10,
                                "color": "rgba(0,0,0,0)",
                                "line": {"color": "#666666", "width": 1.5}
                            },
                        showlegend=False,
                        hoverinfo="none"
                    )
                )

            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"clustering_gmm_{timestamp}.html")
            fig.write_html(output_path)

            return output_path
        except ImportError:
            # 如果没有安装 plotly，跳过交互式图
            return ""

    def visualize_combined(self, train_features: np.ndarray, train_state_info: Dict[str, np.ndarray],
                          test_features: np.ndarray, test_state_info: Dict[str, np.ndarray],
                          output_dir: str = "output/reports/cluster") -> Dict[str, str]:
        """可视化训练集和测试集的组合结果。

        将训练集和测试集的特征和状态信息组合后生成可视化，
        便于比较两个数据集的聚类结果。

        Args:
            train_features: 训练集特征矩阵
            train_state_info: 训练集状态信息字典
            test_features: 测试集特征矩阵
            test_state_info: 测试集状态信息字典
            output_dir: 输出目录

        Returns:
            Dict[str, str]: 输出文件路径字典，包含UMAP和t-SNE可视化

        示例：
        ```python
        # 可视化组合结果
        viz_paths = visualizer.visualize_combined(
            train_features, train_state_info,
            test_features, test_state_info,
            output_dir="output/reports/cluster"
        )
        print("组合可视化已生成:", viz_paths)
        ```
        """
        # 组合特征
        combined_features = np.vstack([train_features, test_features])

        # 组合状态信息
        combined_state_info = {
            "state_id": np.concatenate([train_state_info["state_id"], test_state_info["state_id"]]),
            "is_pure": np.concatenate([train_state_info["is_pure"], test_state_info["is_pure"]]),
            "state_proba": np.concatenate([train_state_info["state_proba"], test_state_info["state_proba"]]),
            "state_name": np.concatenate([train_state_info["state_name"], test_state_info["state_name"]])
        }

        # 生成可视化
        return self.generate_visualization(
            combined_features,
            combined_state_info,
            output_dir,
            "HoloWAN Network State Clustering (Train + Test)"
        )
