"""Visualizer 类的测试文件。"""

import os
import json
import numpy as np
import pytest
from netfaker.simcore.clustering.visualizer import Visualizer


class TestVisualizer:
    """Visualizer 类的测试用例。"""
    
    def test_init(self, tmp_path):
        """测试初始化功能。"""
        # 创建临时元数据文件
        metadata = {
            "n_components": 3,
            "confidence_threshold": 0.85,
            "states": [
                {"state_id": 0, "state_name": "State 0", "color": "#FF0000", "description": "State 0 description"},
                {"state_id": 1, "state_name": "State 1", "color": "#00FF00", "description": "State 1 description"},
                {"state_id": 2, "state_name": "State 2", "color": "#0000FF", "description": "State 2 description"}
            ]
        }
        
        metadata_path = tmp_path / "state_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        # 测试初始化
        visualizer = Visualizer(str(metadata_path))
        
        # 验证属性
        assert visualizer.state_metadata_path == str(metadata_path)
        assert "states" in visualizer.state_metadata
        assert len(visualizer.state_metadata["states"]) == 3
        assert visualizer.color_map[0] == "#FF0000"
        assert visualizer.color_map[1] == "#00FF00"
        assert visualizer.color_map[2] == "#0000FF"
    
    def test_init_no_metadata(self):
        """测试没有元数据文件的情况。"""
        # 使用不存在的文件路径
        visualizer = Visualizer("non_existent_metadata.json")
        
        # 验证属性
        assert visualizer.state_metadata_path == "non_existent_metadata.json"
        assert visualizer.state_metadata == {}
        assert visualizer.color_map == {}
    
    def test_run_tsne(self):
        """测试 t-SNE 降维功能。"""
        visualizer = Visualizer()
        
        # 创建测试数据
        X = np.random.randn(100, 16)  # 100个样本，16维特征
        
        # 执行 t-SNE 降维
        embedding = visualizer._run_tsne(X)
        
        # 验证结果
        assert embedding.shape == (100, 2)
        assert embedding.dtype == np.float64
    
    def test_get_state_name(self, tmp_path):
        """测试获取状态名称功能。"""
        # 创建临时元数据文件
        metadata = {
            "states": [
                {"state_id": 0, "state_name": "Good", "color": "#FF0000"},
                {"state_id": 1, "state_name": "Moderate", "color": "#00FF00"},
                {"state_id": 2, "state_name": "Poor", "color": "#0000FF"}
            ]
        }
        
        metadata_path = tmp_path / "state_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        visualizer = Visualizer(str(metadata_path))
        
        # 测试存在的状态
        assert visualizer._get_state_name(0) == "Good"
        assert visualizer._get_state_name(1) == "Moderate"
        assert visualizer._get_state_name(2) == "Poor"
        
        # 测试不存在的状态
        assert visualizer._get_state_name(3) == "state_3"
        assert visualizer._get_state_name(999) == "state_999"
    
    def test_visualize_combined(self, tmp_path):
        """测试组合可视化功能。"""
        # 创建临时元数据文件
        metadata = {
            "states": [
                {"state_id": 0, "state_name": "State 0", "color": "#FF0000"},
                {"state_id": 1, "state_name": "State 1", "color": "#00FF00"},
                {"state_id": 2, "state_name": "State 2", "color": "#0000FF"}
            ]
        }
        
        metadata_path = tmp_path / "state_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        visualizer = Visualizer(str(metadata_path))
        
        # 创建测试数据
        train_features = np.random.randn(50, 16)
        test_features = np.random.randn(50, 16)
        
        # 创建状态信息
        train_state_info = {
            "state_id": np.random.randint(0, 3, 50),
            "is_pure": np.random.choice([True, False], 50),
            "state_proba": np.random.rand(50),
            "state_name": [f"State {i}" for i in np.random.randint(0, 3, 50)]
        }
        
        test_state_info = {
            "state_id": np.random.randint(0, 3, 50),
            "is_pure": np.random.choice([True, False], 50),
            "state_proba": np.random.rand(50),
            "state_name": [f"State {i}" for i in np.random.randint(0, 3, 50)]
        }
        
        # 创建输出目录
        output_dir = tmp_path / "output" / "reports" / "cluster"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行可视化
        viz_paths = visualizer.visualize_combined(
            train_features, train_state_info,
            test_features, test_state_info,
            output_dir=str(output_dir)
        )
        
        # 验证结果
        assert isinstance(viz_paths, dict)
        assert "tsne" in viz_paths
        
        # 验证文件存在
        tsne_path = viz_paths["tsne"]
        assert os.path.exists(tsne_path)
        assert tsne_path.endswith(".png")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
