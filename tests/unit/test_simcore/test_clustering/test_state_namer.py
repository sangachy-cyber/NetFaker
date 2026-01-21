"""StateNamer 类的测试文件。"""

import os
import json
import numpy as np
import pytest
from netfaker.simcore.clustering.state_namer import StateNamer


class TestStateNamer:
    """StateNamer 类的测试用例。"""
    
    def test_init(self):
        """测试初始化功能。"""
        # 测试默认参数
        namer = StateNamer()
        assert namer.n_components == 3
        assert namer.confidence_threshold == 0.85
        
        # 测试自定义参数
        namer = StateNamer(n_components=5, confidence_threshold=0.9)
        assert namer.n_components == 5
        assert namer.confidence_threshold == 0.9
    
    def test_assign_states_pure(self):
        """测试分配纯净状态。"""
        namer = StateNamer(n_components=3, confidence_threshold=0.85)
        
        # 创建高置信度的概率矩阵
        probas = np.array([
            [0.95, 0.03, 0.02],  # 属于状态0
            [0.01, 0.98, 0.01],  # 属于状态1
            [0.02, 0.03, 0.95]   # 属于状态2
        ])
        
        state_info = namer.assign_states(probas)
        
        # 验证结果
        assert "state_id" in state_info
        assert "state_name" in state_info
        assert "is_pure" in state_info
        assert "state_proba" in state_info
        assert "top2_state_ids" in state_info
        assert "top2_state_probas" in state_info
        
        # 验证纯净状态
        assert all(state_info["is_pure"])
        assert np.array_equal(state_info["state_id"], [0, 1, 2])
        assert len(state_info["state_name"]) == 3
        assert len(state_info["state_proba"]) == 3
        assert state_info["top2_state_ids"].shape == (3, 2)
        assert state_info["top2_state_probas"].shape == (3, 2)
    
    def test_assign_states_mixed(self):
        """测试分配混合状态。"""
        namer = StateNamer(n_components=3, confidence_threshold=0.85)
        
        # 创建低置信度的概率矩阵
        probas = np.array([
            [0.6, 0.3, 0.1],   # 混合状态
            [0.4, 0.5, 0.1],   # 混合状态
            [0.3, 0.3, 0.4]    # 混合状态
        ])
        
        state_info = namer.assign_states(probas)
        
        # 验证结果
        assert "state_id" in state_info
        assert "state_name" in state_info
        assert "is_pure" in state_info
        assert "state_proba" in state_info
        assert "top2_state_ids" in state_info
        assert "top2_state_probas" in state_info
        
        # 验证混合状态
        assert not any(state_info["is_pure"])
        assert all(state_info["state_id"] >= 3)  # 混合状态ID >= 3
        assert len(state_info["state_name"]) == 3
        assert len(state_info["state_proba"]) == 3
        assert state_info["top2_state_ids"].shape == (3, 2)
        assert state_info["top2_state_probas"].shape == (3, 2)
    
    def test_generate_metadata(self):
        """测试生成元数据。"""
        namer = StateNamer(n_components=3, confidence_threshold=0.85)
        
        # 创建概率矩阵并分配状态
        probas = np.array([
            [0.95, 0.03, 0.02],  # 属于状态0
            [0.01, 0.98, 0.01],  # 属于状态1
            [0.02, 0.03, 0.95],  # 属于状态2
            [0.6, 0.3, 0.1]       # 混合状态
        ])
        
        state_info = namer.assign_states(probas)
        metadata = namer.generate_metadata()
        
        # 验证元数据结构
        assert "n_components" in metadata
        assert "confidence_threshold" in metadata
        assert "states" in metadata
        
        # 验证基础状态
        base_states = [s for s in metadata["states"] if s["state_id"] < 3]
        assert len(base_states) == 3
        
        # 验证混合状态
        mixed_states = [s for s in metadata["states"] if s["state_id"] >= 3]
        assert len(mixed_states) >= 1
        
        # 验证状态属性
        for state in metadata["states"]:
            assert "state_id" in state
            assert "state_name" in state
            assert "color" in state
            assert "description" in state
    
    def test_save_metadata(self, tmp_path):
        """测试保存元数据到文件。"""
        namer = StateNamer(n_components=3, confidence_threshold=0.85)
        
        # 创建概率矩阵并分配状态
        probas = np.array([
            [0.95, 0.03, 0.02],  # 属于状态0
            [0.01, 0.98, 0.01],  # 属于状态1
            [0.02, 0.03, 0.95]   # 属于状态2
        ])
        
        state_info = namer.assign_states(probas)
        
        # 保存元数据
        output_path = tmp_path / "state_metadata.json"
        namer.save_metadata(str(output_path))
        
        # 验证文件存在
        assert output_path.exists()
        
        # 验证文件内容
        with open(output_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        assert "n_components" in metadata
        assert "confidence_threshold" in metadata
        assert "states" in metadata
        assert len(metadata["states"]) >= 3


if __name__ == "__main__":
    pytest.main(["-v", __file__])
