#!/usr/bin/env python3

"""测试generate_valid_cond_vector函数在各种情况下的行为"""

import os
import sys
import tempfile
import json
from pathlib import Path

import pytest

# 添加项目根目录到Python搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.full_training_evaluation import generate_valid_cond_vector


@pytest.fixture
def valid_train_file():
    """提供有效的训练文件路径"""
    train_file = "output/run_20251223_062813_UTC/datasets/train.jsonl"
    if os.path.exists(train_file):
        return train_file
    return None


@pytest.fixture
def temp_empty_file():
    """创建一个空的临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    yield temp_file
    # 清理临时文件
    os.unlink(temp_file)


@pytest.fixture
def temp_file_without_cond():
    """创建一个没有cond字段的临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        json.dump({"window": []}, f)
        f.write('\n')
        temp_file = f.name
    yield temp_file
    # 清理临时文件
    os.unlink(temp_file)


@pytest.fixture
def temp_valid_file():
    """创建一个包含有效条件向量的临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # 写入多个包含cond字段的样本
        for i in range(5):
            data = {
                "window": [],
                "cond": [0.0] * 23  # 23维条件向量
            }
            json.dump(data, f)
            f.write('\n')
        temp_file = f.name
    yield temp_file
    # 清理临时文件
    os.unlink(temp_file)


@pytest.mark.skip(reason="需要实际训练文件，测试成本较高")
def test_valid_train_file(valid_train_file):
    """测试提供有效的训练文件时是否成功生成条件向量"""
    if valid_train_file is None:
        pytest.skip("有效训练文件不存在，跳过测试")
    
    # 直接测试generate_valid_cond_vector函数
    cond_vector, window_data, _ = generate_valid_cond_vector(valid_train_file)
    assert cond_vector is not None
    assert cond_vector.shape == (23,)  # 条件向量应该是23维的
    assert isinstance(window_data, list)  # 窗口数据应该是列表类型


def test_valid_temp_file(temp_valid_file):
    """测试使用有效的临时训练文件时是否成功生成条件向量"""
    # 直接测试generate_valid_cond_vector函数
    cond_vector, window_data, _ = generate_valid_cond_vector(temp_valid_file)
    assert cond_vector is not None
    assert cond_vector.shape == (23,)  # 条件向量应该是23维的
    assert isinstance(window_data, list)  # 窗口数据应该是列表类型


def test_missing_file():
    """测试提供不存在的训练文件时是否报错"""
    with pytest.raises(FileNotFoundError, match="训练文件不存在"):
        # 直接测试generate_valid_cond_vector函数
        generate_valid_cond_vector("non_existent_file.jsonl")


def test_empty_file(temp_empty_file):
    """测试提供空的训练文件时是否报错"""
    with pytest.raises(ValueError, match="训练文件中没有真实条件向量"):
        # 直接测试generate_valid_cond_vector函数
        generate_valid_cond_vector(temp_empty_file)


def test_file_without_cond(temp_file_without_cond):
    """测试提供没有cond字段的训练文件时是否报错"""
    with pytest.raises(KeyError, match="'cond'"):
        # 直接测试generate_valid_cond_vector函数
        generate_valid_cond_vector(temp_file_without_cond)


if __name__ == "__main__":
    pytest.main([__file__])
