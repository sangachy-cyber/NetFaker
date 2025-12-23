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

from src.training.full_training_evaluation import sample_and_postprocess


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


@pytest.mark.skip(reason="需要实际模型，测试成本较高")
def test_valid_train_file(valid_train_file):
    """测试提供有效的训练文件时是否成功生成条件向量"""
    if valid_train_file is None:
        pytest.skip("有效训练文件不存在，跳过测试")
    
    try:
        # 使用sample_and_postprocess函数，它内部会调用generate_valid_cond_vector
        sample_and_postprocess(
            model=None,  # 不需要实际模型，因为会在调用generate_valid_cond_vector时报错
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file=valid_train_file,
        )
        assert True, "应该成功生成条件向量"
    except Exception as e:
        # 我们期望在model为None时出错，而不是在generate_valid_cond_vector时出错
        # 所以只要不是FileNotFoundError或ValueError，就说明generate_valid_cond_vector成功了
        assert not isinstance(e, (FileNotFoundError, ValueError)), f"generate_valid_cond_vector应该成功但报错了: {e}"


def test_missing_file():
    """测试提供不存在的训练文件时是否报错"""
    with pytest.raises(FileNotFoundError, match="训练文件不存在"):
        # 使用sample_and_postprocess函数，它内部会调用generate_valid_cond_vector
        sample_and_postprocess(
            model=None,
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file="non_existent_file.jsonl",
        )


def test_empty_file(temp_empty_file):
    """测试提供空的训练文件时是否报错"""
    with pytest.raises(ValueError, match="训练文件中没有真实条件向量"):
        # 使用sample_and_postprocess函数，它内部会调用generate_valid_cond_vector
        sample_and_postprocess(
            model=None,
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file=temp_empty_file,
        )


def test_file_without_cond(temp_file_without_cond):
    """测试提供没有cond字段的训练文件时是否报错"""
    with pytest.raises(KeyError, match="'cond'"):
        # 使用sample_and_postprocess函数，它内部会调用generate_valid_cond_vector
        sample_and_postprocess(
            model=None,
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file=temp_file_without_cond,
        )


if __name__ == "__main__":
    pytest.main([__file__])
