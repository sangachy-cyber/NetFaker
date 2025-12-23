#!/usr/bin/env python3

"""测试generate_valid_cond_vector函数在各种情况下的行为"""

import os
import sys
import tempfile
import json

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.full_training_evaluation import sample_and_postprocess


def test_missing_file():
    """测试提供不存在的训练文件时是否报错"""
    print("测试1: 提供不存在的训练文件")
    try:
        # 使用sample_and_postprocess函数，它内部会调用generate_valid_cond_vector
        sample_and_postprocess(
            model=None,  # 不需要实际模型，因为会在调用generate_valid_cond_vector时报错
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file="non_existent_file.jsonl",
        )
        print("❌ 测试失败：应该报错但没有报错")
    except FileNotFoundError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")


def test_empty_file():
    """测试提供空的训练文件时是否报错"""
    print("\n测试2: 提供空的训练文件")
    
    # 创建一个空的临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    
    try:
        sample_and_postprocess(
            model=None,
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file=temp_file,
        )
        print("❌ 测试失败：应该报错但没有报错")
    except ValueError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")
    finally:
        # 清理临时文件
        os.unlink(temp_file)


def test_file_without_cond():
    """测试提供没有cond字段的训练文件时是否报错"""
    print("\n测试3: 提供没有cond字段的训练文件")
    
    # 创建一个临时文件，包含没有cond字段的数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        json.dump({"window": []}, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        sample_and_postprocess(
            model=None,
            scheduler_config_path="",
            assets_dir="output/run_20251223_062813_UTC/assets",
            num_samples=1,
            output_dir="",
            device="cpu",
            train_file=temp_file,
        )
        print("❌ 测试失败：应该报错但没有报错")
    except KeyError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")
    finally:
        # 清理临时文件
        os.unlink(temp_file)


if __name__ == "__main__":
    test_missing_file()
    test_empty_file()
    test_file_without_cond()
