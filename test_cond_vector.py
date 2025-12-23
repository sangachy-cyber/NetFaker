#!/usr/bin/env python3

"""测试generate_valid_cond_vector函数在各种情况下的行为"""

import os
import sys
import tempfile
import json
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_generate_valid_cond_vector():
    """测试generate_valid_cond_vector函数的各种情况"""
    
    # 模拟函数定义，直接复制修改后的函数代码进行测试
    def generate_valid_cond_vector(train_file, num_network_states=8):
        """生成有效的条件向量
        
        Args:
            train_file: 训练数据文件路径，必须存在且包含真实条件向量
            num_network_states: 网络状态ID的数量
            
        Returns:
            从训练数据中提取的真实条件向量
            
        Raises:
            FileNotFoundError: 如果训练文件不存在
            ValueError: 如果训练文件中没有真实条件向量
        """
        # 检查训练文件是否存在
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        
        # 从训练数据中提取真实的条件向量分布
        real_conds = []
        with open(train_file) as f:
            for line in f:
                data = json.loads(line)
                real_conds.append(data["cond"])
                if len(real_conds) >= 100:  # 最多使用100个真实条件向量
                    break
        
        # 检查是否提取到真实条件向量
        if not real_conds:
            raise ValueError(f"训练文件中没有真实条件向量: {train_file}")
        
        # 随机选择一个真实条件向量
        # 训练数据中的条件向量已经是标准化的，所以可以直接使用
        real_cond = real_conds[np.random.randint(0, len(real_conds))]
        cond = torch.FloatTensor(real_cond)
        
        # 确保网络状态ID在合理范围内
        cond[11] = float(np.random.randint(0, num_network_states))
        
        return cond
    
    # 测试1: 提供不存在的训练文件
    print("测试1: 提供不存在的训练文件")
    try:
        generate_valid_cond_vector("non_existent_file.jsonl")
        print("❌ 测试失败：应该报错但没有报错")
    except FileNotFoundError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")
    
    # 测试2: 提供空的训练文件
    print("\n测试2: 提供空的训练文件")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name
    
    try:
        generate_valid_cond_vector(temp_file)
        print("❌ 测试失败：应该报错但没有报错")
    except ValueError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")
    finally:
        os.unlink(temp_file)
    
    # 测试3: 提供没有cond字段的训练文件
    print("\n测试3: 提供没有cond字段的训练文件")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        json.dump({"window": []}, f)
        f.write('\n')
        temp_file = f.name
    
    try:
        generate_valid_cond_vector(temp_file)
        print("❌ 测试失败：应该报错但没有报错")
    except KeyError as e:
        print(f"✅ 测试通过：正确报错：{e}")
    except Exception as e:
        print(f"❌ 测试失败：报错类型不正确：{type(e).__name__}: {e}")
    finally:
        os.unlink(temp_file)
    
    # 测试4: 提供有效的训练文件
    print("\n测试4: 提供有效的训练文件")
    
    # 检查是否存在有效的训练文件
    valid_train_file = "output/run_20251223_062813_UTC/datasets/train.jsonl"
    if os.path.exists(valid_train_file):
        try:
            cond = generate_valid_cond_vector(valid_train_file)
            print(f"✅ 测试通过：成功生成条件向量，形状为: {cond.shape}")
        except Exception as e:
            print(f"❌ 测试失败：应该成功但报错了：{type(e).__name__}: {e}")
    else:
        print("⚠️  跳过测试4：有效训练文件不存在")


if __name__ == "__main__":
    test_generate_valid_cond_vector()
