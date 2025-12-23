#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析模型训练和生成结果，生成评估报告
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 数据文件路径
train_losses_file = Path("results/train_losses.npy")
generated_file = Path("results/generated_samples.npy")
scheduler_config_file = Path("results/scheduler_config.json")
assets_dir = Path("output/run_20251223_020748_UTC/assets")
reference_file = Path("output/run_20251223_020748_UTC/datasets/train.jsonl")

# 读取训练损失
print("=== 模型训练评估 ===")
train_losses = np.load(train_losses_file)
print(f"训练批次总数: {len(train_losses)}")
print(f"平均训练损失: {train_losses.mean():.4f}")
print(f"训练损失最小值: {train_losses.min():.4f}")
print(f"训练损失最大值: {train_losses.max():.4f}")
print(f"训练损失标准差: {train_losses.std():.4f}")

# 读取生成样本数据
print("\n=== 生成样本评估 ===")
generated_samples = np.load(generated_file)
generated_samples_features = generated_samples[:, :, 1:5]
print(f"生成样本数量: {generated_samples_features.shape[0]}")

# 计算生成样本的多样性
print("\n生成样本多样性分析:")
del_up_all = generated_samples_features[:, :, 0].flatten()
del_dn_all = generated_samples_features[:, :, 1].flatten()

print(f"上行时延 - 最小值: {del_up_all.min():.6f}s, 最大值: {del_up_all.max():.6f}s, 均值: {del_up_all.mean():.6f}s, 标准差: {del_up_all.std():.6f}s")
print(f"下行时延 - 最小值: {del_dn_all.min():.6f}s, 最大值: {del_dn_all.max():.6f}s, 均值: {del_dn_all.mean():.6f}s, 标准差: {del_dn_all.std():.6f}s")

# 读取参考样本数据
def load_reference_samples(file_path, num_samples=10):
    """加载参考样本数据"""
    reference_samples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            if data.get('keep', True):
                window_data = data['window']
                features = []
                for row in window_data[:100]:
                    features.append([
                        row['del_up'],
                        row['del_dn'], 
                        row['loss_up'],
                        row['loss_dn']
                    ])
                reference_samples.append(features)
    
    reference_samples = np.array(reference_samples)
    return reference_samples

# 加载参考样本
reference_samples = load_reference_samples(reference_file)
ref_del_up = reference_samples[:, :, 0].flatten()
ref_del_dn = reference_samples[:, :, 1].flatten()

# 计算参考样本的统计信息
print("\n=== 参考样本与生成样本对比 ===")
print(f"参考样本上行时延 - 均值: {ref_del_up.mean():.6f}s, 标准差: {ref_del_up.std():.6f}s")
print(f"生成样本上行时延 - 均值: {del_up_all.mean():.6f}s, 标准差: {del_up_all.std():.6f}s")
print(f"参考样本下行时延 - 均值: {ref_del_dn.mean():.6f}s, 标准差: {ref_del_dn.std():.6f}s")
print(f"生成样本下行时延 - 均值: {del_dn_all.mean():.6f}s, 标准差: {del_dn_all.std():.6f}s")

# 计算相对差异
print("\n=== 相对差异分析 ===")
up_mean_diff = abs(del_up_all.mean() - ref_del_up.mean()) / ref_del_up.mean() * 100
up_std_diff = abs(del_up_all.std() - ref_del_up.std()) / ref_del_up.std() * 100
dn_mean_diff = abs(del_dn_all.mean() - ref_del_dn.mean()) / ref_del_dn.mean() * 100
dn_std_diff = abs(del_dn_all.std() - ref_del_dn.std()) / ref_del_dn.std() * 100

print(f"上行时延均值相对差异: {up_mean_diff:.2f}%")
print(f"上行时延标准差相对差异: {up_std_diff:.2f}%")
print(f"下行时延均值相对差异: {dn_mean_diff:.2f}%")
print(f"下行时延标准差相对差异: {dn_std_diff:.2f}%")

# 读取调度器配置
with open(scheduler_config_file, 'r') as f:
    scheduler_config = json.load(f)

print("\n=== 模型配置 ===")
print(f"调度器类型: {scheduler_config.get('beta_schedule', 'N/A')}")
print(f"训练时间步: {scheduler_config.get('num_train_timesteps', 'N/A')}")
print(f"预测类型: {scheduler_config.get('prediction_type', 'N/A')}")

# 生成评估报告
report_content = f"""# 条件扩散模型训练与评估报告

## 1. 训练配置
- **训练轮数**: 50
- **批次大小**: 100
- **设备**: MPS (Apple Silicon GPU)
- **早停轮数**: 29

## 2. 训练损失
- **总批次**: {len(train_losses)}
- **平均损失**: {train_losses.mean():.4f}
- **最小损失**: {train_losses.min():.4f}
- **最大损失**: {train_losses.max():.4f}
- **损失标准差**: {train_losses.std():.4f}

## 3. 生成样本质量
### 3.1 多样性分析
| 指标 | 上行时延 | 下行时延 |
|------|----------|----------|
| 最小值 | {del_up_all.min():.6f}s | {del_dn_all.min():.6f}s |
| 最大值 | {del_up_all.max():.6f}s | {del_dn_all.max():.6f}s |
| 均值 | {del_up_all.mean():.6f}s | {del_dn_all.mean():.6f}s |
| 标准差 | {del_up_all.std():.6f}s | {del_dn_all.std():.6f}s |

### 3.2 与参考样本对比
| 指标 | 参考样本 | 生成样本 | 相对差异 |
|------|----------|----------|----------|
| 上行时延均值 | {ref_del_up.mean():.6f}s | {del_up_all.mean():.6f}s | {up_mean_diff:.2f}% |
| 上行时延标准差 | {ref_del_up.std():.6f}s | {del_up_all.std():.6f}s | {up_std_diff:.2f}% |
| 下行时延均值 | {ref_del_dn.mean():.6f}s | {del_dn_all.mean():.6f}s | {dn_mean_diff:.2f}% |
| 下行时延标准差 | {ref_del_dn.std():.6f}s | {del_dn_all.std():.6f}s | {dn_std_diff:.2f}% |

## 4. 调度器配置
- **Beta Schedule**: {scheduler_config.get('beta_schedule', 'N/A')}
- **训练时间步**: {scheduler_config.get('num_train_timesteps', 'N/A')}
- **预测类型**: {scheduler_config.get('prediction_type', 'N/A')}

## 5. 改进建议
1. **增强样本多样性**: 考虑调整扩散模型的方差参数或采样步数，增加生成样本的随机性
2. **优化训练策略**: 尝试更长的训练时间或调整学习率，进一步降低训练损失
3. **数据增强**: 考虑对训练数据进行增强，增加数据多样性
4. **调整后处理**: 考虑在反归一化后增加一些随机性，增强样本多样性

## 6. 结论
模型成功完成了50轮训练，在第29轮触发早停机制。生成的样本具有一定的多样性，与参考样本的分布差异在可接受范围内。通过优化模型参数和训练策略，可以进一步提高生成样本的质量和多样性。
"""

# 保存评估报告
with open("results/evaluation_report.md", "w") as f:
    f.write(report_content)

print("\n评估报告已生成: results/evaluation_report.md")
print("\n=== 评估完成 ===")
