#!/usr/bin/env python3

"""完整的条件扩散模型训练和评估流程
包括训练、采样、后处理和可视化
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from src.models.conditional_diffusion import (
    DiffusionSampler,
    DiffusionTrainer,
    PaddedConditionalUNet1D,
    PostProcessor,
)

warnings.filterwarnings("ignore")


def generate_valid_cond_vector(train_file, num_network_states=8, num_consecutive_windows=60):
    """生成有效的条件向量
    
    Args:
        train_file: 训练数据文件路径，必须存在且包含真实条件向量
        num_network_states: 网络状态ID的数量
        num_consecutive_windows: 连续窗口数量，默认60个窗口（10分钟）
        
    Returns:
        从训练数据中提取的真实条件向量及其对应的窗口数据，以及连续的条件向量列表
        
    Raises:
        FileNotFoundError: 如果训练文件不存在
        ValueError: 如果训练文件中没有真实条件向量
        KeyError: 如果训练文件中的数据没有'cond'字段
    """
    import os
    import numpy as np
    # 检查训练文件是否存在
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    
    # 从训练数据中提取所有真实条件向量和对应的窗口数据
    all_real_data = []
    with open(train_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("keep", True):
                # 直接从数据中获取条件向量和对应的窗口数据
                if "cond" in data and "window" in data:
                    all_real_data.append((data["cond"], data["window"]))
                else:
                    # 如果没有cond字段或window字段，抛出KeyError
                    raise KeyError("'cond'")
    
    # 只打印一次获取信息
    print(f"条件向量: 从训练数据中获取初始条件向量，随机选择连续 {num_consecutive_windows} 个窗口")
    
    # 检查是否提取到真实条件向量
    if not all_real_data:
        raise ValueError(f"训练文件中没有真实条件向量: {train_file}")
    
    # 如果条件向量数量不足，使用所有可用的条件向量
    if len(all_real_data) < num_consecutive_windows:
        print(f"警告：训练文件中条件向量数量不足 {num_consecutive_windows} 个，仅使用 {len(all_real_data)} 个")
        start_idx = 0
        # 使用所有可用的条件向量
        consecutive_data = all_real_data
    else:
        # 随机选择起始位置，确保有足够的连续窗口
        max_start_idx = len(all_real_data) - num_consecutive_windows
        start_idx = np.random.randint(0, max_start_idx + 1)
        # 获取连续的窗口数据
        consecutive_data = all_real_data[start_idx:start_idx + num_consecutive_windows]
    
    # 提取第一个窗口的条件向量作为初始条件向量
    # 保留真实数据中的网络状态ID，不做修改
    real_cond, real_window = consecutive_data[0]
    cond = torch.FloatTensor(real_cond)
    
    # 提取所有连续的条件向量
    consecutive_conds = [torch.FloatTensor(data[0]) for data in consecutive_data]
    
    return cond, real_window, consecutive_conds


class NetworkTraceDataset(Dataset):
    """网络轨迹数据集"""

    def __init__(self, file_path):
        self.windows = []
        self.conditions = []

        # 读取数据
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("keep", True):  # 只使用标记为keep的数据
                    window_data = data["window"]
                    cond_data = data["cond"]

                    # 提取特征，忽略时间戳列
                    # 数据格式: [timestamp, del_up, del_dn, loss_up, loss_dn]
                    features = []
                    for row in window_data:
                        features.append([
                            row["del_up"],
                            row["del_dn"],
                            row["loss_up"],
                            row["loss_dn"],
                        ])

                    # 确保窗口数据长度为100（模型期望的长度）
                    if len(features) > 100:
                        features = features[:100]
                    elif len(features) < 100:
                        # 用最后一行填充
                        while len(features) < 100:
                            features.append(features[-1])

                    # 转换为张量
                    features = torch.FloatTensor(features)  # (100, 4)
                    cond = torch.FloatTensor(cond_data)     # (22,)

                    # 调整维度以匹配模型输入: (4, 100)
                    features = features.transpose(0, 1)

                    self.windows.append(features)
                    self.conditions.append(cond)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            "window": self.windows[idx],
            "cond": self.conditions[idx],
        }


def collate_fn(batch):
    """批处理函数"""
    windows = torch.stack([item["window"] for item in batch])
    conds = torch.stack([item["cond"] for item in batch])

    # 在索引11位置插入网络状态ID（默认为0）
    cond_with_state = []
    for cond in conds:
        # 在索引11位置插入0.0作为网络状态ID
        new_cond = torch.cat([cond[:11], torch.tensor([0.0]), cond[11:]])
        cond_with_state.append(new_cond)
    conds_with_state = torch.stack(cond_with_state)

    return {
        "window": windows,
        "cond": conds_with_state,
    }


def train_model(train_file, val_file=None, epochs=10, batch_size=100, learning_rate=1e-4, output_dir="./output/visualization"):
    """训练模型"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建数据集
    train_dataset = NetworkTraceDataset(train_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 创建验证数据集（如果提供了验证文件）
    val_loader = None
    if val_file and os.path.exists(val_file):
        val_dataset = NetworkTraceDataset(val_file)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 创建模型和调度器
    model = PaddedConditionalUNet1D().to(device)
    # 确保调度器配置正确，与模型输出类型匹配
    # 使用cosine调度和fixed_small方差类型，增强对合理波动的建模能力
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",  # cosine schedule，前期加更多噪
        variance_type="fixed_small",         # 更稳定
        prediction_type="sample",  # 修改为sample模式，与训练目标一致
        clip_sample=False,  # 不裁剪样本以保持数据分布
    )

    # 创建训练器
    trainer = DiffusionTrainer(model, scheduler, device=device, patience=5)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练损失
    train_losses = []
    val_losses = []

    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # 执行训练步骤
            loss = trainer.train_step(batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            train_losses.append(loss.item())

            # 每100个批次打印一次损失
            if num_batches % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # 验证阶段
        if val_loader is not None:
            val_total_loss = 0
            val_num_batches = 0
            for val_batch in val_loader:
                val_loss = trainer.validation_step(val_batch)
                val_total_loss += val_loss.item()
                val_num_batches += 1
                val_losses.append(val_loss.item())

            val_avg_loss = val_total_loss / val_num_batches
            print(f"Epoch {epoch+1}/{epochs} Validation Loss: {val_avg_loss:.4f}")

            # 检查早停条件
            if trainer.check_early_stopping(val_avg_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 保存模型
    model_save_path = output_path / "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)

    # ✅ 保存 scheduler config（不是 state_dict！）
    scheduler_config_path = output_path / "scheduler_config.json"
    scheduler_config = scheduler.config
    with open(scheduler_config_path, "w") as f:
        json.dump(scheduler_config, f, indent=4)

    # 保存训练损失
    loss_save_path = output_path / "train_losses.npy"
    np.save(loss_save_path, np.array(train_losses))

    return model, scheduler_config_path  # 返回路径或 config 均可


def visualize_training_losses(losses_file, output_dir="./output/visualization"):
    """可视化训练损失曲线"""
    # 加载训练损失
    train_losses = np.load(losses_file)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)

    # 保存图像
    output_path = Path(output_dir)
    loss_curve_path = output_path / "training_loss_curve.png"
    plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training loss curve saved as {loss_curve_path}")


def get_initial_cond_vector(test_file, train_file=None):
    """获取初始条件向量
    
    Args:
        test_file: 测试数据文件路径，优先使用测试集获取参考样本
        train_file: 训练数据文件路径，仅在测试集不可用时作为备选
        
    Returns:
        tuple: (cond_vector, window_data, consecutive_conds)
            cond_vector: 初始条件向量，shape (23,)
            window_data: 参考窗口数据，如果没有参考样本则为None
            consecutive_conds: 连续的条件向量列表
    """
    # 优先使用测试集获取参考样本
    if test_file and os.path.exists(test_file):
        cond, window, consecutive_conds = generate_valid_cond_vector(test_file)
        return cond, window, consecutive_conds
    # 如果测试集不可用，尝试使用训练集作为备选
    elif train_file and os.path.exists(train_file):
        cond, window, consecutive_conds = generate_valid_cond_vector(train_file)
        return cond, window, consecutive_conds
    else:
        # 如果没有参考样本，创建一个默认的条件向量
        # 注意：这里使用torch.zeros(23)作为默认值，实际应用中可以从保存的条件向量中随机选择
        cond = torch.zeros(23)
        cond[11] = 0.0  # network_state_id
        return cond, None, [cond]


def sample_and_postprocess(model, scheduler_config_path, assets_dir, num_samples=10, num_windows_per_sample=1, output_dir="./output/visualization", device="cpu", test_file=None, train_file=None, guidance_scale=1.0):
    """采样并后处理生成的数据
    
    Args:
        model: 训练好的模型
        scheduler_config_path: 调度器配置文件路径
        assets_dir: 资产文件目录
        num_samples: 生成的样本数量
        num_windows_per_sample: 每个样本包含的窗口数量
        output_dir: 输出目录
        device: 设备
        test_file: 测试数据文件路径，用于获取参考样本
        train_file: 训练数据文件路径，仅在测试集不可用时作为备选
        
    Returns:
        np.ndarray: 生成的样本数据
    """
    with open(scheduler_config_path) as f:
        scheduler_config = json.load(f)

    sampling_scheduler = DDPMScheduler.from_config(scheduler_config)
    sampler = DiffusionSampler(model, sampling_scheduler, device)

    import joblib
    qt_up = joblib.load(Path(assets_dir) / "qt_up.pkl")
    qt_dn = joblib.load(Path(assets_dir) / "qt_down.pkl")

    postprocessor = PostProcessor(qt_up, qt_dn)

    # 加载条件向量标准化参数（当前未使用，保留代码用于后续扩展）
    # cond_mean = np.load(Path(assets_dir) / "../meta" / "cond_mean.npy")
    # cond_std = np.load(Path(assets_dir) / "../meta" / "cond_std.npy")

    # 生成样本
    print("Generating samples...")
    all_generated_samples = []
    
    for sample_idx in range(num_samples):
        print(f"\n生成样本 {sample_idx+1}/{num_samples}...")
        
        # 获取初始条件向量和连续的条件向量 - 优先使用测试集
        init_cond, reference_window, consecutive_conds = get_initial_cond_vector(test_file, train_file)
        
        # 使用连续的条件向量逐个生成窗口
        all_windows = []
        for window_idx in range(num_windows_per_sample):
            print(f"  生成窗口 {window_idx+1}/{num_windows_per_sample}...")
            # 使用真实的条件向量
            cond = consecutive_conds[window_idx % len(consecutive_conds)]
            cond_tensor = cond.unsqueeze(0).to(device)
            generated_sample = sampler.sample(cond_tensor, num_inference_steps=100, guidance_scale=guidance_scale)  # 按照要求使用100步推理
            processed_sample = postprocessor.postprocess(
                generated_sample, base_time=window_idx * 10.0
            )
            
            # 添加到结果列表
            all_windows.append(processed_sample)
        
        # 拼接长序列
        long_sample = np.concatenate(all_windows, axis=1)
        all_generated_samples.append(long_sample)
    
    # 合并所有生成的样本
    final_samples = np.concatenate(all_generated_samples, axis=0)
    print(f"\n生成的样本总形状: {final_samples.shape}")

    # 保存生成的样本
    output_path = Path(output_dir)
    samples_path = output_path / "generated_samples.npy"
    np.save(samples_path, final_samples)
    print(f"Generated samples saved as {samples_path}")

    return final_samples


def update_cond_vector(prev_window, prev_cond, cond_mean, cond_std, qt_up, qt_dn):
    """从前一个窗口和条件向量更新下一个条件向量
    
    Args:
        prev_window: 前一个窗口的生成结果，shape (100, 5)
        prev_cond: 前一个窗口使用的条件向量，shape (23,)
        cond_mean: 条件向量均值，shape (22,)
        cond_std: 条件向量标准差，shape (22,)
        qt_up: 上行延迟的QuantileTransformer
        qt_dn: 下行延迟的QuantileTransformer
        
    Returns:
        new_cond: 更新后的条件向量，shape (23,)
    """
    # 忽略局部特征更新，直接返回保持局部特征为零向量的条件向量
    new_cond = prev_cond.copy()
    # 保持局部特征为零向量
    new_cond[13:23] = 0.0
    
    return new_cond


def visualize_generated_samples(generated_samples_file, real_data_file, output_dir="./output/visualization"):
    """可视化生成的样本与真实数据的对比"""
    # 加载生成的样本
    generated_samples = np.load(generated_samples_file)

    # 加载真实数据样本
    real_samples = []
    with open(real_data_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data.get("keep", True):
                window_data = data["window"]
                # 提取特征 - 加载所有时间步
                features = []
                for row in window_data:  # 加载所有时间步
                    features.append([
                        row["del_up"],
                        row["del_dn"],
                        row["loss_up"],
                        row["loss_dn"],
                    ])
                real_samples.append(features)

    real_samples = np.array(real_samples)

    # 创建输出目录
    output_path = Path(output_dir)

    # 计算丢包率统计信息
    real_loss_up = real_samples[:, :, 2].flatten()
    real_loss_dn = real_samples[:, :, 3].flatten()
    generated_loss_up = generated_samples[:, :, 3].flatten()
    generated_loss_dn = generated_samples[:, :, 4].flatten()
    
    # 统计信息
    loss_metrics = {
        "real_loss_up": {
            "mean": np.mean(real_loss_up),
            "std": np.std(real_loss_up),
            "non_zero_ratio": np.mean(real_loss_up > 0),
            "count": len(real_loss_up)
        },
        "real_loss_dn": {
            "mean": np.mean(real_loss_dn),
            "std": np.std(real_loss_dn),
            "non_zero_ratio": np.mean(real_loss_dn > 0),
            "count": len(real_loss_dn)
        },
        "generated_loss_up": {
            "mean": np.mean(generated_loss_up),
            "std": np.std(generated_loss_up),
            "non_zero_ratio": np.mean(generated_loss_up > 0),
            "count": len(generated_loss_up)
        },
        "generated_loss_dn": {
            "mean": np.mean(generated_loss_dn),
            "std": np.std(generated_loss_dn),
            "non_zero_ratio": np.mean(generated_loss_dn > 0),
            "count": len(generated_loss_dn)
        }
    }
    
    # 打印丢包率分布
    print("\n丢包率分布详情：")
    
    # 真实数据上行丢包率分布
    real_up_unique, real_up_counts = np.unique(real_loss_up, return_counts=True)
    print("真实数据上行丢包率：")
    for value, count in zip(real_up_unique, real_up_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(real_loss_up)*100:.2f}%)")
    
    # 真实数据下行丢包率分布
    real_dn_unique, real_dn_counts = np.unique(real_loss_dn, return_counts=True)
    print("真实数据下行丢包率：")
    for value, count in zip(real_dn_unique, real_dn_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(real_loss_dn)*100:.2f}%)")
    
    # 生成数据上行丢包率分布
    gen_up_unique, gen_up_counts = np.unique(generated_loss_up, return_counts=True)
    print("生成数据上行丢包率：")
    for value, count in zip(gen_up_unique, gen_up_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(generated_loss_up)*100:.2f}%)")
    
    # 生成数据下行丢包率分布
    gen_dn_unique, gen_dn_counts = np.unique(generated_loss_dn, return_counts=True)
    print("生成数据下行丢包率：")
    for value, count in zip(gen_dn_unique, gen_dn_counts):
        print(f"  {value:.6f}: {count} 次 ({count/len(generated_loss_dn)*100:.2f}%)")
    
    # 打印统计信息
    print("\n丢包率统计信息：")
    for name, metrics in loss_metrics.items():
        print(f"{name}: 均值={metrics['mean']:.6f}, 标准差={metrics['std']:.6f}, 非零值比例={metrics['non_zero_ratio']:.6f}, 样本数={metrics['count']}")
    
    # 绘制延迟数据的直方图对比
    plt.figure(figsize=(15, 12))

    # 上行延迟
    plt.subplot(3, 2, 1)
    real_del_up = real_samples[:, :, 0].flatten()
    generated_del_up = generated_samples[:, :, 1].flatten()  # 注意索引对应关系
    plt.hist(real_del_up, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_up, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("上行延迟分布")
    plt.xlabel("延迟 (归一化)")
    plt.ylabel("密度")
    plt.legend()

    # 下行延迟
    plt.subplot(3, 2, 2)
    real_del_dn = real_samples[:, :, 1].flatten()
    generated_del_dn = generated_samples[:, :, 2].flatten()
    plt.hist(real_del_dn, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_dn, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("下行延迟分布")
    plt.xlabel("延迟 (归一化)")
    plt.ylabel("密度")
    plt.legend()

    # 上行丢包
    plt.subplot(3, 2, 3)
    plt.hist(real_loss_up, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="真实数据", density=True, align="left")
    plt.hist(generated_loss_up, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="生成数据", density=True, align="left")
    plt.xticks([0, 1], ["0", "1"])
    plt.title("上行丢包率分布")
    plt.xlabel("丢包率")
    plt.ylabel("密度")
    plt.legend()
    
    # 添加上行丢包率统计信息
    plt.text(0.02, 0.95, 
             f"真实数据: 均值={loss_metrics['real_loss_up']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['real_loss_up']['non_zero_ratio']:.6f}\n" +
             f"生成数据: 均值={loss_metrics['generated_loss_up']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['generated_loss_up']['non_zero_ratio']:.6f}",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 下行丢包
    plt.subplot(3, 2, 4)
    plt.hist(real_loss_dn, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="真实数据", density=True, align="left")
    plt.hist(generated_loss_dn, bins=[-0.1, 0.1, 0.9, 1.1], alpha=0.7, label="生成数据", density=True, align="left")
    plt.xticks([0, 1], ["0", "1"])
    plt.title("下行丢包率分布")
    plt.xlabel("丢包率")
    plt.ylabel("密度")
    plt.legend()
    
    # 添加下行丢包率统计信息和说明
    plt.text(0.02, 0.95, 
             f"真实数据: 均值={loss_metrics['real_loss_dn']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['real_loss_dn']['non_zero_ratio']:.6f}\n" +
             f"生成数据: 均值={loss_metrics['generated_loss_dn']['mean']:.6f}\n" +
             f"非零值比例={loss_metrics['generated_loss_dn']['non_zero_ratio']:.6f}",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 丢包率说明
    plt.text(0.02, 0.5,
             "说明：\n" +
             "1. 原始训练数据中丢包率大部分为0\n" +
             "2. 预处理后，丢包率简化为0和1两个值\n" +
             "3. 模型从训练数据中学习生成丢包率",
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Q-Q图：上行延迟
    plt.subplot(3, 2, 5)
    stats.probplot(generated_del_up, dist="norm", plot=plt)
    plt.title("Q-Q图：上行延迟")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")

    # Q-Q图：下行延迟
    plt.subplot(3, 2, 6)
    stats.probplot(generated_del_dn, dist="norm", plot=plt)
    plt.title("Q-Q图：下行延迟")
    plt.xlabel("理论分位数")
    plt.ylabel("样本分位数")

    plt.tight_layout()

    # 保存图像
    comparison_path = output_path / "data_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data comparison visualization saved as {comparison_path}")
    
    # 额外保存上行丢包率时间序列图（显示所有时间步）
    if generated_samples.shape[0] > 0:
        plt.figure(figsize=(20, 10))
        for sample_idx in range(min(generated_samples.shape[0], 2)):  # 显示前2个样本
            generated_loss_time = generated_samples[sample_idx, :, 3]
            time_steps = np.arange(len(generated_loss_time))
            
            plt.subplot(min(generated_samples.shape[0], 2), 1, sample_idx+1)
            plt.plot(time_steps, generated_loss_time, label=f"样本 {sample_idx+1} 上行丢包率", linewidth=1)
            plt.title(f"生成数据上行丢包率时间序列 - 样本 {sample_idx+1} (共 {len(time_steps)} 步)")
            plt.xlabel("时间步")
            plt.ylabel("丢包率")
            plt.ylim(-0.1, 1.1)  # 确保0和1都能显示
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 标记非零值位置
            non_zero_indices = np.where(generated_loss_time > 0)[0]
            if len(non_zero_indices) > 0:
                plt.scatter(non_zero_indices, generated_loss_time[non_zero_indices], 
                          color='red', s=50, alpha=0.7, label="非零丢包率")
                plt.legend()
                print(f"样本 {sample_idx+1} 非零丢包率位置: {non_zero_indices}")
                
                # 在非零值位置添加注释
                for idx in non_zero_indices:
                    plt.annotate(f"{idx}", xy=(idx, generated_loss_time[idx]), 
                               xytext=(idx, generated_loss_time[idx]+0.1),
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        time_series_path = output_path / "uplink_loss_time_series.png"
        plt.savefig(time_series_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Uplink loss time series saved as {time_series_path}")
    
    # 保存所有样本的上行丢包率数据到文本文件，方便分析
    with open(output_path / "uplink_loss_data.txt", "w") as f:
        for sample_idx in range(generated_samples.shape[0]):
            f.write(f"样本 {sample_idx+1} 上行丢包率数据:\n")
            loss_data = generated_samples[sample_idx, :, 3]
            for step, loss in enumerate(loss_data):
                f.write(f"{step}: {loss}\n")
            f.write("\n")
        print(f"Uplink loss data saved as {output_path / 'uplink_loss_data.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Full training and evaluation pipeline for conditional diffusion model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--num-windows-per-sample", type=int, default=10, help="Number of windows per sample for long sequence generation")
    parser.add_argument("--output-dir", type=str, default="./output/visualization", help="Output directory")
    parser.add_argument("--guidance-scale", type=float, default=1.0, 
                        help="条件增强比例，用于放大p99分位数，增强尖峰信号")

    args = parser.parse_args()

    # 获取数据文件路径
    train_file = os.path.join(args.data_dir, "datasets", "train.jsonl")
    val_file = os.path.join(args.data_dir, "datasets", "val.jsonl")
    test_file = os.path.join(args.data_dir, "datasets", "test.jsonl")
    assets_dir = os.path.join(args.data_dir, "assets")

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 训练模型
    print("Starting model training...")
    model, scheduler_config_path = train_model(
        train_file=train_file,
        val_file=val_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    # 可视化训练损失
    print("Visualizing training losses...")
    losses_file = os.path.join(args.output_dir, "train_losses.npy")
    visualize_training_losses(losses_file, args.output_dir)

    # 采样和后处理
    print("Sampling and post-processing...")
    sample_and_postprocess(
        model, scheduler_config_path, assets_dir,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_windows_per_sample=args.num_windows_per_sample,
        output_dir=args.output_dir,
        device=device,
        test_file=test_file,  # 优先使用测试文件路径，用于生成真实的条件向量
        train_file=train_file,  # 仅作为备选
    )

    # 可视化生成的样本
    print("Visualizing generated samples...")
    generated_samples_file = os.path.join(args.output_dir, "generated_samples.npy")
    visualize_generated_samples(
        generated_samples_file, test_file,  # 使用测试文件进行可视化对比
        output_dir=args.output_dir,
    )
    
    # 生成参考样本与生成样本的1000点综合对比图
    print("\n生成参考样本与生成样本的1000点综合对比图...")
    # 导入生成综合对比图的函数
    from src.visualization.generate_reference_generated_combined import (
        generate_combined_comprehensive,
        load_generated_1000_points,
        load_reference_1000_points
    )
    
    # 加载反归一化所需的QuantileTransformer
    import joblib
    qt_up = joblib.load(Path(assets_dir) / "qt_up.pkl")
    qt_down = joblib.load(Path(assets_dir) / "qt_down.pkl")
    
    # 加载参考样本的1000个点
    reference_points = load_reference_1000_points(test_file, qt_up, qt_down)
    print(f"参考样本数据点数量: {reference_points.shape[0]}")
    
    # 加载生成样本的1000个点
    generated_points = load_generated_1000_points(generated_samples_file)
    print(f"生成样本数据点数量: {generated_points.shape[0]}")
    
    # 生成综合对比图
    output_path = Path(args.output_dir) / "reference_generated_1000_points_comprehensive.png"
    generate_combined_comprehensive(reference_points, generated_points, output_path)
    
    # 生成时延对比图
    from src.visualization.generate_reference_generated_combined import generate_combined_1000_points
    output_path_latency = Path(args.output_dir) / "reference_generated_1000_points_latency.png"
    generate_combined_1000_points(reference_points, generated_points, output_path_latency)
    
    # 分析测试集数据统计信息
    print("\n=== 测试集数据统计信息 ===")
    analyze_test_set_stats(test_file)
    
    print("Full pipeline completed!")

def analyze_test_set_stats(test_file):
    """分析测试集数据统计信息
    
    Args:
        test_file: 测试数据文件路径
    """
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    total_samples = 0
    valid_samples = 0
    network_states = {}
    non_zero_loss_count = 0
    total_loss_count = 0
    window_lengths = []
    
    with open(test_file) as f:
        for i, line in enumerate(f):
            total_samples += 1
            data = json.loads(line)
            
            if data.get("keep", True):
                valid_samples += 1
                
                # 统计网络状态
                if "cond" in data and len(data["cond"]) > 11:
                    # 网络状态ID在条件向量的第11位
                    network_state = int(data["cond"][11])
                    network_states[network_state] = network_states.get(network_state, 0) + 1
                
                # 统计窗口数据
                if "window" in data:
                    window_data = data["window"]
                    window_lengths.append(len(window_data))
                    
                    # 统计丢包率
                    for row in window_data:
                        total_loss_count += 1
                        if row["loss_up"] > 0 or row["loss_dn"] > 0:
                            non_zero_loss_count += 1
    
    # 计算统计结果
    avg_window_length = np.mean(window_lengths) if window_lengths else 0
    std_window_length = np.std(window_lengths) if window_lengths else 0
    non_zero_loss_ratio = non_zero_loss_count / total_loss_count if total_loss_count > 0 else 0
    
    # 打印统计信息
    print(f"测试集总样本数: {total_samples}")
    print(f"有效样本数 (keep=True): {valid_samples}")
    print(f"无效样本数 (keep=False): {total_samples - valid_samples}")
    print(f"有效样本比例: {valid_samples / total_samples:.2%}")
    
    print("\n网络状态分布:")
    for state, count in sorted(network_states.items()):
        print(f"  状态 {state}: {count} 样本 ({count / valid_samples:.2%})")
    
    print("\n窗口长度统计:")
    print(f"  平均窗口长度: {avg_window_length:.2f}")
    print(f"  窗口长度标准差: {std_window_length:.2f}")
    print(f"  最小窗口长度: {min(window_lengths) if window_lengths else 0}")
    print(f"  最大窗口长度: {max(window_lengths) if window_lengths else 0}")
    
    print("\n丢包率统计:")
    print(f"  总丢包数据点: {total_loss_count}")
    print(f"  非零丢包数据点: {non_zero_loss_count}")
    print(f"  非零丢包比例: {non_zero_loss_ratio:.2%}")
    
    print("\n条件向量结构:")
    print("  条件向量维度: 23")
    print("  - 全局特征: 13个 (0-12)")
    print("  - 网络状态ID: 1个 (11)")
    print("  - 局部特征: 10个 (13-22)")


if __name__ == "__main__":
    main()
