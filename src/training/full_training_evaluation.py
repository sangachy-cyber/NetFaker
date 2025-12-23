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
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
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


def sample_and_postprocess(model, scheduler_config_path, assets_dir, num_samples=10, output_dir="./output/visualization", device="cpu", train_file=None):
    """采样并后处理生成的数据"""
    with open(scheduler_config_path) as f:
        scheduler_config = json.load(f)

    sampling_scheduler = DDPMScheduler.from_config(scheduler_config)
    sampler = DiffusionSampler(model, sampling_scheduler, device)

    import joblib
    qt_up = joblib.load(Path(assets_dir) / "qt_up.pkl")
    qt_dn = joblib.load(Path(assets_dir) / "qt_down.pkl")

    postprocessor = PostProcessor(qt_up, qt_dn)

    # 加载条件向量标准化参数
    cond_mean = np.load(Path(assets_dir) / "../meta" / "cond_mean.npy")
    cond_std = np.load(Path(assets_dir) / "../meta" / "cond_std.npy")

    # 生成条件向量
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

    # 生成一些条件向量用于采样
    cond_vectors = []
    for i in range(num_samples):
        cond = generate_valid_cond_vector(train_file)
        cond_vectors.append(cond)

    cond_batch = torch.stack(cond_vectors)

    # 生成样本
    print("Generating samples...")
    generated_samples = sampler.sample(cond_batch, num_inference_steps=1000)
    print(f"Generated samples with shape: {generated_samples.shape}")

    # 后处理生成的样本
    print("Post-processing samples...")
    processed_samples = postprocessor.postprocess(generated_samples)
    print(f"Processed samples with shape: {processed_samples.shape}")

    # 保存生成的样本
    output_path = Path(output_dir)
    samples_path = output_path / "generated_samples.npy"
    np.save(samples_path, processed_samples)
    print(f"Generated samples saved as {samples_path}")

    return processed_samples


def visualize_generated_samples(generated_samples_file, real_data_file, output_dir="./output/visualization"):
    """可视化生成的样本与真实数据的对比"""
    # 加载生成的样本
    generated_samples = np.load(generated_samples_file)

    # 加载真实数据样本
    real_samples = []
    with open(real_data_file) as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只加载前100个样本用于比较
                break
            data = json.loads(line)
            if data.get("keep", True):
                window_data = data["window"]
                # 提取特征
                features = []
                for row in window_data[:100]:  # 只取前100个时间步
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

    # 绘制延迟数据的直方图对比
    plt.figure(figsize=(15, 10))

    # 上行延迟
    plt.subplot(2, 3, 1)
    real_del_up = real_samples[:, :, 0].flatten()
    generated_del_up = generated_samples[:, :, 1].flatten()  # 注意索引对应关系
    plt.hist(real_del_up, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_up, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("Uplink Delay Distribution")
    plt.xlabel("Delay (normalized)")
    plt.ylabel("Density")
    plt.legend()

    # 下行延迟
    plt.subplot(2, 3, 2)
    real_del_dn = real_samples[:, :, 1].flatten()
    generated_del_dn = generated_samples[:, :, 2].flatten()
    plt.hist(real_del_dn, bins=50, alpha=0.7, label="Real Data", density=True)
    plt.hist(generated_del_dn, bins=50, alpha=0.7, label="Generated Data", density=True)
    plt.title("Downlink Delay Distribution")
    plt.xlabel("Delay (normalized)")
    plt.ylabel("Density")
    plt.legend()

    # 上行丢包
    plt.subplot(2, 3, 3)
    real_loss_up = real_samples[:, :, 2].flatten()
    generated_loss_up = generated_samples[:, :, 3].flatten()
    plt.hist(real_loss_up, bins=3, alpha=0.7, label="Real Data", density=True, align="left")
    plt.hist(generated_loss_up, bins=3, alpha=0.7, label="Generated Data", density=True, align="left")
    plt.title("Uplink Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.legend()

    # 下行丢包
    plt.subplot(2, 3, 4)
    real_loss_dn = real_samples[:, :, 3].flatten()
    generated_loss_dn = generated_samples[:, :, 4].flatten()
    plt.hist(real_loss_dn, bins=3, alpha=0.7, label="Real Data", density=True, align="left")
    plt.hist(generated_loss_dn, bins=3, alpha=0.7, label="Generated Data", density=True, align="left")
    plt.title("Downlink Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.legend()

    # Q-Q图：上行延迟
    plt.subplot(2, 3, 5)
    stats.probplot(generated_del_up, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Uplink Delay")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    # Q-Q图：下行延迟
    plt.subplot(2, 3, 6)
    stats.probplot(generated_del_dn, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Downlink Delay")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    plt.tight_layout()

    # 保存图像
    comparison_path = output_path / "data_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data comparison visualization saved as {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Full training and evaluation pipeline for conditional diffusion model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="./output/visualization", help="Output directory")

    args = parser.parse_args()

    # 获取数据文件路径
    train_file = os.path.join(args.data_dir, "datasets", "train.jsonl")
    val_file = os.path.join(args.data_dir, "datasets", "val.jsonl")
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
        output_dir=args.output_dir,
        device=device,
        train_file=train_file,  # 传递训练文件路径，用于生成真实的条件向量
    )

    # 可视化生成的样本
    print("Visualizing generated samples...")
    generated_samples_file = os.path.join(args.output_dir, "generated_samples.npy")
    visualize_generated_samples(
        generated_samples_file, train_file,
        output_dir=args.output_dir,
    )

    print("Full pipeline completed!")


if __name__ == "__main__":
    main()
