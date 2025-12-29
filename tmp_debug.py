import numpy as np
import pandas as pd
from src.preprocessing.processing import fit_and_normalize, compute_window_features, prepare_norm_params
from src.preprocessing.features import normalize_condition_vector
from src.preprocessing.pipeline import PreprocessingPipeline

# 简单的调试脚本，用于检查归一化流程

# 创建示例数据
def create_sample_data():
    # 创建一些示例数据
    timestamps = np.arange(0, 100, 0.1)
    delay_up = np.random.normal(50, 10, len(timestamps))
    delay_down = np.random.normal(60, 15, len(timestamps))
    loss_up = np.random.choice([0, 1], len(timestamps), p=[0.95, 0.05])
    loss_down = np.random.choice([0, 1], len(timestamps), p=[0.9, 0.1])
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "delay_up_origin": delay_up,
        "delay_down_origin": delay_down,
        "loss_up_origin": loss_up,
        "loss_down_origin": loss_down,
        "delay_up": delay_up,  # 初始值与原始值相同
        "delay_down": delay_down,
        "loss_up": loss_up,
        "loss_down": loss_down,
    })
    
    return df

# 测试归一化流程
def test_normalization():
    print("创建示例数据...")
    df = create_sample_data()
    
    # 模拟窗口数据
    window_df = df[:100].copy()
    
    # 创建训练集元数据
    train_meta = {
        "window": window_df,
        "is_first": True,
        "start_time": df["timestamp"][0],
        "trace_id": "test_trace_1"
    }
    
    train = [train_meta]
    val = []
    test = []
    
    print("\n开始归一化流程...")
    renamed_datasets, assets = fit_and_normalize(train, val, test)
    
    print("\n归一化前后对比：")
    print(f"原始delay_up均值: {np.mean(window_df['delay_up_origin'])}")
    print(f"原始delay_down均值: {np.mean(window_df['delay_down_origin'])}")
    
    normalized_window = renamed_datasets["train"][0]["window"]
    print(f"归一化后delay_up均值: {np.mean(normalized_window['delay_up'])}")
    print(f"归一化后delay_down均值: {np.mean(normalized_window['delay_down'])}")
    
    print("\nZ-score参数：")
    print(f"delay_up_mean: {assets['delay_up_mean']}")
    print(f"delay_up_std: {assets['delay_up_std']}")
    print(f"delay_down_mean: {assets['delay_down_mean']}")
    print(f"delay_down_std: {assets['delay_down_std']}")
    
    # 检查delay_qt列
    print("\n检查delay_qt列：")
    print(f"delay_up_qt均值: {np.mean(normalized_window['delay_up_qt'])}")
    print(f"delay_down_qt均值: {np.mean(normalized_window['delay_down_qt'])}")
    
    # 计算条件特征
    print("\n计算条件特征...")
    global_features = compute_window_features(normalized_window)
    print(f"条件特征前5个值: {global_features[:5]}")
    
    # 准备归一化参数
    cond_mean, cond_std = prepare_norm_params(renamed_datasets["train"])
    print(f"\n条件特征归一化参数：")
    print(f"cond_mean前5个值: {cond_mean[:5]}")
    print(f"cond_std前5个值: {cond_std[:5]}")
    
    # 归一化条件特征
    normalized_cond = normalize_condition_vector(global_features, cond_mean, cond_std)
    print(f"\n归一化后的条件特征前5个值: {normalized_cond[:5]}")

if __name__ == "__main__":
    test_normalization()