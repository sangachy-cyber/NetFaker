import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# ======================
# 配置区（按需修改路径）
# ======================
RAW_DATA_DIR = "data"  # 原始 HoloWAN .txt 文件目录
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_holowan_file(filepath):
    """
    解析单个 HoloWAN 记录文件。
    跳过元信息，从 '---' 后读取数据行。
    """
    data_lines = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reading_data = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("---"):
                reading_data = True
                continue
            if not reading_data:
                continue

            parts = stripped.split(",")
            if len(parts) < 6:
                continue
            try:
                # 提取: Delay1, Loss1, BW1, Delay2, Loss2, BW2
                d1 = float(parts[0])
                l1 = float(parts[1])
                bw1 = float(parts[2])
                d2 = float(parts[3])
                l2 = float(parts[4])
                bw2 = float(parts[5])
                data_lines.append([d1, l1, bw1, d2, l2, bw2])
            except (ValueError, IndexError):
                continue  # 跳过格式错误行
    return np.array(data_lines) if data_lines else None


def main():
    print("🧹 开始预处理原始 HoloWAN 数据...")
    all_points = []  # 存储所有有效点: [log(Delay1+1), p1, log(Delay2+1), p2]

    # 遍历所有 .txt 文件
    txt_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"在 '{RAW_DATA_DIR}' 中未找到 .txt 文件！")

    for filename in sorted(txt_files):
        filepath = os.path.join(RAW_DATA_DIR, filename)
        print(f"  → 处理 {filename}")

        raw_data = parse_holowan_file(filepath)
        if raw_data is None or raw_data.size == 0:
            print(f"    ⚠️ 无有效数据，跳过")
            continue

        d1, l1, bw1, d2, l2, bw2 = raw_data.T

        # 🔧 纠错：带宽为0时，丢包率强制为100%
        l1 = np.where(bw1 == 0, 100.0, l1)
        l2 = np.where(bw2 == 0, 100.0, l2)

        # 📉 转换丢包率为 [0, 1]
        p1 = np.clip(l1 / 100.0, 0.0, 1.0)
        p2 = np.clip(l2 / 100.0, 0.0, 1.0)

        # 🧹 清洗：剔除极端延迟（>2000ms）
        valid_mask = (d1 <= 2000) & (d2 <= 2000)
        if not np.any(valid_mask):
            print(f"    ⚠️ 所有延迟 >2000ms，跳过")
            continue

        d1, p1, d2, p2 = d1[valid_mask], p1[valid_mask], d2[valid_mask], p2[valid_mask]

        # 📐 对延迟做 log 变换（稳定长尾分布）
        d1_log = np.log(d1 + 1.0)
        d2_log = np.log(d2 + 1.0)

        # 拼接为 (N, 4) 矩阵
        points = np.stack([d1_log, p1, d2_log, p2], axis=1)
        all_points.append(points)

    if not all_points:
        raise ValueError("未收集到任何有效数据点！")

    # 合并所有文件的数据
    clean_data = np.concatenate(all_points, axis=0)
    print(f"\n✅ 清洗完成！共 {len(clean_data):,} 个有效采样点")

    # 📏 拟合全局 StandardScaler（合并 UL+DL 的 log Delay）
    all_delays = np.concatenate([clean_data[:, 0], clean_data[:, 2]]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(all_delays)

    # 💾 保存结果
    np.save(os.path.join(OUTPUT_DIR, "clean_data.npy"), clean_data)
    with open(os.path.join(OUTPUT_DIR, "scaler_delay.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n📁 输出文件:")
    print(f"   • {OUTPUT_DIR}/clean_data.npy      (shape: {clean_data.shape})")
    print(f"   • {OUTPUT_DIR}/scaler_delay.pkl    (StandardScaler for log-Delay)")
    print("\n✨ 预处理完成！下一步：运行 build_train_dataset.py")


if __name__ == "__main__":
    main()
