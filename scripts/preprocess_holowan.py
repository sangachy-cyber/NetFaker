#!/usr/bin/env python3
"""HoloWAN 数据预处理主脚本。

执行完整的 HoloWAN 日志预处理流程，包括：
1. 加载原始 .txt 文件
2. 重命名列（添加 _up/_down 后缀）
3. 延迟归一化（log 变换 + 标准化）
4. 丢包率缩放（转换为 [0, 1] 范围）
5. 带宽保持不变
6. 保存处理后的数据和 scaler

使用示例：
    # 使用默认参数（共享延迟 scaler）
    python scripts/preprocess_holowan.py
    
    # 使用独立延迟 scaler
    python scripts/preprocess_holowan.py --shared-delay-scaler=False
    
    # 指定自定义数据目录
    python scripts/preprocess_holowan.py --raw-data-dir=data/raw/ --processed-data-dir=data/processed/
"""

import argparse
import os

from netfaker.simcore.io.holowan_loader import HoloWANLoader
from netfaker.simcore.io.holowan_saver import HoloWANWriter
from netfaker.simcore.preprocessing.holowan_preprocessor import HoloWANPreprocessor


def main():
    """主函数。
    
    解析命令行参数，执行预处理流程，保存结果。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="HoloWAN 数据预处理脚本")
    parser.add_argument(
        "--shared-delay-scaler",
        action="store_true",
        default=True,
        help="使用共享的全局延迟 scaler (默认: True)"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw/",
        help="原始数据目录 (默认: data/raw/)"
    )
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default="data/processed/",
        help="处理后数据目录 (默认: data/processed/)"
    )

    args = parser.parse_args()

    # 确保数据目录存在
    os.makedirs(args.raw_data_dir, exist_ok=True)
    os.makedirs(args.processed_data_dir, exist_ok=True)

    print("=== HoloWAN 数据预处理开始 ===")
    print(f"原始数据目录: {args.raw_data_dir}")
    print(f"处理后数据目录: {args.processed_data_dir}")
    print(f"使用共享延迟 scaler: {args.shared_delay_scaler}")
    print()

    # 1. 加载原始数据
    print("步骤 1: 加载原始数据...")
    loader = HoloWANLoader(raw_data_dir=args.raw_data_dir)
    raw_data = loader.load_all_files()

    if not raw_data:
        print("❌ 未找到可处理的文件")
        return

    print(f"✅ 成功加载 {len(raw_data)} 个文件")
    print()

    # 2. 预处理数据
    print("步骤 2: 预处理数据...")
    preprocessor = HoloWANPreprocessor(shared_delay_scaler=args.shared_delay_scaler)
    processed_data = preprocessor.fit_transform(raw_data)
    print("✅ 数据预处理完成")
    print()

    # 3. 保存处理后的数据
    print("步骤 3: 保存处理后的数据...")
    writer = HoloWANWriter(processed_data_dir=args.processed_data_dir)
    writer.save_all_data(processed_data)
    print()

    # 4. 保存 scaler
    print("步骤 4: 保存全局延迟 scaler...")
    scaler_info = preprocessor.get_scaler_info()
    writer.save_scaler(scaler_info)
    print()

    print("=== HoloWAN 数据预处理完成 ===")
    print(f"处理后文件数: {len(processed_data)}")
    print(f"输出目录: {args.processed_data_dir}")
    print("✅ 所有步骤已完成")


if __name__ == "__main__":
    main()
