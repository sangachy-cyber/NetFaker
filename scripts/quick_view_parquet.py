#!/usr/bin/env python3
"""快速查看 Parquet 文件的交互式脚本。
"""

import pandas as pd

# 读取文件
df = pd.read_parquet('data/processed/20260118_231050_VAj-playback.parquet')

# 查看基本信息
print("基本信息:")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(f"列名: {list(df.columns)}")

# 查看前几行
print("\n前 5 行:")
print(df.head())

# 查看统计信息
print("\n统计信息:")
print(df.describe())