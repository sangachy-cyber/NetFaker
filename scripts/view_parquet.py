#!/usr/bin/env python3
"""Parquet 文件查看器。

快速查看 Parquet 文件的内容，包括前几行数据和基本统计信息。
"""

import argparse
import pandas as pd

def view_parquet(filepath):
    """查看 Parquet 文件内容。
    
    Args:
        filepath: Parquet 文件路径
    """
    print(f"=== 查看 Parquet 文件: {filepath} ===")
    
    # 读取文件
    df = pd.read_parquet(filepath)
    
    # 显示基本信息
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    print()
    
    # 显示前 10 行数据
    print("=== 前 10 行数据 ===")
    print(df.head(10))
    print()
    
    # 显示数值列的统计信息
    print("=== 数值列统计信息 ===")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("无数值列")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet 文件查看器")
    parser.add_argument("filepath", type=str, help="Parquet 文件路径")
    args = parser.parse_args()
    
    view_parquet(args.filepath)