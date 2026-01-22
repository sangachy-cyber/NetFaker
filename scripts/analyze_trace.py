#!/usr/bin/env python3
"""HoloWAN 状态序列提取工具。

从真实HoloWAN仿真文件中自动提取结构化的状态序列（segments），
用于回放、分析或再生成。

Usage:
    python scripts/analyze_trace.py --input data/raw/real_trace.txt --output data/outputs/segments.json
"""

import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
from loguru import logger

# 直接导入需要的类，避免复杂的导入链
from netfaker.simcore.io.holowan_loader import HoloWANLoader
from netfaker.simcore.dataset.window_extractor import WindowExtractor
from netfaker.simcore.clustering.state_predictor import StatePredictor
from netfaker.simcore.synthesizer.state_sequence_compressor import StateSequenceCompressor


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="HoloWAN 状态序列提取工具")
    parser.add_argument("--input", required=True, help="输入HoloWAN文件路径")
    parser.add_argument("--output", required=True, help="输出segments文件路径")
    parser.add_argument("--model-path", default="data/clusters/gmm_model.joblib", 
                      help="聚类模型路径")
    parser.add_argument("--scaler-path", default="data/clusters/feature_scaler.joblib", 
                      help="特征缩放器路径")
    parser.add_argument("--metadata-path", default="data/clusters/state_metadata.json", 
                      help="状态元数据路径")
    return parser.parse_args()


def load_and_process_file(input_path: str) -> List[Dict[str, Any]]:
    """加载并处理HoloWAN文件。

    Args:
        input_path: 输入HoloWAN文件路径

    Returns:
        提取的窗口列表
    """
    # 加载HoloWAN文件
    loader = HoloWANLoader()
    df = loader.load_file(input_path)
    
    if df is None or df.empty:
        logger.error(f"文件 {input_path} 为空或加载失败")
        raise ValueError(f"文件 {input_path} 为空或加载失败")
    
    # 检查文件行数是否足够
    if len(df) < 100:
        logger.warning(f"文件 {input_path} 少于100行，无法提取有效窗口")
        return []
    
    # 提取窗口
    extractor = WindowExtractor(window_size=100, step_size=100)  # 非重叠窗口
    windows = extractor.extract_windows(df, os.path.basename(input_path))
    
    logger.info(f"从文件 {input_path} 提取了 {len(windows)} 个窗口")
    return windows


def predict_states(windows: List[Dict[str, Any]], predictor: StatePredictor) -> List[int]:
    """预测窗口的状态ID。

    Args:
        windows: 窗口列表
        predictor: 状态预测器实例

    Returns:
        每个窗口的状态ID列表
    """
    state_ids = []
    for i, window in enumerate(windows):
        try:
            state_id = predictor.predict_from_window(window)
            state_ids.append(state_id)
            logger.debug(f"窗口 {i+1} 预测状态为: s{state_id}")
        except Exception as e:
            logger.error(f"预测窗口 {i+1} 失败: {e}")
            # 跳过失败的窗口
            continue
    
    logger.info(f"成功预测了 {len(state_ids)} 个窗口的状态")
    return state_ids


def main() -> None:
    """主函数。
    """
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化状态预测器
        logger.info("初始化状态预测器...")
        predictor = StatePredictor(
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            metadata_path=args.metadata_path
        )
        
        # 初始化状态序列压缩器
        compressor = StateSequenceCompressor()
        
        # 加载并处理文件
        logger.info(f"加载并处理文件: {args.input}")
        windows = load_and_process_file(args.input)
        
        # 预测状态
        logger.info("预测窗口状态...")
        state_ids = predict_states(windows, predictor)
        
        # 压缩为segments
        logger.info("压缩状态序列...")
        segments = compressor.compress(state_ids)
        
        # 保存结果
        logger.info(f"保存结果到: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        logger.success(f"状态序列提取完成！共生成 {len(segments)} 个segments")
        logger.success(f"输出文件: {args.output}")
        
    except Exception as e:
        logger.exception(f"处理失败: {e}")
        raise


if __name__ == "__main__":
    main()