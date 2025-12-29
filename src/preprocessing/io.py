import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


class DataSaver:
    """数据保存器，负责保存预处理结果和资源"""
    
    def __init__(self, out_dir: Path, dtype: np.dtype = np.float32):
        """初始化数据保存器
        
        Args:
            out_dir: 输出目录
            dtype: 输出数据类型
        """
        self.out_dir = out_dir
        self.dtype = dtype
        self.assets_dir = out_dir / "assets"
        self.meta_dir = out_dir / "meta"
        self.datasets_dir = out_dir / "datasets"
        
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def save_assets(self, assets: Dict[str, Any]):
        """保存预处理资源
        
        Args:
            assets: 预处理资源字典
        """
        # 保存QuantileTransformer模型
        if "qt_up" in assets and assets["qt_up"] is not None:
            with open(self.assets_dir / "qt_up.pkl", "wb") as f:
                pickle.dump(assets["qt_up"], f)
        
        if "qt_down" in assets and assets["qt_down"] is not None:
            with open(self.assets_dir / "qt_down.pkl", "wb") as f:
                pickle.dump(assets["qt_down"], f)
        
        # 保存条件向量标准化参数
        if "cond_mean" in assets:
            np.save(self.meta_dir / "cond_mean.npy", assets["cond_mean"].astype(self.dtype))
        
        if "cond_std" in assets:
            np.save(self.meta_dir / "cond_std.npy", assets["cond_std"].astype(self.dtype))
        
        # 保存初始局部条件向量
        if "init_local_cond" in assets:
            np.save(self.assets_dir / "init_local_cond.npy", assets["init_local_cond"].astype(self.dtype))
        
        # 保存部分丢包均值（已废弃）
        if "mean_loss_cat2_up" in assets:
            np.save(self.assets_dir / "mean_loss_cat2_up.npy", 
                    np.array(assets["mean_loss_cat2_up"]).astype(self.dtype))
        
        if "mean_loss_cat2_dn" in assets:
            np.save(self.assets_dir / "mean_loss_cat2_dn.npy", 
                    np.array(assets["mean_loss_cat2_dn"]).astype(self.dtype))
        
        # 保存z-score均值和标准差
        if "delay_up_mean" in assets:
            np.save(self.assets_dir / "delay_up_mean.npy", 
                    np.array(assets["delay_up_mean"]).astype(self.dtype))
        
        if "delay_up_std" in assets:
            np.save(self.assets_dir / "delay_up_std.npy", 
                    np.array(assets["delay_up_std"].astype(self.dtype)))
        
        if "delay_down_mean" in assets:
            np.save(self.assets_dir / "delay_down_mean.npy", 
                    np.array(assets["delay_down_mean"]).astype(self.dtype))
        
        if "delay_down_std" in assets:
            np.save(self.assets_dir / "delay_down_std.npy", 
                    np.array(assets["delay_down_std"].astype(self.dtype)))
    
    def save_dataset(self, datasets: Dict[str, List[Dict]]):
        """保存数据集
        
        Args:
            datasets: 按数据集类型划分的归一化窗口元数据
        """
        for dataset_name, dataset in datasets.items():
            filepath = self.datasets_dir / f"{dataset_name}.jsonl"
            with open(filepath, "w") as f:
                for meta in dataset:
                    if meta["keep"]:
                        # 构造要保存的数据
                        record = {
                            "trace_id": meta["trace_id"],
                            "start_time": float(meta["start_time"]),
                            "window": meta["window"].to_dict("records"),
                            "cond": meta["cond"].tolist(),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_metadata(self, pipeline_version: str, schema: Dict[str, Any], stats: Dict[str, Any]):
        """保存元数据
        
        Args:
            pipeline_version: 流水线版本
            schema: 数据模式
            stats: 统计信息
        """
        # 保存流水线版本
        with open(self.meta_dir / "pipeline_version.txt", "w") as f:
            f.write(pipeline_version)
        
        # 保存数据模式
        with open(self.meta_dir / "schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        
        # 保存统计信息
        if "state_id_stats" in stats:
            # 转换state_id_stats中的numpy类型为Python原生类型
            converted_stats = {}
            for dataset_name, stats_item in stats["state_id_stats"].items():
                converted_stats[dataset_name] = {}
                for state_id, stat in stats_item.items():
                    converted_stats[dataset_name][int(state_id)] = {
                        "count": int(stat["count"]),
                        "percentage": float(stat["percentage"])
                    }
            
            clustering_info = {
                "cluster_features": "6D",
                "optimal_k": 6,
                "state_id_stats": converted_stats
            }
            
            with open(self.meta_dir / "clustering_info.json", "w", encoding="utf-8") as f:
                json.dump(clustering_info, f, ensure_ascii=False, indent=2)
    
    def save_failed_files(self, failed_files: List[str]):
        """保存失败文件列表
        
        Args:
            failed_files: 失败文件列表
        """
        if failed_files:
            with open(self.meta_dir / "failed_files.txt", "w") as f:
                f.write("\n".join(failed_files))
    
    def save_empty_files(self, empty_files: List[str]):
        """保存空文件列表
        
        Args:
            empty_files: 空文件列表
        """
        if empty_files:
            with open(self.meta_dir / "empty_files.txt", "w") as f:
                f.write("\n".join(empty_files))


class ReportGenerator:
    """报告生成器，负责生成预处理报告"""
    
    @staticmethod
    def generate_report(stats: dict, config: dict, template_path: Path, out_path: Path):
        """生成数据报告
        
        Args:
            stats: 统计信息
            config: 配置信息
            template_path: 模板路径
            out_path: 输出路径
        """
        try:
            from jinja2 import Template
            with open(template_path) as f:
                template_str = f.read()
            template = Template(template_str)
            report_content = template.render(stats=stats, config=config)
        except Exception:
            # 如果模板加载失败，使用简单格式
            report_content = f"""# 数据处理报告

## 概述

- 处理时间: {stats.get('timestamp', 'N/A')}
- 总窗口数: {stats.get('total_windows', 0)}
- 训练集窗口数: {stats.get('train_count', 0)}
- 验证集窗口数: {stats.get('val_count', 0)}
- 测试集窗口数: {stats.get('test_count', 0)}

## 文件统计

- 处理文件总数: {stats.get('processed_files', 0)}
- 失败文件数: {stats.get('failed_files', 0)}
- 空文件数: {stats.get('empty_files', 0)}

## 配置信息

- 管道版本: {config.get('pipeline_version', 'N/A')}
- 窗口大小: {config.get('window_size', 'N/A')}
- 步长: {config.get('step_size', 'N/A')}
- 最大延迟(ms): {config.get('max_delay_ms', 'N/A')}
- 最大间隙(秒): {config.get('max_gap_sec', 'N/A')}
"""

        with open(out_path, "w") as f:
            f.write(report_content)
