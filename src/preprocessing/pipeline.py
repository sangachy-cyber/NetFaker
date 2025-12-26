from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

from .data import PreprocessingConfig, PreprocessingResult, WindowMetaNormed
from .processing import (
    parse_txt, clean_and_truncate, split_and_resample, extract_windows,
    mark_first_window, group_by_trace_id, assign_split_by_trace,
    fit_and_normalize, prepare_init_local_cond, prepare_norm_params
)
from .features import compute_window_features, compute_local_features, merge_features, normalize_condition_vector
from .io import DataSaver, ReportGenerator


class PreprocessingPipeline:
    """预处理流水线，负责协调各个预处理阶段的执行"""
    
    def __init__(self, config: PreprocessingConfig):
        """初始化预处理流水线
        
        Args:
            config: 预处理配置
        """
        self.config = config
        self.pipeline_version = "v1.3"
    
    def run(self) -> PreprocessingResult:
        """执行完整的预处理流程
        
        Returns:
            预处理结果
        """
        # 1. 初始化统计信息
        stats = {
            "total_windows": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "processed_files": 0,
            "failed_files": 0,
            "empty_files": 0,
        }
        
        # 2. 加载配置和文件
        input_dir = Path(self.config["input_dir"])
        out_dir = Path(self.config["output_dir"])
        txt_files = list(input_dir.glob("*.txt"))
        
        # 3. 构建 network_state_map
        network_state_map = self._build_network_state_map(txt_files)
        
        # 4. 处理所有文件
        all_windows_meta = []
        failed_files = []
        empty_files = []
        
        for txt_file in tqdm(txt_files, desc="Processing files"):
            stats["processed_files"] += 1
            try:
                # 阶段1-5：解析、清洗、分割、提取窗口、标记首窗
                file_result = self._process_single_file(txt_file)
                if file_result:
                    all_windows_meta.extend(file_result)
                else:
                    empty_files.append(str(txt_file))
                    stats["empty_files"] += 1
            except Exception as e:
                failed_files.append(f"{txt_file}: {e!s}")
                stats["failed_files"] += 1
                continue
        
        # 5. 阶段6-7：分组和数据集划分
        traces_dict = group_by_trace_id(all_windows_meta)
        train, val, test = assign_split_by_trace(
            traces_dict,
            train_ratio=self.config["split"]["train"],
            val_ratio=self.config["split"]["val"],
            random_state=self.config["split"]["random_state"],
        )
        
        # 6. 阶段8：归一化和列重命名
        renamed_datasets, assets = fit_and_normalize(
            train, val, test, 
            random_state=self.config["quantile_transformer"]["random_state"]
        )
        
        # 7. 阶段9：重新计算条件向量
        final_datasets, extra_assets = self._recompute_condition_vectors(
            renamed_datasets, assets, network_state_map
        )
        
        # 更新assets字典
        assets.update(extra_assets)
        
        # 8. 统计结果
        stats["train_count"] = sum(1 for w in final_datasets["train"] if w["keep"])
        stats["val_count"] = sum(1 for w in final_datasets["val"] if w["keep"])
        stats["test_count"] = sum(1 for w in final_datasets["test"] if w["keep"])
        stats["total_windows"] = stats["train_count"] + stats["val_count"] + stats["test_count"]
        
        # 9. 阶段10：保存结果
        saver = DataSaver(out_dir, dtype=getattr(np, self.config["output_dtype"]))
        saver.save_assets(assets)
        saver.save_dataset(final_datasets)
        saver.save_failed_files(failed_files)
        saver.save_empty_files(empty_files)
        
        # 10. 生成schema
        schema = self._generate_schema()
        saver.save_metadata(self.pipeline_version, schema, assets)
        
        # 11. 阶段11：生成报告
        report_generator = ReportGenerator()
        report_generator.generate_report(
            stats, self.config, 
            Path(self.config["report_template"]), 
            out_dir / "data_report.md"
        )
        
        # 返回结果
        return {
            "datasets": final_datasets,
            "assets": assets,
            "stats": stats
        }
    
    def _process_single_file(self, txt_file: Path) -> List[Dict]:
        """处理单个文件
        
        Args:
            txt_file: 输入txt文件
            
        Returns:
            窗口元数据列表，若文件为空则返回空列表
        """
        # 阶段1：解析原始txt文件
        df_raw = parse_txt(txt_file, self.config["raw_interval_sec"])
        
        # 阶段2：清洗并截断数据
        df_clean = clean_and_truncate(df_raw, self.config["max_delay_ms"])
        
        # 阶段3：分割并重新采样
        segments = split_and_resample(
            df_clean,
            max_gap_sec=self.config["max_gap_sec"],
            min_rows=self.config["min_segment_rows"],
            max_invalid_ratio=self.config.get("max_invalid_ratio", 0.01),
        )
        
        # 阶段4：提取窗口
        windows = extract_windows(segments, self.config["window_size"], self.config["step_size"])
        
        if not windows:
            return []
        
        # 阶段5：标记首个窗口
        trace_id = txt_file.stem
        window_metas = mark_first_window(windows, trace_id)
        
        return window_metas
    
    def _build_network_state_map(self, txt_files: List[Path]) -> Dict[str, int]:
        """构建网络状态映射
        
        Args:
            txt_files: 输入文件列表
            
        Returns:
            网络状态映射字典
        """
        import re
        pattern = self.config.get("filename_pattern", ".*_state(\d+)\.txt")
        default_id = self.config.get("default_network_state_id", 0)
        network_state_map = {}
        
        for txt_file in txt_files:
            match = re.search(pattern, txt_file.name)
            network_state_map[txt_file.stem] = int(match.group(1)) if match else default_id
        
        return network_state_map
    
    def _recompute_condition_vectors(
        self, datasets: Dict[str, List[Dict]], assets: Dict[str, Any], network_state_map: Dict[str, int]
    ) -> Tuple[Dict[str, List[WindowMetaNormed]], Dict[str, Any]]:
        """重新计算条件向量
        
        Args:
            datasets: 重命名后的数据集
            assets: 资源字典
            network_state_map: 网络状态映射
            
        Returns:
            (final_datasets, extra_assets) 二元组
        """
        # 准备初始局部条件向量和标准化参数
        train_dataset = datasets["train"]
        init_local_cond = prepare_init_local_cond(train_dataset)
        cond_mean, cond_std = prepare_norm_params(train_dataset)
        
        # 初始化extra_assets
        extra_assets = {
            "init_local_cond": init_local_cond,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
            "mean_loss_cat2_up": 0.5,
            "mean_loss_cat2_dn": 0.5,
            "state_id_stats": {},
        }
        
        # 处理所有数据集
        final_datasets = {}
        state_id_stats = {}
        
        for dataset_name, dataset in datasets.items():
            # 按 trace_id 分组并排序
            traces = group_by_trace_id(dataset)
            
            # 对每个trace内的窗口按 start_time 排序
            for trace_id, metas in traces.items():
                traces[trace_id] = sorted(metas, key=lambda x: x["start_time"])
            
            # 处理每个窗口
            normed_metas = []
            dataset_state_ids = []
            
            for trace_id, metas in traces.items():
                for i, meta in enumerate(metas):
                    normed_meta = self._process_single_window(
                        meta, i, metas, network_state_map, 
                        init_local_cond, cond_mean, cond_std
                    )
                    normed_metas.append(normed_meta)
                    
                    # 统计state_id分布（只统计keep=True的样本）
                    if normed_meta["keep"]:
                        dataset_state_ids.append(int(normed_meta["cond"][11]))
            
            final_datasets[dataset_name] = normed_metas
            
            # 统计当前数据集的state_id分布
            state_id_stats[dataset_name] = self._compute_state_id_stats(dataset_state_ids)
        
        extra_assets["state_id_stats"] = state_id_stats
        return final_datasets, extra_assets
    
    def _process_single_window(
        self, meta: Dict, i: int, metas: List[Dict], network_state_map: Dict[str, int],
        init_local_cond: np.ndarray, cond_mean: np.ndarray, cond_std: np.ndarray
    ) -> WindowMetaNormed:
        """处理单个窗口
        
        Args:
            meta: 窗口元数据
            i: 窗口索引
            metas: 同trace的所有窗口元数据
            network_state_map: 网络状态映射
            init_local_cond: 初始局部条件向量
            cond_mean: 条件向量均值
            cond_std: 条件向量标准差
            
        Returns:
            归一化后的窗口元数据
        """
        window_df = meta["window"].copy()  # 创建副本以避免修改原始数据
        
        # 保存原始丢包率
        # 上行丢包只保留0和1两个类别
        window_df.loc[window_df["loss_up"] > 0, "loss_up"] = 1.0
        # 下行丢包只保留0和1两个类别
        window_df.loc[window_df["loss_dn"] > 0, "loss_dn"] = 1.0
        
        # 计算全局特征
        global_features = compute_window_features(window_df)
        
        # 使用默认网络状态ID 0
        network_state_id = 0
        global_features[11] = float(network_state_id)
        
        # 计算局部特征
        if i == 0:  # 该 trace 在此数据集中的第一个窗口，视为逻辑首窗
            local_features = init_local_cond
        else:
            # 使用前一个窗口的最后5行
            prev_window_df = metas[i-1]["window"]
            local_features = compute_local_features(prev_window_df)
        
        # 合并特征形成23维条件向量
        cond_vector = merge_features(global_features, local_features)
        
        # Z-score 标准化（仅对22个float维度）
        normalized_cond_vector = normalize_condition_vector(cond_vector, cond_mean, cond_std)
        
        # 构造归一化后的窗口元数据
        normed_meta: WindowMetaNormed = {
            "window": window_df,
            "is_first": meta["is_first"],
            "start_time": meta["start_time"],
            "trace_id": meta["trace_id"],
            "cond": normalized_cond_vector,
            "keep": not meta["is_first"],
        }
        
        return normed_meta
    
    def _compute_state_id_stats(self, state_ids: List[int]) -> Dict[int, Dict[str, float]]:
        """计算状态ID统计信息
        
        Args:
            state_ids: 状态ID列表
            
        Returns:
            状态ID统计信息
        """
        if not state_ids:
            return {}
        
        from collections import Counter
        counter = Counter(state_ids)
        total = len(state_ids)
        stats = {}
        
        for state_id, count in counter.items():
            stats[state_id] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        
        return stats
    
    def _generate_schema(self) -> Dict[str, Any]:
        """生成数据模式
        
        Returns:
            数据模式
        """
        schema = {
            "pipeline_version": self.pipeline_version,
            "columns": ["timestamp", "del_up", "del_dn", "loss_up", "loss_dn"],
            "window_size": self.config["window_size"],
            "freq_hz": 10,
            "normalized": True,
            "condition_vector_dim": 23,
            "condition_vector_structure": {
                "global_features": list(range(13)),
                "local_features": list(range(13, 23)),
            },
            "network_state_id_source": "filename_regex_or_default",
            "network_state_id_encoding": "integer category stored as float (e.g., 2.0)",
            "level2_normalization": "Z-score normalization applied to 22 float dimensions (indices 0–10 and 12–22), excluding the integer-encoded network_state_id at index 11.",
            "split_strategy": "Entire traces are assigned to a single split to prevent data leakage.",
            "normalization": {
                "del_up/del_dn": "QuantileTransformer(output_distribution='normal'), fitted on train set",
                "loss_up/loss_dn": "Clipped to [0.0, 1.0], no transformation applied",
            },
        }
        return schema
