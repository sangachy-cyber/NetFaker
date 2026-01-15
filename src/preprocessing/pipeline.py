from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import PreprocessingConfig, PreprocessingResult, WindowMetaNormed
from .processing import (
    parse_txt, clean_and_truncate, split_and_resample, extract_windows,
    mark_first_window, group_by_trace_id, assign_split_by_trace,
    fit_and_normalize
)
from .features import compute_window_features, normalize_condition_vector
from .io import DataSaver, ReportGenerator

# 导入行为发现模块
from behavior_discovery.pattern_identifier import PatternIdentifier


class PreprocessingPipeline:
    """预处理流水线，负责协调各个预处理阶段的执行"""
    
    def __init__(self, config: PreprocessingConfig):
        """初始化预处理流水线
        
        Args:
            config: 预处理配置
        """
        self.config = config
        self.pipeline_version = "v1.3"
        # 初始化行为发现模块
        self.pattern_identifier = PatternIdentifier()
    
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
        
        # 保存所有原始数据，用于行为发现
        all_raw_data = []
        
        for txt_file in tqdm(txt_files, desc="Processing files"):
            stats["processed_files"] += 1
            try:
                # 读取对应的配置文件
                conf_file = Path("data/conf") / f"{txt_file.name}"
                file_config = {}
                if conf_file.exists():
                    with open(conf_file, "r") as f:
                        for line in f:
                            if ":" in line:
                                key, value = line.strip().split(":", 1)
                                file_config[key] = value.strip()
                
                # 阶段1-5：解析、清洗、分割、提取窗口、标记首窗
                file_result, raw_data = self._process_single_file(txt_file, file_config)
                if file_result:
                    all_windows_meta.extend(file_result)
                    all_raw_data.append(raw_data)
                else:
                    empty_files.append(str(txt_file))
                    stats["empty_files"] += 1
            except Exception as e:
                failed_files.append(f"{txt_file}: {e!s}")
                stats["failed_files"] += 1
                continue
        
        # 5. 阶段6：计算行为ID（使用原始数据，不需要归一化）
        behavior_windows, behavior_assets = self._compute_behavior_ids(
            {"all": all_windows_meta}, {}, network_state_map, all_raw_data
        )
        
        # 6. 阶段7-8：分组和数据集划分（基于行为ID）
        traces_dict = group_by_trace_id(behavior_windows)
        train, val, test = assign_split_by_trace(
            traces_dict,
            train_ratio=self.config["split"]["train"],
            val_ratio=self.config["split"]["val"],
            random_state=self.config["split"]["random_state"],
        )
        
        # 7. 阶段9：归一化和列重命名
        renamed_datasets, assets = fit_and_normalize(
            train, val, test, 
            random_state=self.config["quantile_transformer"]["random_state"]
        )
        
        # 8. 阶段10：重新计算条件向量（使用已经计算好的行为ID）
        final_datasets, extra_assets = self._recompute_condition_vectors(
            renamed_datasets, assets, network_state_map, all_raw_data
        )
        
        # 更新assets字典
        assets.update(extra_assets)
        
        # 8. 统计结果
        stats["train_count"] = sum(1 for w in final_datasets["train"] if w["keep"])
        stats["val_count"] = sum(1 for w in final_datasets["val"] if w["keep"])
        stats["test_count"] = sum(1 for w in final_datasets["test"] if w["keep"])
        stats["total_windows"] = stats["train_count"] + stats["val_count"] + stats["test_count"]
        
        # 9. 优化训练数据提取：按行为类别筛选和重划分数据集
        final_datasets, other_datasets = self._filter_and_reorganize_datasets(final_datasets, out_dir)
        
        # 10. 更新统计结果
        stats["train_count"] = sum(1 for w in final_datasets["train"] if w["keep"])
        stats["val_count"] = sum(1 for w in final_datasets["val"] if w["keep"])
        stats["test_count"] = sum(1 for w in final_datasets["test"] if w["keep"])
        stats["total_windows"] = stats["train_count"] + stats["val_count"] + stats["test_count"]
        
        # 11. 阶段10：保存结果
        saver = DataSaver(out_dir, dtype=getattr(np, self.config["output_dtype"]))
        saver.save_assets(assets)
        saver.save_dataset(final_datasets)
        saver.save_failed_files(failed_files)
        saver.save_empty_files(empty_files)
        
        # 保存other类别数据
        self._save_other_category(other_datasets, out_dir)
        
        # 12. 生成schema
        schema = self._generate_schema()
        saver.save_metadata(self.pipeline_version, schema, assets)
        
        # 13. 阶段11：生成报告
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
    
    def _process_single_file(self, txt_file: Path, file_config: Dict[str, str]) -> Tuple[List[Dict], pd.DataFrame]:
        """处理单个文件
        
        Args:
            txt_file: 输入txt文件
            file_config: 文件对应的配置信息
            
        Returns:
            (窗口元数据列表, 原始数据DataFrame) 二元组，若文件为空则返回空列表和空DataFrame
        """
        # 阶段1：解析原始txt文件
        df_raw = parse_txt(txt_file, self.config["raw_interval_sec"], file_config)
        
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
            return [], df_raw
        
        # 阶段5：标记首个窗口
        trace_id = txt_file.stem
        window_metas = mark_first_window(windows, trace_id)
        
        return window_metas, df_raw
    
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
            # 如果匹配且有捕获组，则使用捕获组的值，否则使用默认ID
            network_state_map[txt_file.stem] = int(match.group(1)) if (match and len(match.groups()) > 0) else default_id
        
        return network_state_map
    
    def _compute_behavior_ids(
        self, datasets: Dict[str, List[Dict]], assets: Dict[str, Any], network_state_map: Dict[str, int], all_raw_data: List[pd.DataFrame]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """计算行为ID

        Args:
            datasets: 数据集字典
            assets: 资源字典（可以为空）
            network_state_map: 网络状态映射
            all_raw_data: 所有原始数据
            
        Returns:
            (behavior_windows, behavior_assets) 二元组
        """
        # 初始化behavior_assets，只包含必要的均值和标准差（后续会被实际值覆盖）
        behavior_assets = {
            "mean_loss_cat2_up": 0.5,
            "mean_loss_cat2_dn": 0.5,
        }
        
        # 收集所有窗口元数据
        all_windows = []
        for dataset in datasets.values():
            all_windows.extend(dataset)
        
        # 执行行为发现，计算每个窗口的行为ID
        behavior_labels = []
        if all_windows:
            try:
                labels_list = []
                
                for window_meta in all_windows:
                    window_df = window_meta["window"]
                    
                    if window_df.empty:
                        # 空窗口标记为INVALID
                        labels_list.append(8)
                        continue
                    
                    # 确保窗口数据包含必要的列
                    # 优先使用原始数据列
                    if all(col in window_df.columns for col in ["delay_up_origin", "loss_up_origin", "delay_down_origin", "loss_down_origin"]):
                        delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up_origin", "loss_up_origin", "delay_down_origin", "loss_down_origin"
                    elif all(col in window_df.columns for col in ["delay_up", "loss_up", "delay_down", "loss_down"]):
                        delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up", "loss_up", "delay_down", "loss_down"
                    elif all(col in window_df.columns for col in ["delay_up", "loss_up", "delay_down", "loss_dn"]):
                        delay_col1, loss_col1, delay_col2, loss_col2 = "delay_up", "loss_up", "delay_down", "loss_dn"
                    elif all(col in window_df.columns for col in ["delay1", "loss_rate1", "delay2", "loss_rate2"]):
                        delay_col1, loss_col1, delay_col2, loss_col2 = "delay1", "loss_rate1", "delay2", "loss_rate2"
                    else:
                        # 列不匹配，标记为INVALID
                        labels_list.append(8)
                        continue
                    
                    # 提取上下行数据
                    delay_up = window_df[delay_col1].values
                    loss_up = window_df[loss_col1].values
                    delay_down = window_df[delay_col2].values
                    loss_down = window_df[loss_col2].values
                    
                    # 计算动态阈值
                    thresholds = {
                        "delay_mean_low": self.pattern_identifier.DEFAULT_DELAY_MEAN_LOW,
                        "delay_mean_high": self.pattern_identifier.DEFAULT_DELAY_MEAN_HIGH,
                        "delay_std_high": self.pattern_identifier.DEFAULT_DELAY_STD_HIGH,
                        "loss_mean_low": self.pattern_identifier.DEFAULT_LOSS_MEAN_LOW,
                        "loss_mean_high": self.pattern_identifier.DEFAULT_LOSS_MEAN_HIGH,
                        "loss_std_high": self.pattern_identifier.DEFAULT_LOSS_STD_HIGH,
                    }
                    
                    # 对上下行数据分别检测行为
                    behavior_up = self.pattern_identifier._detect_behavior_with_raw_data(delay_up, loss_up, thresholds)
                    behavior_down = self.pattern_identifier._detect_behavior_with_raw_data(delay_down, loss_down, thresholds)
                    
                    # 合并上下行标签，取更严重的行为（数值更大表示更严重）
                    merged_label = max(behavior_up, behavior_down)
                    labels_list.append(merged_label)
                
                behavior_labels = labels_list
                
                # 计算行为统计信息
                import numpy as np
                behavior_assets["behavior_stats"] = self.pattern_identifier._calculate_behavior_statistics(np.array(behavior_labels))
                
                # 打印行为ID分布，用于调试
                from collections import Counter
                behavior_counts = Counter(behavior_labels)
                print(f"行为ID分布: {dict(behavior_counts)}")
                print(f"行为ID列表: {sorted(list(behavior_counts.keys()))}")
                
            except Exception as e:
                print(f"行为发现模块执行失败，使用默认标签: {e}")
                # 如果行为发现失败，使用默认标签
                behavior_labels = [0] * len(all_windows)
        
        # 为每个窗口添加行为ID
        behavior_windows = []
        for i, window_meta in enumerate(all_windows):
            if i < len(behavior_labels):
                window_with_behavior = window_meta.copy()
                window_with_behavior["behavior_id"] = behavior_labels[i]
                behavior_windows.append(window_with_behavior)
            else:
                # 如果没有行为标签，添加默认值
                window_with_behavior = window_meta.copy()
                window_with_behavior["behavior_id"] = 0
                behavior_windows.append(window_with_behavior)
        
        return behavior_windows, behavior_assets
    
    def _recompute_condition_vectors(
        self, datasets: Dict[str, List[Dict]], assets: Dict[str, Any], network_state_map: Dict[str, int], all_raw_data: List[pd.DataFrame]
    ) -> Tuple[Dict[str, List[WindowMetaNormed]], Dict[str, Any]]:
        """重新计算条件向量

        Args:
            datasets: 重命名后的数据集
            assets: 资源字典
            network_state_map: 网络状态映射
            all_raw_data: 所有原始数据
            
        Returns:
            (final_datasets, extra_assets) 二元组
        """
        # 初始化extra_assets，不再包含z-score参数
        extra_assets = {
            "mean_loss_cat2_up": 0.5,
            "mean_loss_cat2_dn": 0.5,
            "state_id_stats": {},
        }
        
        # 处理所有数据集
        final_datasets = {}
        state_id_stats = {}
        
        # 收集所有窗口元数据，确保按处理顺序排列
        all_windows_sorted = []
        for dataset_name, dataset in datasets.items():
            traces = group_by_trace_id(dataset)
            # 先按trace_id排序，再按start_time排序
            for trace_id in sorted(traces.keys()):
                metas = traces[trace_id]
                metas_sorted = sorted(metas, key=lambda x: x["start_time"])
                all_windows_sorted.extend(metas_sorted)
        
        # 行为ID已经在之前计算过，直接从窗口元数据中获取
        # 计算行为统计信息
        all_behavior_ids = []
        for window_meta in all_windows_sorted:
            behavior_id = window_meta.get("behavior_id", 0)
            all_behavior_ids.append(behavior_id)
        
        # 计算行为统计信息
        import numpy as np
        extra_assets["behavior_stats"] = self.pattern_identifier._calculate_behavior_statistics(np.array(all_behavior_ids))
        
        # 分配行为标签到各个数据集
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
                    # 直接从窗口元数据中获取行为ID
                    network_state_id = meta.get("behavior_id", 0)
                    
                    normed_meta = self._process_single_window(
                        meta, i, metas, network_state_id
                    )
                    normed_metas.append(normed_meta)
                    
                    # 统计state_id分布（只统计keep=True的样本）
                    if normed_meta["keep"]:
                        dataset_state_ids.append(int(normed_meta["cond"][0]))
            
            final_datasets[dataset_name] = normed_metas
            
            # 统计当前数据集的state_id分布
            state_id_stats[dataset_name] = self._compute_state_id_stats(dataset_state_ids)
        
        extra_assets["state_id_stats"] = state_id_stats
        return final_datasets, extra_assets
    
    def _process_single_window(
        self, meta: Dict, i: int, metas: List[Dict], network_state_id: int
    ) -> WindowMetaNormed:
        """处理单个窗口
        
        Args:
            meta: 窗口元数据
            i: 窗口索引
            metas: 同trace的所有窗口元数据
            network_state_id: 网络状态ID（来自行为发现模块）
            
        Returns:
            归一化后的窗口元数据
        """
        window_df = meta["window"].copy()  # 创建副本以避免修改原始数据
        
        # 保存原始丢包率
        # 上行丢包只保留0和1两个类别
        window_df.loc[window_df["loss_up"] > 0, "loss_up"] = 1.0
        # 下行丢包只保留0和1两个类别
        window_df.loc[window_df["loss_dn"] > 0, "loss_dn"] = 1.0
        
        # 计算全局特征（15维：1个state_id + 7个上行分位点 + 7个下行分位点）
        global_features = compute_window_features(window_df)
        
        # 使用行为发现模块生成的网络状态ID，替换第0维
        global_features[0] = float(network_state_id)
        
        # 使用全局特征作为条件向量，直接返回，不进行z-score标准化
        normalized_cond_vector = normalize_condition_vector(global_features)
        
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
    
    def _filter_and_reorganize_datasets(self, datasets: Dict[str, List[WindowMetaNormed]], out_dir: Path) -> Tuple[Dict[str, List[WindowMetaNormed]], List[WindowMetaNormed]]:
        """按行为类别筛选和重划分数据集
        
        Args:
            datasets: 原始数据集
            out_dir: 输出目录
            
        Returns:
            (final_datasets, other_datasets) 二元组
        """
        from collections import Counter
        import json
        
        # 1. 收集所有窗口数据
        all_windows = []
        for dataset_name, windows in datasets.items():
            all_windows.extend(windows)
        
        # 2. 按行为类别分组
        behavior_groups = {}
        for window in all_windows:
            # 行为标签位于条件向量的第0维（索引0）
            behavior_id = int(window['cond'][0])
            if behavior_id not in behavior_groups:
                behavior_groups[behavior_id] = []
            behavior_groups[behavior_id].append(window)
        
        # 3. 统计每个行为类别的样本数量
        behavior_counts = Counter()
        for behavior_id, windows in behavior_groups.items():
            behavior_counts[behavior_id] = sum(1 for w in windows if w['keep'])
        
        # 4. 筛选符合条件的行为类别（数目大于600的类别）
        selected_behaviors = [behavior_id for behavior_id, count in behavior_counts.items() if count > 600]
        other_behaviors = [behavior_id for behavior_id, count in behavior_counts.items() if count <= 600]
        
        print(f"\n=== 行为类别筛选结果 ===")
        print(f"总行为类别数: {len(behavior_counts)}")
        print(f"选中的行为类别: {selected_behaviors}")
        print(f"Other类别: {other_behaviors}")
        for behavior_id, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
            status = "选中" if behavior_id in selected_behaviors else "Other"
            print(f"  行为ID {behavior_id}: {count}个样本 ({status})")
        
        # 5. 初始化最终数据集
        final_train = []
        final_val = []
        final_test = []
        other_datasets = []
        
        # 6. 处理每个行为类别
        for behavior_id, windows in behavior_groups.items():
            # 只保留keep=True的窗口
            valid_windows = [w for w in windows if w['keep']]
            
            if behavior_id in other_behaviors:
                # 保存到other_datasets
                other_datasets.extend(valid_windows)
                continue
            
            # 按时间排序
            sorted_windows = sorted(valid_windows, key=lambda x: x['start_time'])
            num_windows = len(sorted_windows)
            
            print(f"\n处理行为ID {behavior_id}: {num_windows}个有效样本")
            
            # 7. 划分训练集、验证集和测试集
            val_windows = []
            test_windows = []
            
            if num_windows > 1000:
                # 样本数大于1000：随机挑选1000个作为训练集
                import numpy as np
                np.random.seed(42)
                # 随机挑选1000个样本作为训练集
                train_indices = np.random.choice(
                    num_windows, 
                    size=1000, 
                    replace=False
                )
                # 剩余样本的索引
                remaining_indices = np.setdiff1d(np.arange(num_windows), train_indices)
                
                # 构建训练集和剩余样本
                train_windows = [sorted_windows[i] for i in train_indices]
                remaining_windows = [sorted_windows[i] for i in remaining_indices]
                
                # 从剩余样本中随机挑选100个，均分为验证集和测试集
                if len(remaining_windows) >= 100:
                    # 随机挑选100个样本
                    selected_indices = np.random.choice(
                        len(remaining_windows), 
                        size=100, 
                        replace=False
                    )
                    selected_remaining = [remaining_windows[i] for i in selected_indices]
                    
                    # 均分为验证集和测试集
                    mid = len(selected_remaining) // 2
                    val_windows = selected_remaining[:mid]
                    test_windows = selected_remaining[mid:]
                    print(f"  训练集: 1000个样本, 验证集: {len(val_windows)}个样本, 测试集: {len(test_windows)}个样本")
                else:
                    # 剩余样本不足100个，全部作为验证集
                    val_windows = remaining_windows
                    print(f"  训练集: 1000个样本, 验证集: {len(val_windows)}个样本, 测试集: 0个样本")
            else:
                # 样本数小于1000：挑选最后10个左右作为验证集和测试集
                if num_windows >= 10:
                    # 取最后10个样本
                    last_10 = sorted_windows[-10:]
                    mid = len(last_10) // 2
                    val_windows = last_10[:mid]
                    test_windows = last_10[mid:]
                    # 训练集为剩下的样本
                    train_windows = sorted_windows[:-10]
                    print(f"  训练集: {len(train_windows)}个样本, 验证集: {len(val_windows)}个样本, 测试集: {len(test_windows)}个样本")
                else:
                    # 样本数不足10个，全部作为训练集
                    train_windows = sorted_windows
                    print(f"  训练集: {len(train_windows)}个样本, 验证集: 0个样本, 测试集: 0个样本")
            
            # 添加到最终数据集
            final_train.extend(train_windows)
            final_val.extend(val_windows)
            final_test.extend(test_windows)
        
        # 8. 构建最终数据集
        final_datasets = {
            "train": final_train,
            "val": final_val,
            "test": final_test
        }
        
        print(f"\n=== 最终数据集统计 ===")
        print(f"训练集: {len(final_train)}个样本")
        print(f"验证集: {len(final_val)}个样本")
        print(f"测试集: {len(final_test)}个样本")
        print(f"Other类别: {len(other_datasets)}个样本")
        
        return final_datasets, other_datasets
    
    def _save_other_category(self, other_datasets: List[WindowMetaNormed], out_dir: Path) -> None:
        """保存other类别数据
        
        Args:
            other_datasets: other类别数据
            out_dir: 输出目录
        """
        import json
        import numpy as np
        
        # 创建自定义JSON编码器，处理NumPy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.timestamp()
                return super(NumpyEncoder, self).default(obj)
        
        # 创建datasets目录路径
        datasets_dir = out_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建other.jsonl文件，保存到datasets目录下
        other_file = datasets_dir / "other.jsonl"
        
        # 保存数据
        with open(other_file, "w", encoding="utf-8") as f:
            for window in other_datasets:
                # 转换为可序列化的格式
                window_dict = {
                    "trace_id": window["trace_id"],
                    "start_time": window["start_time"],
                    "is_first": window["is_first"],
                    "keep": window["keep"],
                    "cond": window["cond"],
                    "window": window["window"].to_dict(orient="records")
                }
                
                # 使用自定义编码器写入文件
                f.write(json.dumps(window_dict, ensure_ascii=False, cls=NumpyEncoder) + "\n")
        
        print(f"\nOther类别数据已保存到: {other_file}")
    
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
            "network_state_id_source": "behavior_discovery_module",
            "network_state_id_encoding": "integer category stored as float (e.g., 2.0)",
            "level2_normalization": "Z-score normalization applied to 22 float dimensions (indices 0–10 and 12–22), excluding the integer-encoded network_state_id at index 11.",
            "split_strategy": "Entire traces are assigned to a single split to prevent data leakage.",
            "normalization": {
                "del_up/del_dn": "QuantileTransformer(output_distribution='normal'), fitted on train set",
                "loss_up/loss_dn": "Clipped to [0.0, 1.0], no transformation applied",
            },
        }
        return schema
