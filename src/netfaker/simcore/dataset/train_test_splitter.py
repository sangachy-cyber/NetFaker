"""HoloWAN 训练测试数据划分模块。

负责：
1. 扫描处理后的文件目录
2. 按字母顺序排序文件
3. 将文件分配到训练集和测试集
4. 聚合所有文件的窗口
5. 添加元数据（window_id, state_id=0）
6. 协调整个数据集生成过程
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from netfaker.simcore.dataset.window_extractor import WindowExtractor
from netfaker.simcore.io.dataset_saver import DatasetSaver


class TrainTestSplitter:
    """将文件划分为训练集和测试集的类。"""

    def __init__(self, processed_data_dir: str = 'data/processed/',
                 datasets_dir: str = 'data/datasets/'):
        """初始化训练测试数据划分器。

        Args:
            processed_data_dir: 包含处理后 Parquet 文件的目录
            datasets_dir: 保存生成数据集的目录
        """
        self.processed_data_dir = processed_data_dir
        self.datasets_dir = datasets_dir
        self.window_extractor = WindowExtractor()
        self.dataset_saver = DatasetSaver(datasets_dir)

        # 创建输出目录（如果不存在）
        os.makedirs(datasets_dir, exist_ok=True)

    def generate_datasets(self) -> Tuple[int, int]:
        """生成训练和测试数据集。

        Returns:
            (训练集大小, 测试集大小) 的元组

        Examples:
            >>> splitter = TrainTestSplitter()
            >>> train_size, test_size = splitter.generate_datasets()
            >>> print(f"训练样本: {train_size}, 测试样本: {test_size}")
            训练样本: 17774, 测试样本: 1198
        """
        # 步骤 1: 扫描并排序文件
        parquet_files = self._scan_parquet_files()

        if not parquet_files:
            logging.error("在 %s 中未找到 Parquet 文件", self.processed_data_dir)
            return 0, 0

        # 步骤 2: 将文件分割为训练集和测试集
        train_files, test_files = self._split_files(parquet_files)

        logging.info("在 %s 中找到 %d 个文件", self.processed_data_dir, len(parquet_files))
        logging.info("训练文件: %s", [os.path.splitext(f)[0] for f in train_files])
        logging.info("测试文件: %s", [os.path.splitext(f)[0] for f in test_files])

        # 步骤 3: 处理训练文件
        train_windows = self._process_files(train_files)

        # 步骤 4: 处理测试文件
        test_windows = self._process_files(test_files)

        # 步骤 5: 添加元数据并保存
        train_size = self.dataset_saver.save_dataset(train_windows, 'train.parquet')
        test_size = self.dataset_saver.save_dataset(test_windows, 'test.parquet')

        logging.info("已保存 train.parquet (%d 个样本), test.parquet (%d 个样本)",
                    train_size, test_size)

        return train_size, test_size

    def _scan_parquet_files(self) -> List[str]:
        """扫描目录中的 Parquet 文件。

        Returns:
            Parquet 文件名列表
        """
        parquet_files = []

        if not os.path.exists(self.processed_data_dir):
            logging.warning("目录 %s 不存在", self.processed_data_dir)
            return parquet_files

        for filename in os.listdir(self.processed_data_dir):
            if filename.endswith('.parquet'):
                parquet_files.append(filename)

        # 按字母顺序排序
        parquet_files.sort()

        return parquet_files

    def _split_files(self, files: List[str]) -> Tuple[List[str], List[str]]:
        """将文件分割为训练集和测试集。

        Args:
            files: 排序后的 Parquet 文件名列表

        Returns:
            (训练文件列表, 测试文件列表) 的元组
        """
        if len(files) == 0:
            return [], []
        elif len(files) == 1:
            # 如果只有一个文件，同时用于训练和测试
            return files, files
        else:
            # 最后一个文件作为测试集，其余作为训练集
            return files[:-1], files[-1:]

    def _process_files(self, files: List[str]) -> List[Dict[str, Any]]:
        """处理文件列表并提取窗口。

        Args:
            files: Parquet 文件名列表

        Returns:
            带有元数据的窗口字典列表
        """
        all_windows = []

        for filename in files:
            file_path = os.path.join(self.processed_data_dir, filename)

            try:
                # 加载 Parquet 文件
                df = pd.read_parquet(file_path)

                # 提取窗口
                windows = self.window_extractor.extract_windows(df, filename)

                # 添加元数据
                windows_with_metadata = self._add_metadata(windows, filename)

                all_windows.extend(windows_with_metadata)

                # 记录统计信息
                valid_count = sum(1 for w in windows if w['is_valid'])
                total_count = len(windows)
                logging.info("从 %s 生成了 %d 个窗口 (%d 个有效)",
                            filename, total_count, valid_count)

            except Exception as e:
                logging.error("处理文件 %s 时出错: %s", filename, str(e))

        return all_windows

    def _add_metadata(self, windows: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """为窗口添加元数据。

        Args:
            windows: 窗口字典列表
            filename: 源文件名

        Returns:
            带有元数据的窗口字典列表
        """
        source_file_stem = os.path.splitext(filename)[0]

        for window in windows:
            # 添加 window_id
            window_id = f"{source_file_stem}_idx{window['start_index']}"
            window['window_id'] = window_id

            # 添加 state_id (默认 0)
            window['state_id'] = 0

        return windows
