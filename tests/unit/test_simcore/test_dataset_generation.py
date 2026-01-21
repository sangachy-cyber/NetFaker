"""Test cases for HoloWAN dataset generation module.

Tests for:
1. Window extraction and validation rules
2. Train/test splitting logic
3. Dataset saving functionality
"""

import os
import tempfile

import pandas as pd
import pytest

from netfaker.simcore.dataset.train_test_splitter import TrainTestSplitter
from netfaker.simcore.dataset.window_extractor import WindowExtractor
from netfaker.simcore.io.dataset_saver import DatasetSaver


class TestWindowExtractor:
    """Test window extraction and validation logic."""

    def test_init(self):
        """Test initialization."""
        extractor = WindowExtractor()
        assert extractor.window_size == 100
        assert extractor.step_size == 50

    def test_extract_windows(self):
        """Test window extraction from single file."""
        # Create test data
        test_data = {
            'raw_delay_up': [100.0] * 150,
            'raw_loss_up': [0.1] * 150,
            'raw_bw_up': [10.0] * 150,
            'raw_delay_down': [95.0] * 150,
            'raw_loss_down': [0.0] * 150,
            'raw_bw_down': [12.0] * 150,
            'norm_delay_up': [0.0] * 150,
            'norm_loss_up': [0.001] * 150,
            'norm_bw_up': [10.0] * 150,
            'norm_delay_down': [0.0] * 150,
            'norm_loss_down': [0.0] * 150,
            'norm_bw_down': [12.0] * 150
        }
        df = pd.DataFrame(test_data)

        extractor = WindowExtractor()
        windows = extractor.extract_windows(df, 'test.parquet')

        # Should generate 2 windows: 0-100, 50-150
        assert len(windows) == 2
        assert windows[0]['start_index'] == 0
        assert windows[1]['start_index'] == 50
        assert all('is_valid' in window for window in windows)

    def test_high_delay_validation(self):
        """Test high delay validation."""
        # Create test data with high delay
        test_data = {
            'raw_delay_up': [2500.0] * 110,  # High delay
            'raw_loss_up': [0.1] * 110,
            'raw_bw_up': [10.0] * 110,
            'raw_delay_down': [95.0] * 110,
            'raw_loss_down': [0.0] * 110,
            'raw_bw_down': [12.0] * 110,
            'norm_delay_up': [0.0] * 110,
            'norm_loss_up': [0.001] * 110,
            'norm_bw_up': [10.0] * 110,
            'norm_delay_down': [0.0] * 110,
            'norm_loss_down': [0.0] * 110,
            'norm_bw_down': [12.0] * 110
        }
        df = pd.DataFrame(test_data)

        extractor = WindowExtractor()
        windows = extractor.extract_windows(df, 'test.parquet')

        # Should mark as invalid due to high delay
        assert len(windows) == 1  # Only 0-100
        assert not windows[0]['is_valid']

    def test_constant_delay_validation(self):
        """Test constant delay validation."""
        # Create test data with constant delay
        test_data = {
            'raw_delay_up': [100.0] * 110,  # Constant delay
            'raw_loss_up': [0.1] * 110,
            'raw_bw_up': [10.0] * 110,
            'raw_delay_down': [95.0] * 110,
            'raw_loss_down': [0.0] * 110,
            'raw_bw_down': [12.0] * 110,
            'norm_delay_up': [0.0] * 110,
            'norm_loss_up': [0.001] * 110,
            'norm_bw_up': [10.0] * 110,
            'norm_delay_down': [0.0] * 110,
            'norm_loss_down': [0.0] * 110,
            'norm_bw_down': [12.0] * 110
        }
        df = pd.DataFrame(test_data)

        extractor = WindowExtractor()
        windows = extractor.extract_windows(df, 'test.parquet')

        # Should mark as invalid due to constant delay
        assert len(windows) == 1  # Only 0-100
        assert not windows[0]['is_valid']


class TestTrainTestSplitter:
    """Test train/test splitting logic."""

    def test_init(self):
        """Test initialization."""
        splitter = TrainTestSplitter()
        assert splitter.processed_data_dir == 'data/processed/'
        assert splitter.datasets_dir == 'data/datasets/'

    def test_split_files(self):
        """Test file splitting logic."""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, 'file1.parquet')
            file2 = os.path.join(tmpdir, 'file2.parquet')

            # Create test data
            df1 = pd.DataFrame({
                'raw_delay_up': [100.0] * 150,
                'raw_loss_up': [0.1] * 150,
                'raw_bw_up': [10.0] * 150,
                'raw_delay_down': [95.0] * 150,
                'raw_loss_down': [0.0] * 150,
                'raw_bw_down': [12.0] * 150,
                'norm_delay_up': [0.0] * 150,
                'norm_loss_up': [0.001] * 150,
                'norm_bw_up': [10.0] * 150,
                'norm_delay_down': [0.0] * 150,
                'norm_loss_down': [0.0] * 150,
                'norm_bw_down': [12.0] * 150
            })
            df1.to_parquet(file1, index=False)

            df2 = pd.DataFrame({
                'raw_delay_up': [110.0] * 120,
                'raw_loss_up': [0.2] * 120,
                'raw_bw_up': [11.0] * 120,
                'raw_delay_down': [105.0] * 120,
                'raw_loss_down': [0.1] * 120,
                'raw_bw_down': [13.0] * 120,
                'norm_delay_up': [0.0] * 120,
                'norm_loss_up': [0.002] * 120,
                'norm_bw_up': [11.0] * 120,
                'norm_delay_down': [0.0] * 120,
                'norm_loss_down': [0.0] * 120,
                'norm_bw_down': [13.0] * 120
            })
            df2.to_parquet(file2, index=False)

            # Test splitter
            splitter = TrainTestSplitter(processed_data_dir=tmpdir, datasets_dir=tmpdir)
            train_files, test_files = splitter._split_files([os.path.basename(file1), os.path.basename(file2)])

            assert len(train_files) == 1
            assert len(test_files) == 1
            assert os.path.basename(file1) in train_files
            assert os.path.basename(file2) in test_files

    def test_split_files_empty(self):
        """Test file splitting with empty list."""
        splitter = TrainTestSplitter()
        train_files, test_files = splitter._split_files([])
        assert len(train_files) == 0
        assert len(test_files) == 0

    def test_split_files_single(self):
        """Test file splitting with single file."""
        splitter = TrainTestSplitter()
        train_files, test_files = splitter._split_files(['file1.parquet'])
        assert len(train_files) == 1
        assert len(test_files) == 1
        assert train_files == ['file1.parquet']
        assert test_files == ['file1.parquet']

    def test_scan_parquet_files(self):
        """Test scanning parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, 'file1.parquet')
            file2 = os.path.join(tmpdir, 'file2.parquet')
            non_parquet = os.path.join(tmpdir, 'non_parquet.txt')

            # Create test data
            df = pd.DataFrame({
                'raw_delay_up': [100.0] * 50,
                'raw_loss_up': [0.1] * 50,
                'raw_bw_up': [10.0] * 50,
                'raw_delay_down': [95.0] * 50,
                'raw_loss_down': [0.0] * 50,
                'raw_bw_down': [12.0] * 50,
                'norm_delay_up': [0.0] * 50,
                'norm_loss_up': [0.001] * 50,
                'norm_bw_up': [10.0] * 50,
                'norm_delay_down': [0.0] * 50,
                'norm_loss_down': [0.0] * 50,
                'norm_bw_down': [12.0] * 50
            })
            df.to_parquet(file1, index=False)
            df.to_parquet(file2, index=False)

            # Create non-parquet file
            with open(non_parquet, 'w') as f:
                f.write('test')

            # Test scanning
            splitter = TrainTestSplitter(processed_data_dir=tmpdir)
            files = splitter._scan_parquet_files()

            assert len(files) == 2
            assert 'file1.parquet' in files
            assert 'file2.parquet' in files
            assert files == sorted(files)  # Should be sorted

    def test_scan_parquet_files_nonexistent_dir(self):
        """Test scanning non-existent directory."""
        splitter = TrainTestSplitter(processed_data_dir='nonexistent_directory')
        files = splitter._scan_parquet_files()
        assert len(files) == 0

    def test_add_metadata(self):
        """Test adding metadata to windows."""
        splitter = TrainTestSplitter()

        # Create test windows
        test_windows = [
            {
                'start_index': 0,
                'is_valid': True,
                'raw_delay_up': [100.0] * 100
            },
            {
                'start_index': 50,
                'is_valid': False,
                'raw_delay_up': [200.0] * 100
            }
        ]

        # Test adding metadata
        windows_with_metadata = splitter._add_metadata(test_windows, 'test_file.parquet')

        assert len(windows_with_metadata) == 2
        assert windows_with_metadata[0]['window_id'] == 'test_file_idx0'
        assert windows_with_metadata[1]['window_id'] == 'test_file_idx50'
        assert windows_with_metadata[0]['state_id'] == 0
        assert windows_with_metadata[1]['state_id'] == 0

    def test_process_files(self):
        """Test processing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            file1 = os.path.join(tmpdir, 'file1.parquet')

            # Create test data
            df = pd.DataFrame({
                'raw_delay_up': [100.0] * 150,
                'raw_loss_up': [0.1] * 150,
                'raw_bw_up': [10.0] * 150,
                'raw_delay_down': [95.0] * 150,
                'raw_loss_down': [0.0] * 150,
                'raw_bw_down': [12.0] * 150,
                'norm_delay_up': [0.0] * 150,
                'norm_loss_up': [0.001] * 150,
                'norm_bw_up': [10.0] * 150,
                'norm_delay_down': [0.0] * 150,
                'norm_loss_down': [0.0] * 150,
                'norm_bw_down': [12.0] * 150
            })
            df.to_parquet(file1, index=False)

            # Test processing
            splitter = TrainTestSplitter(processed_data_dir=tmpdir)
            windows = splitter._process_files(['file1.parquet'])

            assert len(windows) > 0
            assert all('window_id' in window for window in windows)
            assert all('state_id' in window for window in windows)

    def test_process_files_error_handling(self):
        """Test error handling in file processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty file to cause error
            file1 = os.path.join(tmpdir, 'file1.parquet')
            open(file1, 'a').close()  # Create empty file

            # Test processing with error
            splitter = TrainTestSplitter(processed_data_dir=tmpdir)
            windows = splitter._process_files(['file1.parquet'])

            # Should handle error gracefully and return empty list
            assert isinstance(windows, list)

    def test_generate_datasets(self):
        """Test generate_datasets method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            file1 = os.path.join(tmpdir, 'file1.parquet')

            # Create test data
            df = pd.DataFrame({
                'raw_delay_up': [100.0] * 150,
                'raw_loss_up': [0.1] * 150,
                'raw_bw_up': [10.0] * 150,
                'raw_delay_down': [95.0] * 150,
                'raw_loss_down': [0.0] * 150,
                'raw_bw_down': [12.0] * 150,
                'norm_delay_up': [0.0] * 150,
                'norm_loss_up': [0.001] * 150,
                'norm_bw_up': [10.0] * 150,
                'norm_delay_down': [0.0] * 150,
                'norm_loss_down': [0.0] * 150,
                'norm_bw_down': [12.0] * 150
            })
            df.to_parquet(file1, index=False)

            # Test generate datasets
            splitter = TrainTestSplitter(processed_data_dir=tmpdir, datasets_dir=tmpdir)
            train_size, test_size = splitter.generate_datasets()

            # Should return valid sizes
            assert isinstance(train_size, int)
            assert isinstance(test_size, int)
            assert train_size >= 0
            assert test_size >= 0

    def test_generate_datasets_no_files(self):
        """Test generate_datasets with no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with empty directory
            splitter = TrainTestSplitter(processed_data_dir=tmpdir, datasets_dir=tmpdir)
            train_size, test_size = splitter.generate_datasets()

            assert train_size == 0
            assert test_size == 0


class TestDatasetSaver:
    """Test dataset saving functionality."""

    def test_init(self):
        """Test initialization."""
        saver = DatasetSaver()
        assert saver.datasets_dir == 'data/datasets/'

    def test_save_dataset(self):
        """Test saving dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test windows
            windows = [
                {
                    'window_id': 'test_idx0',
                    'source_file': 'test.parquet',
                    'start_index': 0,
                    'is_valid': True,
                    'raw_delay_up': [100.0] * 100,
                    'raw_loss_up': [0.1] * 100,
                    'raw_bw_up': [10.0] * 100,
                    'raw_delay_down': [95.0] * 100,
                    'raw_loss_down': [0.0] * 100,
                    'raw_bw_down': [12.0] * 100,
                    'norm_delay_up': [0.0] * 100,
                    'norm_loss_up': [0.001] * 100,
                    'norm_bw_up': [10.0] * 100,
                    'norm_delay_down': [0.0] * 100,
                    'norm_loss_down': [0.0] * 100,
                    'norm_bw_down': [12.0] * 100
                }
            ]

            # Test saving
            saver = DatasetSaver(datasets_dir=tmpdir)
            saved_count = saver.save_dataset(windows, 'test_dataset.parquet')

            assert saved_count == 1

            # Verify file exists
            output_file = os.path.join(tmpdir, 'test_dataset.parquet')
            assert os.path.exists(output_file)

            # Verify content
            df = pd.read_parquet(output_file)
            assert len(df) == 1
            assert df['is_valid'].iloc[0]


if __name__ == '__main__':
    pytest.main(['-v', __file__])
