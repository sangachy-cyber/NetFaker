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

from netfaker.simcore.dataset.window_extractor import WindowExtractor
from netfaker.simcore.dataset.train_test_splitter import TrainTestSplitter
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
            assert df['is_valid'].iloc[0] == True


if __name__ == '__main__':
    pytest.main(['-v', __file__])
