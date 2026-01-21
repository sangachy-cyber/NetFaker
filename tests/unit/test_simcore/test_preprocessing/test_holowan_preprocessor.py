"""HoloWAN 预处理模块测试。

测试 HoloWAN 数据加载、预处理和保存功能。
"""

import os
import tempfile

import pandas as pd

from netfaker.simcore.io.holowan_loader import HoloWANLoader
from netfaker.simcore.io.holowan_saver import HoloWANWriter
from netfaker.simcore.preprocessing.holowan_preprocessor import HoloWANPreprocessor


class TestHoloWANLoader:
    """测试 HoloWAN 数据加载器。"""

    def test_init(self):
        """测试初始化。"""
        loader = HoloWANLoader()
        assert loader.raw_data_dir == "data/raw/"

    def test_load_file(self):
        """测试加载文件。"""
        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""HoloWAN Recorder File (www.msytest.com)
Operator: "TestOperator" NetworkType: "4G" SignalStrength: -85(dbm)
test_name: "test"
Destination: "192.168.1.1:8080"
Start Time: 2024-01-21 10:00:00
End Time: 2024-01-21 10:00:01
Interval(sec): 0.1
Packet Size(byte): 500
Loss Average: 0.05
Enable Reordering: true
Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)
Switch: 1,1,1,1,1,1
------------------------------------------------
100.0,0.1,10.0,95.0,0.0,12.0
105.0,0.2,9.5,98.0,0.1,11.5
""")
            temp_file = f.name

        try:
            loader = HoloWANLoader()
            df = loader.load_file(temp_file)

            # 验证 DataFrame 结构
            assert isinstance(df, pd.DataFrame)
            expected_columns = [
                'raw_delay_up', 'raw_loss_up', 'raw_bw_up',
                'raw_delay_down', 'raw_loss_down', 'raw_bw_down'
            ]
            assert all(col in df.columns for col in expected_columns)
            assert len(df) == 2

        finally:
            os.unlink(temp_file)


class TestHoloWANPreprocessor:
    """测试 HoloWAN 数据预处理器。"""

    def test_init(self):
        """测试初始化。"""
        preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)
        assert preprocessor.shared_delay_scaler is True

        preprocessor = HoloWANPreprocessor(shared_delay_scaler=False)
        assert preprocessor.shared_delay_scaler is False

    def test_transform(self):
        """测试数据转换。"""
        # 创建测试数据
        data = {
            'raw_delay_up': [100.0, 105.0, 98.0],
            'raw_loss_up': [0.1, 0.2, 0.0],
            'raw_bw_up': [10.0, 9.5, 10.5],
            'raw_delay_down': [95.0, 98.0, 92.0],
            'raw_loss_down': [0.0, 0.1, 0.0],
            'raw_bw_down': [12.0, 11.5, 12.5]
        }
        df = pd.DataFrame(data)

        # 创建预处理器
        preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)

        # 拟合 scaler
        preprocessor._fit_shared_scaler({'test': df})

        # 转换数据
        processed_df = preprocessor.transform(df)

        # 验证结果
        expected_columns = [
            'raw_delay_up', 'raw_loss_up', 'raw_bw_up',
            'raw_delay_down', 'raw_loss_down', 'raw_bw_down',
            'norm_delay_up', 'norm_loss_up', 'norm_bw_up',
            'norm_delay_down', 'norm_loss_down', 'norm_bw_down'
        ]
        assert all(col in processed_df.columns for col in expected_columns)

        # 验证丢包率缩放
        assert all(0 <= val <= 1 for val in processed_df['norm_loss_up'])
        assert all(0 <= val <= 1 for val in processed_df['norm_loss_down'])

        # 验证带宽保持不变
        assert all(processed_df['raw_bw_up'] == processed_df['norm_bw_up'])
        assert all(processed_df['raw_bw_down'] == processed_df['norm_bw_down'])


class TestHoloWANWriter:
    """测试 HoloWAN 数据保存器。"""

    def test_save_data(self):
        """测试保存数据。"""
        # 创建测试数据
        data = {
            'raw_delay_up': [100.0],
            'raw_loss_up': [0.1],
            'raw_bw_up': [10.0],
            'raw_delay_down': [95.0],
            'raw_loss_down': [0.0],
            'raw_bw_down': [12.0],
            'norm_delay_up': [0.0],
            'norm_loss_up': [0.001],
            'norm_bw_up': [10.0],
            'norm_delay_down': [0.0],
            'norm_loss_down': [0.0],
            'norm_bw_down': [12.0]
        }
        df = pd.DataFrame(data)

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
            temp_output = f.name

        try:
            writer = HoloWANWriter()
            writer.save_data(df, temp_output)

            # 验证文件存在
            assert os.path.exists(temp_output)

            # 验证文件可以读取
            loaded_df = pd.read_parquet(temp_output)
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) == 1

        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)

    def test_save_scaler(self):
        """测试保存 scaler。"""
        # 创建测试 scaler 信息
        scaler_info = {
            'mode': 'shared',
            'scaler': None  # 实际测试中会有真实的 scaler
        }

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.joblib', delete=False) as f:
            temp_output = f.name

        try:
            writer = HoloWANWriter()
            writer.save_scaler(scaler_info, temp_output)

            # 验证文件存在
            assert os.path.exists(temp_output)

        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)


class TestHoloWANPreprocessingPipeline:
    """测试完整的 HoloWAN 预处理流水线。"""

    def test_full_pipeline(self):
        """测试完整流水线。"""
        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""HoloWAN Recorder File (www.msytest.com)
Operator: "TestOperator" NetworkType: "4G" SignalStrength: -85(dbm)
test_name: "test"
Destination: "192.168.1.1:8080"
Start Time: 2024-01-21 10:00:00
End Time: 2024-01-21 10:00:01
Interval(sec): 0.1
Packet Size(byte): 500
Loss Average: 0.05
Enable Reordering: true
Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)
Switch: 1,1,1,1,1,1
------------------------------------------------
100.0,0.1,10.0,95.0,0.0,12.0
105.0,0.2,9.5,98.0,0.1,11.5
""")
            temp_input = f.name

        # 创建临时输出目录
        temp_output_dir = tempfile.mkdtemp()

        try:
            # 1. 加载数据
            loader = HoloWANLoader(raw_data_dir=os.path.dirname(temp_input))
            data_dict = loader.load_all_files()
            assert len(data_dict) == 1

            # 2. 预处理数据
            preprocessor = HoloWANPreprocessor(shared_delay_scaler=True)
            processed_data = preprocessor.fit_transform(data_dict)
            assert len(processed_data) == 1

            # 3. 保存数据
            writer = HoloWANWriter(processed_data_dir=temp_output_dir)
            writer.save_all_data(processed_data)

            # 4. 保存 scaler
            scaler_info = preprocessor.get_scaler_info()
            scaler_path = os.path.join(temp_output_dir, "global_delay_scaler.joblib")
            writer.save_scaler(scaler_info, scaler_path)

            # 验证输出文件存在
            expected_output = os.path.join(temp_output_dir, f"{os.path.splitext(os.path.basename(temp_input))[0]}.parquet")
            assert os.path.exists(expected_output)

            expected_scaler = os.path.join(temp_output_dir, "global_delay_scaler.joblib")
            assert os.path.exists(expected_scaler)

        finally:
            os.unlink(temp_input)
            import shutil
            shutil.rmtree(temp_output_dir)
