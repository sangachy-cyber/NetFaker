"""HoloWAN 模块测试用例。

测试 holowan.py 模块中的各个类和函数的功能，确保其正确工作。
"""

import os
import tempfile
import numpy as np
import pytest
from src.netfaker.simcore.utils.holowan import (
    HoloWANDataPoint,
    HoloWANWindow,
    HoloWANFile
)


class TestHoloWANDataPoint:
    """测试 HoloWANDataPoint 类的功能。"""
    
    def test_init(self):
        """测试初始化方法。"""
        data_point = HoloWANDataPoint(
            delay1=100.0,
            loss1=0.1,
            bw1=10.0,
            delay2=95.0,
            loss2=0.0,
            bw2=12.0
        )
        
        assert data_point.delay1 == 100.0
        assert data_point.loss1 == 0.1
        assert data_point.bw1 == 10.0
        assert data_point.delay2 == 95.0
        assert data_point.loss2 == 0.0
        assert data_point.bw2 == 12.0
    
    def test_to_list(self):
        """测试转换为列表的方法。"""
        data_point = HoloWANDataPoint(
            delay1=100.0,
            loss1=0.1,
            bw1=10.0,
            delay2=95.0,
            loss2=0.0,
            bw2=12.0
        )
        
        expected_list = [100.0, 0.1, 10.0, 95.0, 0.0, 12.0]
        assert data_point.to_list() == expected_list
    
    def test_default_values(self):
        """测试默认值是否正确。"""
        data_point = HoloWANDataPoint()
        
        assert data_point.delay1 == 0.0
        assert data_point.loss1 == 0.0
        assert data_point.bw1 == 0.0
        assert data_point.delay2 == 0.0
        assert data_point.loss2 == 0.0
        assert data_point.bw2 == 0.0


class TestHoloWANWindow:
    """测试 HoloWANWindow 类的功能。"""
    
    def test_init(self):
        """测试初始化方法。"""
        window = HoloWANWindow()
        assert window.data_points == []
    
    def test_add_data_point(self):
        """测试添加数据点的方法。"""
        window = HoloWANWindow()
        data_point = HoloWANDataPoint(delay1=100.0)
        
        window.add_data_point(data_point)
        assert len(window.data_points) == 1
        assert window.data_points[0] == data_point
    
    def test_get_all_data(self):
        """测试获取所有数据的方法。"""
        window = HoloWANWindow()
        data_point1 = HoloWANDataPoint(delay1=100.0, loss1=0.1)
        data_point2 = HoloWANDataPoint(delay1=105.0, loss1=0.0)
        
        window.add_data_point(data_point1)
        window.add_data_point(data_point2)
        
        all_data = window.get_all_data()
        expected_data = [[100.0, 0.1, 0.0, 0.0, 0.0, 0.0], [105.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        
        assert all_data == expected_data


class TestHoloWANFile:
    """测试 HoloWANFile 类的功能。"""
    
    def test_init(self):
        """测试初始化方法。"""
        holowan_file = HoloWANFile(
            operator="TestOperator",
            network_type="5G",
            signal_strength=-80,
            test_name="Test-2024-01-01"
        )
        
        assert holowan_file.operator == "TestOperator"
        assert holowan_file.network_type == "5G"
        assert holowan_file.signal_strength == -80
        assert holowan_file.test_name == "Test-2024-01-01"
        assert holowan_file.data == []
    
    def test_add_data(self):
        """测试添加数据的方法。"""
        holowan_file = HoloWANFile()
        holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
        
        assert len(holowan_file.data) == 1
        assert holowan_file.data[0].delay1 == 100.0
    
    def test_add_data_point(self):
        """测试添加数据点对象的方法。"""
        holowan_file = HoloWANFile()
        data_point = HoloWANDataPoint(delay1=100.0)
        
        holowan_file._add_data_point(data_point)
        assert len(holowan_file.data) == 1
        assert holowan_file.data[0] == data_point
    
    def test_write_to_file(self):
        """测试写入文件的方法。"""
        holowan_file = HoloWANFile()
        # 开启带宽开关
        holowan_file.switch_bw1 = True
        holowan_file.switch_bw2 = True
        
        holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
        holowan_file._add_data(105.0, 0.0, 9.5, 98.0, 0.1, 11.5)
        
        # 使用临时文件
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            output_path = holowan_file.write_to_file(tmp_path)
            assert os.path.exists(output_path)
            
            # 验证文件内容
            with open(output_path, "r") as f:
                content = f.read()
                assert "HoloWAN Recorder File" in content
                assert "100.00,0.10,10.000000,95.00,0.00,12.000000" in content
                assert "105.00,0.00,9.500000,98.00,0.10,11.500000" in content
        finally:
            os.unlink(tmp_path)
    
    def test_from_file(self):
        """测试从文件读取的方法。"""
        # 创建一个简单的HoloWAN文件
        file_content = """HoloWAN Recorder File (www.msytest.com)
Operator: "TestOperator" NetworkType: "5G" SignalStrength: -80(dbm)
test_name: "Test-2024-01-01"
Destination: "127.0.0.1:8080"
Start Time: 2024-01-01 00:00:00
End Time: 2024-01-01 00:01:00
Interval(sec): 0.1
Packet Size(byte): 500
Enable Reordering: True
Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)
Switch: 1,1,1,1,1,1
------------------------------------------------
100.00,0.10,10.000000,95.00,0.00,12.000000
105.00,0.00,9.500000,98.00,0.10,11.500000
"""
        
        # 使用临时文件
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(file_content.encode())
            tmp_path = tmp.name
        
        try:
            holowan_file = HoloWANFile.from_file(tmp_path)
            assert holowan_file.operator == "TestOperator"
            assert holowan_file.network_type == "5G"
            assert holowan_file.signal_strength == -80
            assert holowan_file.test_name == "Test-2024-01-01"
            assert len(holowan_file.data) == 2
        finally:
            os.unlink(tmp_path)
    
    def test_get_switch_str(self):
        """测试获取开关字符串的方法。"""
        holowan_file = HoloWANFile()
        holowan_file.switch_delay1 = True
        holowan_file.switch_loss1 = False
        holowan_file.switch_bw1 = True
        holowan_file.switch_delay2 = False
        holowan_file.switch_loss2 = True
        holowan_file.switch_bw2 = False
        
        switch_str = holowan_file._get_switch_str()
        assert switch_str == "1,0,1,0,1,0"


class TestHelperFunctions:
    """测试辅助函数的功能。"""
    
    def test_read_real_file(self):
        """测试读取真实HoloWAN文件的功能。"""
        # 使用项目中的真实数据文件
        real_file_path = "data/raw/20251203_230356_b6x-playback.txt"
        
        if os.path.exists(real_file_path):
            holowan_file = HoloWANFile.from_file(real_file_path)
            assert len(holowan_file.data) > 0
            assert holowan_file.start_time is not None
            assert holowan_file.end_time is not None
