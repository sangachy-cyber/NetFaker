# 损伤仪模块测试用例

"""
本测试文件包含损伤仪模块的单元测试和集成测试。
测试内容包括：
- 抽象损伤仪接口
- 损伤仪工厂
"""

import unittest
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

from netfaker.impairment import (
    AbstractImpairmentDevice,
    ImpairmentFactory
)


class TestAbstractImpairmentDevice(unittest.TestCase):
    """测试抽象损伤仪接口"""

    def test_init(self):
        """测试初始化方法"""
        # 创建一个简单的子类来测试抽象基类
        class TestDevice(AbstractImpairmentDevice):
            def connect(self) -> bool:
                return True
            def disconnect(self) -> bool:
                return True
            def is_connected(self) -> bool:
                return True
            def get_device_info(self) -> Dict[str, Any]:
                return {}
            def get_engine_info(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def create_path(self, engine_id: int, path_id: int, path_name: str = "PATH") -> Dict[str, Any]:
                return {}
            def delete_path(self, engine_id: int, path_id: int, path_name: str, force: bool = False) -> Dict[str, Any]:
                return {}
            def set_path_property(self, engine_id: int, path_id: int, **kwargs) -> Dict[str, Any]:
                return {}
            def get_path_info(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def get_all_paths(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def path_exists(self, engine_id: int, path_id: int) -> bool:
                return False
            def get_all_classifiers(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def classifier_exists(self, engine_id: int, classifier_id: int) -> bool:
                return False
            def get_path_id_by_name(self, engine_id: int, path_name: str) -> Optional[int]:
                return None
            def get_classifier_id_by_name(self, engine_id: int, classifier_name: str) -> Optional[int]:
                return None
            def apply_impairment(self, engine_id: int, path_id: int, impairment_params: Dict[str, Any]) -> Dict[str, Any]:
                return {}
            def clear_impairment(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def set_impairment_param(self, engine_id: int, path_id: int, param_name: str, param_value: Any) -> Dict[str, Any]:
                return {}
            def create_rule(self, engine_id: int, rule_params: Dict[str, Any]) -> Dict[str, Any]:
                return {}
            def bind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def unbind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def delete_rule(self, engine_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def get_device_stats(self) -> Dict[str, Any]:
                return {}
            def get_path_stats(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def start_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def stop_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def reset_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
        
        # 测试初始化
        device = TestDevice(ip="192.168.1.111", port="8080")
        self.assertEqual(device.ip, "192.168.1.111")
        self.assertEqual(device.port, "8080")
        self.assertFalse(device.connected)

        # 测试新添加的方法
        self.assertEqual(device.get_all_paths(1), {})
        self.assertFalse(device.path_exists(1, 1))
        self.assertEqual(device.get_all_classifiers(1), {})
        self.assertFalse(device.classifier_exists(1, 1))
        self.assertIsNone(device.get_path_id_by_name(1, "test_path"))
        self.assertIsNone(device.get_classifier_id_by_name(1, "test_classifier"))


class TestImpairmentFactory(unittest.TestCase):
    """测试损伤仪工厂"""

    def test_register_device(self):
        """测试注册损伤仪类型"""
        # 创建一个测试适配器类
        class TestAdapter(AbstractImpairmentDevice):
            def connect(self) -> bool:
                return True
            def disconnect(self) -> bool:
                return True
            def is_connected(self) -> bool:
                return True
            def get_device_info(self) -> Dict[str, Any]:
                return {}
            def get_engine_info(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def create_path(self, engine_id: int, path_id: int, path_name: str = "PATH") -> Dict[str, Any]:
                return {}
            def delete_path(self, engine_id: int, path_id: int, path_name: str, force: bool = False) -> Dict[str, Any]:
                return {}
            def set_path_property(self, engine_id: int, path_id: int, **kwargs) -> Dict[str, Any]:
                return {}
            def get_path_info(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def apply_impairment(self, engine_id: int, path_id: int, impairment_params: Dict[str, Any]) -> Dict[str, Any]:
                return {}
            def clear_impairment(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def set_impairment_param(self, engine_id: int, path_id: int, param_name: str, param_value: Any) -> Dict[str, Any]:
                return {}
            def create_rule(self, engine_id: int, rule_params: Dict[str, Any]) -> Dict[str, Any]:
                return {}
            def bind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def unbind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def delete_rule(self, engine_id: int, rule_id: int) -> Dict[str, Any]:
                return {}
            def get_device_stats(self) -> Dict[str, Any]:
                return {}
            def get_path_stats(self, engine_id: int, path_id: int) -> Dict[str, Any]:
                return {}
            def start_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def stop_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
            def reset_engine(self, engine_id: int) -> Dict[str, Any]:
                return {}
        
        # 注册设备类型
        ImpairmentFactory.register_device("test_device", TestAdapter)
        
        # 检查是否注册成功
        supported_devices = ImpairmentFactory.get_supported_devices()
        self.assertIn("test_device", supported_devices)
        self.assertEqual(supported_devices["test_device"], TestAdapter)

    def test_create_device_invalid_type(self):
        """测试创建无效类型的损伤仪实例"""
        # 创建无效类型的设备
        device = ImpairmentFactory.create_device(
            device_type="invalid_type",
            ip="192.168.1.111",
            port="8080"
        )
        
        # 检查是否返回None
        self.assertIsNone(device)


if __name__ == '__main__':
    unittest.main()
