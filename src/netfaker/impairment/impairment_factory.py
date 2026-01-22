# 损伤仪工厂

"""
损伤仪工厂类用于创建不同类型的损伤仪实例。
采用工厂模式，支持动态注册和创建不同类型的损伤仪适配器。

使用示例：
```python
from netfaker.impairment import ImpairmentFactory

# 创建HoloWAN损伤仪实例
impairment_device = ImpairmentFactory.create_device(
    device_type="holowan",
    ip="192.168.1.111",
    port="8080"
)
```
"""

from typing import Any, Dict, Optional, Type

from loguru import logger

from netfaker.impairment.abstract_impairment import AbstractImpairmentDevice


class ImpairmentFactory:
    """损伤仪工厂类，用于创建损伤仪实例"""

    # 注册的损伤仪类型字典，key为设备类型，value为适配器类
    _registered_devices: Dict[str, Type[AbstractImpairmentDevice]] = {}

    @classmethod
    def register_device(cls, device_type: str, adapter_class: Type[AbstractImpairmentDevice]) -> None:
        """注册损伤仪类型

        Args:
            device_type: 设备类型名称
            adapter_class: 适配器类，继承自AbstractImpairmentDevice
        """
        if not issubclass(adapter_class, AbstractImpairmentDevice):
            raise TypeError(f"适配器类必须继承自AbstractImpairmentDevice: {adapter_class}")
        
        cls._registered_devices[device_type.lower()] = adapter_class
        logger.info(f"成功注册损伤仪类型: {device_type}")

    @classmethod
    def create_device(cls, device_type: str, **kwargs) -> Optional[AbstractImpairmentDevice]:
        """创建损伤仪实例

        Args:
            device_type: 设备类型名称
            **kwargs: 损伤仪初始化参数

        Returns:
            Optional[AbstractImpairmentDevice]: 损伤仪实例，创建失败返回None
        """
        device_type = device_type.lower()
        
        # 如果设备类型未注册，尝试动态导入
        if device_type not in cls._registered_devices:
            cls._try_dynamic_import(device_type)
        
        # 检查设备类型是否已注册
        if device_type not in cls._registered_devices:
            logger.error(f"未注册的损伤仪类型: {device_type}")
            return None
        
        try:
            # 创建损伤仪实例
            adapter_class = cls._registered_devices[device_type]
            device = adapter_class(**kwargs)
            logger.info(f"成功创建损伤仪实例: {device_type}, 参数: {kwargs}")
            return device
        except Exception as e:
            logger.error(f"创建损伤仪实例失败: {device_type}, 错误: {e}")
            return None

    @classmethod
    def get_supported_devices(cls) -> Dict[str, Type[AbstractImpairmentDevice]]:
        """获取支持的损伤仪类型列表

        Returns:
            Dict[str, Type[AbstractImpairmentDevice]]: 支持的损伤仪类型字典
        """
        return cls._registered_devices.copy()

    @classmethod
    def _try_dynamic_import(cls, device_type: str) -> None:
        """尝试动态导入损伤仪适配器

        Args:
            device_type: 设备类型名称
        """
        try:
            # 尝试动态导入内置的适配器
            if device_type == "holowan":
                from netfaker.impairment.holowan_adapter import HoloWANAdapter
                cls.register_device("holowan", HoloWANAdapter)
            # 可扩展其他内置适配器
        except ImportError as e:
            logger.error(f"动态导入损伤仪适配器失败: {device_type}, 错误: {e}")
        except Exception as e:
            logger.error(f"处理损伤仪适配器失败: {device_type}, 错误: {e}")
