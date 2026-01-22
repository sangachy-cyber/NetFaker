# 损伤仪模块

"""
损伤仪模块提供了与各种网络损伤仪交互的抽象接口和具体实现。
当前支持HoloWAN损伤仪，未来可扩展支持其他类型的损伤仪。

主要组件：
- AbstractImpairmentDevice: 抽象损伤仪基类，定义通用接口
- HoloWANAdapter: HoloWAN损伤仪的具体实现
- ImpairmentFactory: 损伤仪工厂，用于创建损伤仪实例

使用示例：
```python
from netfaker.impairment import ImpairmentFactory

# 创建HoloWAN损伤仪实例
impairment_device = ImpairmentFactory.create_device(
    device_type="holowan",
    ip="192.168.1.111",
    port="8080"
)

# 连接设备
impairment_device.connect()

# 应用损伤配置
impairment_device.apply_impairment(engine_id=1, path_id=1, impairment_params={"delay": 100})

# 断开连接
impairment_device.disconnect()
```
"""

# 导出主要组件
from netfaker.impairment.abstract_impairment import AbstractImpairmentDevice
from netfaker.impairment.impairment_factory import ImpairmentFactory

__all__ = [
    "AbstractImpairmentDevice",
    "HoloWANAdapter",
    "ImpairmentFactory"
]

# 延迟导入HoloWANAdapter，避免循环导入问题
def __getattr__(name):
    if name == "HoloWANAdapter":
        from netfaker.impairment.holowan_adapter import HoloWANAdapter
        return HoloWANAdapter
    raise AttributeError(f"模块 '{__name__}' 没有属性 '{name}'")
