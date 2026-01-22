# 抽象损伤仪接口

"""
抽象损伤仪接口定义了与各种网络损伤仪交互的通用方法。
所有具体的损伤仪实现都应该继承这个抽象基类，并实现其中的抽象方法。

主要功能包括：
- 连接管理
- 设备信息获取
- 路径管理
- 损伤配置
- 规则管理
- 统计信息获取
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class AbstractImpairmentDevice(ABC):
    """抽象损伤仪基类，定义通用接口"""

    def __init__(self, ip: str, port: str, **kwargs):
        """初始化损伤仪实例

        Args:
            ip: 损伤仪IP地址
            port: 损伤仪端口号
            **kwargs: 其他可选参数
        """
        self.ip = ip
        self.port = port
        self.connected = False

    # 连接管理
    @abstractmethod
    def connect(self) -> bool:
        """连接损伤仪

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开损伤仪连接

        Returns:
            bool: 断开是否成功
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查损伤仪是否已连接

        Returns:
            bool: 连接状态
        """
        pass

    # 设备信息获取
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备基本信息

        Returns:
            Dict[str, Any]: 设备信息字典
        """
        pass

    @abstractmethod
    def get_engine_info(self, engine_id: int) -> Dict[str, Any]:
        """获取引擎信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 引擎信息字典
        """
        pass

    # 路径管理
    @abstractmethod
    def create_path(self, engine_id: int, path_id: int, path_name: str = "PATH") -> Dict[str, Any]:
        """创建虚拟路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            path_name: 路径名称

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def delete_path(self, engine_id: int, path_id: int, path_name: str, force: bool = False) -> Dict[str, Any]:
        """删除虚拟路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            path_name: 路径名称
            force: 是否强制删除

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def set_path_property(self, engine_id: int, path_id: int, **kwargs) -> Dict[str, Any]:
        """设置路径属性

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            **kwargs: 路径属性键值对

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def get_path_info(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """获取路径信息

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 路径信息字典
        """
        pass

    @abstractmethod
    def get_all_paths(self, engine_id: int) -> Dict[str, Any]:
        """获取所有路径信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 所有路径信息字典，key为路径ID
        """
        pass

    @abstractmethod
    def path_exists(self, engine_id: int, path_id: int) -> bool:
        """检查路径是否存在

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            bool: 路径是否存在
        """
        pass

    # 分类器管理
    @abstractmethod
    def get_all_classifiers(self, engine_id: int) -> Dict[str, Any]:
        """获取所有分类器信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 所有分类器信息字典，key为分类器ID
        """
        pass

    @abstractmethod
    def classifier_exists(self, engine_id: int, classifier_id: int) -> bool:
        """检查分类器是否存在

        Args:
            engine_id: 引擎ID
            classifier_id: 分类器ID

        Returns:
            bool: 分类器是否存在
        """
        pass

    # 名称查询功能
    @abstractmethod
    def get_path_id_by_name(self, engine_id: int, path_name: str) -> Optional[int]:
        """根据路径名称获取路径ID

        Args:
            engine_id: 引擎ID
            path_name: 路径名称

        Returns:
            Optional[int]: 路径ID，如果未找到返回None
        """
        pass

    @abstractmethod
    def get_classifier_id_by_name(self, engine_id: int, classifier_name: str) -> Optional[int]:
        """根据分类器名称获取分类器ID

        Args:
            engine_id: 引擎ID
            classifier_name: 分类器名称

        Returns:
            Optional[int]: 分类器ID，如果未找到返回None
        """
        pass

    # 损伤配置
    @abstractmethod
    def apply_impairment(self, engine_id: int, path_id: int, impairment_params: Dict[str, Any]) -> Dict[str, Any]:
        """应用损伤配置

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            impairment_params: 损伤参数字典

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def clear_impairment(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """清除路径上的所有损伤

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def set_impairment_param(self, engine_id: int, path_id: int, param_name: str, param_value: Any) -> Dict[str, Any]:
        """设置单个损伤参数

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            param_name: 参数名称
            param_value: 参数值

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    # 规则管理
    @abstractmethod
    def create_rule(self, engine_id: int, rule_params: Dict[str, Any]) -> Dict[str, Any]:
        """创建规则

        Args:
            engine_id: 引擎ID
            rule_params: 规则参数字典

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def bind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
        """绑定规则到路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def unbind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
        """从路径解绑规则

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def delete_rule(self, engine_id: int, rule_id: int) -> Dict[str, Any]:
        """删除规则

        Args:
            engine_id: 引擎ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    # 统计信息获取
    @abstractmethod
    def get_device_stats(self) -> Dict[str, Any]:
        """获取设备统计信息

        Returns:
            Dict[str, Any]: 设备统计信息字典
        """
        pass

    @abstractmethod
    def get_path_stats(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """获取路径统计信息

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 路径统计信息字典
        """
        pass

    # 引擎控制
    @abstractmethod
    def start_engine(self, engine_id: int) -> Dict[str, Any]:
        """启动引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def stop_engine(self, engine_id: int) -> Dict[str, Any]:
        """停止引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass

    @abstractmethod
    def reset_engine(self, engine_id: int) -> Dict[str, Any]:
        """重置引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        pass
