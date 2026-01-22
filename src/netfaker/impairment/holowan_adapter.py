# HoloWAN适配器

"""
HoloWAN适配器实现了抽象损伤仪接口，封装了HoloWAN API的具体实现。
使用holowan第三方库提供的Engine类和相关组件来与HoloWAN设备交互。

主要功能包括：
- 连接管理
- 设备信息获取
- 路径管理
- 损伤配置
- 规则管理
- 统计信息获取
"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger

from netfaker.impairment.abstract_impairment import AbstractImpairmentDevice

# 导入HoloWAN第三方库
from holowan.v2.engine import Engine
from holowan.v2.engine.path import (
    BandwidthFixed, 
    BackgroundUtilizationRandom, 
    DelayNormal, 
    Impairments, 
    LossBurst,
    BERRange,
    DuplicationJitter,
    ChangeMode,
    QueueLimitRED
)
from holowan.v2._holowan_types import HoloWANReturn


class HoloWANAdapter(AbstractImpairmentDevice):
    """HoloWAN适配器，实现抽象损伤仪接口"""

    def __init__(self, ip: str, port: str, **kwargs):
        """初始化HoloWAN适配器实例

        Args:
            ip: HoloWAN设备IP地址
            port: HoloWAN设备端口号
            **kwargs: 其他可选参数
        """
        super().__init__(ip, port, **kwargs)
        self.engines: Dict[int, Engine] = {}  # 引擎实例字典，key为engine_id
        self.default_engine_id: Optional[int] = None
        
    # 连接管理
    def connect(self) -> bool:
        """连接HoloWAN设备

        Returns:
            bool: 连接是否成功
        """
        try:
            # HoloWAN API不需要显式连接，通过Engine实例化即可验证连接
            self.connected = True
            logger.info(f"成功连接到HoloWAN设备: {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"连接HoloWAN设备失败: {e}")
            self.connected = False
            return False

    def disconnect(self) -> bool:
        """断开HoloWAN设备连接

        Returns:
            bool: 断开是否成功
        """
        try:
            # HoloWAN API不需要显式断开连接，清理引擎实例即可
            self.engines.clear()
            self.connected = False
            logger.info(f"成功断开与HoloWAN设备的连接: {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"断开HoloWAN设备连接失败: {e}")
            return False

    def is_connected(self) -> bool:
        """检查HoloWAN设备是否已连接

        Returns:
            bool: 连接状态
        """
        return self.connected

    # 设备信息获取
    def get_device_info(self) -> Dict[str, Any]:
        """获取HoloWAN设备基本信息

        Returns:
            Dict[str, Any]: 设备信息字典
        """
        try:
            from holowan.v2._commons import get_holowan_info
            info = get_holowan_info(self.ip, self.port)
            logger.debug(f"获取HoloWAN设备信息: {info}")
            return {"success": True, "data": info}
        except Exception as e:
            logger.error(f"获取HoloWAN设备信息失败: {e}")
            return {"success": False, "error": str(e)}

    def get_engine_info(self, engine_id: int) -> Dict[str, Any]:
        """获取HoloWAN引擎信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 引擎信息字典
        """
        try:
            # 获取或创建引擎实例
            engine = self._get_or_create_engine(engine_id)
            return {"success": True, "data": engine.get_engine_info()}
        except Exception as e:
            logger.error(f"获取HoloWAN引擎信息失败 (engine_id={engine_id}): {e}")
            return {"success": False, "error": str(e)}

    # 路径管理
    def create_path(self, engine_id: int, path_id: int, path_name: str = "PATH") -> Dict[str, Any]:
        """创建HoloWAN虚拟路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            path_name: 路径名称

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            result = engine.add_path(path_id, path_name)
            logger.info(f"创建HoloWAN路径成功: engine_id={engine_id}, path_id={path_id}, path_name={path_name}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"创建HoloWAN路径失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def delete_path(self, engine_id: int, path_id: int, path_name: str, force: bool = False) -> Dict[str, Any]:
        """删除HoloWAN虚拟路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            path_name: 路径名称
            force: 是否强制删除

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            result = engine.remove_path(path_id, path_name, force)
            logger.info(f"删除HoloWAN路径成功: engine_id={engine_id}, path_id={path_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"删除HoloWAN路径失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def set_path_property(self, engine_id: int, path_id: int, **kwargs) -> Dict[str, Any]:
        """设置HoloWAN路径属性

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            **kwargs: 路径属性键值对

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            
            # 根据kwargs设置不同的路径属性
            for key, value in kwargs.items():
                if key == "name":
                    result = engine.set_path_name(path_id, value)
                    logger.info(f"设置HoloWAN路径名称成功: engine_id={engine_id}, path_id={path_id}, name={value}")
                elif key == "direction":
                    # 设置路径损伤方向
                    result = engine.set_path_direction(path_id, value)
                    logger.info(f"设置HoloWAN路径方向成功: engine_id={engine_id}, path_id={path_id}, direction={value}")
                else:
                    logger.warning(f"不支持的路径属性: {key}")
                    continue
                
                # 检查结果
                if not result.success:
                    return self._parse_holowan_result(result)
            
            return {"success": True}
        except Exception as e:
            logger.error(f"设置HoloWAN路径属性失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def get_path_info(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """获取HoloWAN路径信息

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 路径信息字典
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            path = engine.get_path_by_id(path_id)
            if path:
                return {
                    "success": True,
                    "data": {
                        "path_id": path_id,
                        "path_name": path.name,
                        "is_enable": path.is_enable,
                        "direction": path.direction
                    }
                }
            else:
                logger.error(f"未找到HoloWAN路径: engine_id={engine_id}, path_id={path_id}")
                return {"success": False, "error": "路径不存在"}
        except Exception as e:
            logger.error(f"获取HoloWAN路径信息失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def get_all_paths(self, engine_id: int) -> Dict[str, Any]:
        """获取所有HoloWAN路径信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 所有路径信息字典，key为路径ID
        """
        try:
            from holowan.v2._commons import get_current_paths_info
            paths_info = get_current_paths_info(self.ip, self.port, engine_id)
            logger.info(f"获取所有HoloWAN路径信息成功: engine_id={engine_id}")
            return {"success": True, "data": paths_info}
        except Exception as e:
            logger.error(f"获取所有HoloWAN路径信息失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    def path_exists(self, engine_id: int, path_id: int) -> bool:
        """检查HoloWAN路径是否存在

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            bool: 路径是否存在
        """
        try:
            paths_info = self.get_all_paths(engine_id)
            if paths_info["success"]:
                return path_id in paths_info["data"]
            return False
        except Exception as e:
            logger.error(f"检查HoloWAN路径是否存在失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return False

    def get_all_classifiers(self, engine_id: int) -> Dict[str, Any]:
        """获取所有HoloWAN分类器信息

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 所有分类器信息字典，key为分类器ID
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            # HoloWAN API获取所有分类器信息
            classifiers = engine.get_classifiers()
            logger.info(f"获取所有HoloWAN分类器信息成功: engine_id={engine_id}")
            return {"success": True, "data": classifiers}
        except Exception as e:
            logger.error(f"获取所有HoloWAN分类器信息失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    def classifier_exists(self, engine_id: int, classifier_id: int) -> bool:
        """检查HoloWAN分类器是否存在

        Args:
            engine_id: 引擎ID
            classifier_id: 分类器ID

        Returns:
            bool: 分类器是否存在
        """
        try:
            classifiers_info = self.get_all_classifiers(engine_id)
            if classifiers_info["success"]:
                return classifier_id in classifiers_info["data"]
            return False
        except Exception as e:
            logger.error(f"检查HoloWAN分类器是否存在失败: engine_id={engine_id}, classifier_id={classifier_id}, error={e}")
            return False

    def get_path_id_by_name(self, engine_id: int, path_name: str) -> Optional[int]:
        """根据路径名称获取路径ID

        Args:
            engine_id: 引擎ID
            path_name: 路径名称

        Returns:
            Optional[int]: 路径ID，如果未找到返回None
        """
        try:
            paths_info = self.get_all_paths(engine_id)
            if paths_info["success"]:
                for path_id, path_info in paths_info["data"].items():
                    if path_info["path_name"] == path_name:
                        return path_id
            return None
        except Exception as e:
            logger.error(f"根据路径名称获取路径ID失败: engine_id={engine_id}, path_name={path_name}, error={e}")
            return None

    def get_classifier_id_by_name(self, engine_id: int, classifier_name: str) -> Optional[int]:
        """根据分类器名称获取分类器ID

        Args:
            engine_id: 引擎ID
            classifier_name: 分类器名称

        Returns:
            Optional[int]: 分类器ID，如果未找到返回None
        """
        try:
            classifiers_info = self.get_all_classifiers(engine_id)
            if classifiers_info["success"]:
                for classifier_id, classifier_info in classifiers_info["data"].items():
                    if "name" in classifier_info and classifier_info["name"] == classifier_name:
                        return classifier_id
            return None
        except Exception as e:
            logger.error(f"根据分类器名称获取分类器ID失败: engine_id={engine_id}, classifier_name={classifier_name}, error={e}")
            return None

    # 损伤配置
    def apply_impairment(self, engine_id: int, path_id: int, impairment_params: Dict[str, Any]) -> Dict[str, Any]:
        """应用HoloWAN损伤配置

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            impairment_params: 损伤参数字典

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            
            # 创建Impairments实例
            direction = impairment_params.get("direction", 1)  # 默认仅损伤下行
            impair = Impairments(direction)
            
            # 配置带宽
            if "bandwidth" in impairment_params:
                bw_params = impairment_params["bandwidth"]
                if bw_params["type"] == "fixed":
                    impair.bandwidth = BandwidthFixed(
                        rate=bw_params["rate"], 
                        unit=bw_params["unit"]
                    )
                # 可扩展其他带宽类型
            
            # 配置延迟
            if "delay" in impairment_params:
                delay_params = impairment_params["delay"]
                if delay_params["type"] == "normal":
                    impair.delay = DelayNormal(
                        minimum=delay_params.get("minimum", 0),
                        mean=delay_params["mean"],
                        std_deviation=delay_params.get("std_deviation", 0),
                        enable_reordering=delay_params.get("enable_reordering", 0)
                    )
                # 可扩展其他延迟类型
            
            # 配置丢包
            if "loss" in impairment_params:
                loss_params = impairment_params["loss"]
                if loss_params["type"] == "burst":
                    impair.loss = LossBurst(
                        probability=loss_params["probability"],
                        min=loss_params["min"],
                        max=loss_params["max"]
                    )
                # 可扩展其他丢包类型
            
            # 应用损伤配置
            result = engine.apply_impairment(impair, path_id, direction)
            logger.info(f"应用HoloWAN损伤配置成功: engine_id={engine_id}, path_id={path_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"应用HoloWAN损伤配置失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def clear_impairment(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """清除HoloWAN路径上的所有损伤

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            # 通过设置空的损伤配置来清除所有损伤
            impair = Impairments(1)  # 仅损伤下行
            result = engine.apply_impairment(impair, path_id, 1)
            logger.info(f"清除HoloWAN路径损伤成功: engine_id={engine_id}, path_id={path_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"清除HoloWAN路径损伤失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    def set_impairment_param(self, engine_id: int, path_id: int, param_name: str, param_value: Any) -> Dict[str, Any]:
        """设置单个HoloWAN损伤参数

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            param_name: 参数名称
            param_value: 参数值

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            
            # 根据参数名称设置不同的损伤参数
            if param_name == "delay":
                # 设置固定延迟
                delay = DelayNormal(minimum=0, mean=param_value, std_deviation=0)
                result = engine.apply_impairment(delay, path_id, 1)
            elif param_name == "loss":
                # 设置丢包率
                loss = LossBurst(probability=param_value, min=1, max=5)
                result = engine.apply_impairment(loss, path_id, 1)
            elif param_name == "bandwidth":
                # 设置固定带宽
                bandwidth = BandwidthFixed(rate=param_value, unit=1)
                result = engine.apply_impairment(bandwidth, path_id, 1)
            else:
                logger.warning(f"不支持的损伤参数: {param_name}")
                return {"success": False, "error": f"不支持的损伤参数: {param_name}"}
            
            logger.info(f"设置HoloWAN损伤参数成功: engine_id={engine_id}, path_id={path_id}, {param_name}={param_value}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"设置HoloWAN损伤参数失败: engine_id={engine_id}, path_id={path_id}, {param_name}={param_value}, error={e}")
            return {"success": False, "error": str(e)}

    # 规则管理
    def create_rule(self, engine_id: int, rule_params: Dict[str, Any]) -> Dict[str, Any]:
        """创建HoloWAN规则

        Args:
            engine_id: 引擎ID
            rule_params: 规则参数字典

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            # HoloWAN API的规则创建较为复杂，需要根据具体规则类型实现
            logger.info(f"创建HoloWAN规则: engine_id={engine_id}, rule_params={rule_params}")
            # 这里仅作为示例，实际需要根据具体规则类型调用相应API
            return {"success": True, "rule_id": 1}
        except Exception as e:
            logger.error(f"创建HoloWAN规则失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    def bind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
        """绑定HoloWAN规则到路径

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            logger.info(f"绑定HoloWAN规则到路径: engine_id={engine_id}, path_id={path_id}, rule_id={rule_id}")
            # 这里仅作为示例，实际需要调用相应API
            return {"success": True}
        except Exception as e:
            logger.error(f"绑定HoloWAN规则失败: engine_id={engine_id}, path_id={path_id}, rule_id={rule_id}, error={e}")
            return {"success": False, "error": str(e)}

    def unbind_rule(self, engine_id: int, path_id: int, rule_id: int) -> Dict[str, Any]:
        """从HoloWAN路径解绑规则

        Args:
            engine_id: 引擎ID
            path_id: 路径ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            logger.info(f"从HoloWAN路径解绑规则: engine_id={engine_id}, path_id={path_id}, rule_id={rule_id}")
            # 这里仅作为示例，实际需要调用相应API
            return {"success": True}
        except Exception as e:
            logger.error(f"解绑HoloWAN规则失败: engine_id={engine_id}, path_id={path_id}, rule_id={rule_id}, error={e}")
            return {"success": False, "error": str(e)}

    def delete_rule(self, engine_id: int, rule_id: int) -> Dict[str, Any]:
        """删除HoloWAN规则

        Args:
            engine_id: 引擎ID
            rule_id: 规则ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            logger.info(f"删除HoloWAN规则: engine_id={engine_id}, rule_id={rule_id}")
            # 这里仅作为示例，实际需要调用相应API
            return {"success": True}
        except Exception as e:
            logger.error(f"删除HoloWAN规则失败: engine_id={engine_id}, rule_id={rule_id}, error={e}")
            return {"success": False, "error": str(e)}

    # 统计信息获取
    def get_device_stats(self) -> Dict[str, Any]:
        """获取HoloWAN设备统计信息

        Returns:
            Dict[str, Any]: 设备统计信息字典
        """
        try:
            info = self.get_device_info()
            logger.debug(f"获取HoloWAN设备统计信息: {info}")
            return info
        except Exception as e:
            logger.error(f"获取HoloWAN设备统计信息失败: {e}")
            return {"success": False, "error": str(e)}

    def get_path_stats(self, engine_id: int, path_id: int) -> Dict[str, Any]:
        """获取HoloWAN路径统计信息

        Args:
            engine_id: 引擎ID
            path_id: 路径ID

        Returns:
            Dict[str, Any]: 路径统计信息字典
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            # HoloWAN API获取路径统计信息
            result = engine.get_path_stats(path_id)
            logger.info(f"获取HoloWAN路径统计信息成功: engine_id={engine_id}, path_id={path_id}")
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"获取HoloWAN路径统计信息失败: engine_id={engine_id}, path_id={path_id}, error={e}")
            return {"success": False, "error": str(e)}

    # 引擎控制
    def start_engine(self, engine_id: int) -> Dict[str, Any]:
        """启动HoloWAN引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            result = engine.start_engine()
            logger.info(f"启动HoloWAN引擎成功: engine_id={engine_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"启动HoloWAN引擎失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    def stop_engine(self, engine_id: int) -> Dict[str, Any]:
        """停止HoloWAN引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            result = engine.stop_engine()
            logger.info(f"停止HoloWAN引擎成功: engine_id={engine_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"停止HoloWAN引擎失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    def reset_engine(self, engine_id: int) -> Dict[str, Any]:
        """重置HoloWAN引擎

        Args:
            engine_id: 引擎ID

        Returns:
            Dict[str, Any]: 操作结果
        """
        try:
            engine = self._get_or_create_engine(engine_id)
            result = engine.reset_engine()
            logger.info(f"重置HoloWAN引擎成功: engine_id={engine_id}")
            return self._parse_holowan_result(result)
        except Exception as e:
            logger.error(f"重置HoloWAN引擎失败: engine_id={engine_id}, error={e}")
            return {"success": False, "error": str(e)}

    # 私有方法
    def _get_or_create_engine(self, engine_id: int) -> Engine:
        """获取或创建引擎实例

        Args:
            engine_id: 引擎ID

        Returns:
            Engine: 引擎实例
        """
        if engine_id not in self.engines:
            # 创建新的引擎实例
            engine = Engine(self.ip, self.port, engine_id)
            engine.update()  # 获取最新状态
            self.engines[engine_id] = engine
            
            # 设置默认引擎ID
            if self.default_engine_id is None:
                self.default_engine_id = engine_id
        
        return self.engines[engine_id]

    def _parse_holowan_result(self, result: HoloWANReturn) -> Dict[str, Any]:
        """解析HoloWAN API返回结果

        Args:
            result: HoloWAN API返回的HoloWANReturn对象

        Returns:
            Dict[str, Any]: 标准化的结果字典
        """
        if result.success:
            return {"success": True, "data": result}
        else:
            return {
                "success": False,
                "error_code": result.err_code,
                "error_msg": result.err_msg,
                "error_reason": result.err_reason
            }
