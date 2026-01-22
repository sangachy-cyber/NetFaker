"""
HoloWAN Network Emulator
Python API

engine
"""
from typing import Any

import requests
import holowan.v2.utils.my_util as mt
import holowan.v2.utils.xml_util as xt
from holowan.v2 import IPAddress, PortNumber, EngineID, HoloWANReturn
from holowan.v2 import check_parameter, PathID
from holowan.v2.engine.classifier._classifier import PacketClassifier
from holowan.v2.engine.path._path import Path
from holowan.v2.engine._path_crud import PathCRUD
from holowan.v2.engine._container import Sequential
from holowan.v2.engine.path._route_select import RouteSelect
from holowan.v2._commons import (
    _reset_engine_url, _hold_engine_url, _emulator_cfg_url, _set_engine_name_url,
    _clf_cfg_info_url, _reset_path_url, _add_path_url, _path_cfg_info_url, _csv_data_url,
    _static_info_url, _start_engine_url, _stop_engine_url
)

from holowan.v2.engine.classifier._rule_base import Rule
from typing import overload
from holowan.v2.engine.path._bandwidth import Bandwidth
from holowan.v2.engine.path._bg_utilization import BackgroundUtilization
from holowan.v2.engine.path._corruption import Corruption
from holowan.v2.engine.path._delay import Delay
from holowan.v2.engine.path._duplication import Duplication
from holowan.v2.engine.path._frame_overhead import FrameOverhead
from holowan.v2.engine.path._loss import Loss
from holowan.v2.engine.path._modify import Modify
from holowan.v2.engine.path._mtu import MTU
from holowan.v2.engine.path._queue_limit import QueueLimit
from holowan.v2.engine.path._reordering import Reordering
from holowan.v2.engine.path._path import Impairments
from holowan.v2.engine.path._impairment_base import ImpairmentTypes

_CLEAR_PASSWORD: int = 4123456789


class Engine(object):
    """HoloWAN Engine
    Engine主要包含报文分类器(PacketClassifier) 和 paths(Sequential)
    paths 是一个 Sequential 容器，容器内的每一个元素都是一个 Path 对象来表示一条 path

    Args:
        holowan_ip (IPAddress): HoloWAN 的 ip 地址。
        holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
        engine_id (EngineID): 引擎 id。
        proxy (Any): 代理。 默认为 None。
        username (str): 登录用户名，默认为 "admin"。
        password (str): 登录密码，默认为 "holowan"。

    Notes:
        构造函数会自动执行登录操作，获取 Cookie session_id。
        所有后续的 API 调用都会自动携带该 Cookie。

    Examples:
        >>> engine = Engine(holowan_ip, holowan_port, engine_id)
        >>> classifier = engine.packet_classifier
        >>> classifier.port1 = Sequential(
            rule1,rul2,rule3,rule4,...
        )
        >>> result = engine.apply_classifier_changes()
        # 如果需要应用一个全新的 PacketClassifier 到当前引擎
        >>> classifier = PacketClassifier()
        >>> classifier.port1 = Sequential(
        >>>    rule1,rul2,rule3,rule4,...
        >>> )
        >>> engine.packet_classifier = classifier
        >>> engine.apply_classifier_changes()


    """
    _http_proxy: Any
    _paths: Sequential
    _paths_info: dict

    @check_parameter
    def __init__(self, holowan_ip: IPAddress, holowan_port: PortNumber, engine_id: EngineID, proxy: Any = None, 
                 username: str = "admin", password: str = "holowan"):
        if proxy == None:
            self._http_proxy = {}
        else:
            self._http_proxy = proxy
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port
        
        # Session 对象用于保存 Cookie
        self.session = requests.Session()
        if self._http_proxy:
            self.session.proxies = self._http_proxy
        
        # 在调用任何 API 之前先执行登录
        self._username = username
        self._password = password
        self._login(username, password)
        
        _available_engines = self._get_available_engines()
        if engine_id not in _available_engines:
            raise ValueError("The Engine with id:{0} is not available. Available Engine id(s):{1}.".format(engine_id,
                                                                                                           _available_engines))
        self._engine_id = engine_id

        self._path_curd = PathCRUD()

        self._classifier: PacketClassifier = None
        self._paths_info = {}

        self._paths = Sequential()
        self.update()

    # ===========================================================
    @property
    def holowan_ip(self) -> IPAddress:
        return self._holowan_ip

    # ===========================================================

    @property
    def holowan_port(self) -> PortNumber:
        return self._holowan_port

    # ===========================================================

    @property
    def engine_id(self) -> EngineID:
        return self._engine_id

    @property
    def paths_info(self) -> dict:
        return self._paths_info

    # ===========================================================

    # ==================== engine op start ======================
    def _login(self, username: str = "admin", password: str = "holowan") -> HoloWANReturn:
        """登录接口（私有方法），获取 Cookie session_id。
        
        Args:
            username: 用户名，默认为 "admin"
            password: 密码，默认为 "holowan"
            
        Returns:
            HoloWANReturn: 登录响应结果，成功后会保存 Cookie session_id 到 Session 中
        """
        try:
            from holowan.v2._commons import _api_protocol
            requestURL = _api_protocol(self._holowan_ip) + "{0}:{1}/login".format(self._holowan_ip, self._holowan_port)
            json_data = {"username": username, "password": password}
            response = self.session.post(requestURL, json=json_data, verify=False)
            # Session 会自动保存响应中的 Cookie
            return HoloWANReturn(response.text)
        except requests.exceptions.ConnectionError:
            return HoloWANReturn(r'{"errCode":"-200","errMsg":"ConnectionError","errReason":"Failed to establish a new connection"}')
        except Exception as e:
            return HoloWANReturn(r'{"errCode":"-500","errMsg":"ERROR","errReason":"' + str(e) + '"}')

    def login(self, username: str = "admin", password: str = "holowan") -> HoloWANReturn:
        """登录接口（公共方法），用于重新登录或更新 Cookie。
        
        Args:
            username: 用户名，默认为 "admin"
            password: 密码，默认为 "holowan"
            
        Returns:
            HoloWANReturn: 登录响应结果，成功后会保存 Cookie session_id 到 Session 中
        """
        return self._login(username, password)

    def emulation_on(self) -> HoloWANReturn:
        """开启引擎。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        return self._start_engine_emulation()

    def emulation_off(self):
        """关闭引擎。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        return self._close_engine_emulation()

    def check_lock(self) -> HoloWANReturn:
        """检查引擎是否上锁。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _hold_engine_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id, 0, 0
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def unlock(self, password: int) -> HoloWANReturn:
        """解锁引擎。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            pasword: 密码。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _hold_engine_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id, password, 0
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def set_new_password(self, password: int, new_password: int) -> HoloWANReturn:
        """设置新密码。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            pasword: 旧密码。
            new_password: 新密码。
            
        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _hold_engine_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id, password, new_password
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def clear_password(self) -> HoloWANReturn:
        """清除密码。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _hold_engine_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id, 0, _CLEAR_PASSWORD
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def reset_engine(self) -> HoloWANReturn:
        """重置引擎。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _reset_engine_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def save_engine(self) -> HoloWANReturn:
        """保存当前引擎配置为开机默认配置。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _emulator_cfg_url(
            self._holowan_ip, self._holowan_port
        )
        HoloWAN_information_xml = self._get_holowan_info()
        return HoloWANReturn(mt.post_original_api(url, HoloWAN_information_xml, self.session))

    # ==================== engine op end ========================

    def _get_holowan_info(self) -> str:
        """获取 HoloWAN 设备信息（使用 session 携带 Cookie）
        
        Returns:
            str: HoloWAN 设备信息的 XML 字符串
        """
        url = _static_info_url(self._holowan_ip, self._holowan_port)
        return self.session.get(url, verify=False).text

    def _get_current_paths_info(self) -> dict:
        """获取当前引擎的 paths 信息（使用 session 携带 Cookie）
        
        Returns:
            dict: paths 信息字典，格式为 {pathID: {"path_name": str, "is_enable": str}}
        """
        path_dic = {}
        info_str = self._get_holowan_info()
        engine_nodes = xt.xmlString_to_Object(info_str)
        for node in engine_nodes.findall("e"):
            each_engine_id = xt.get_node(node, "ei").text
            if int(each_engine_id) == self._engine_id:
                path_nodes = xt.get_nodes(node, "./ep/p")
                for path_node in path_nodes:
                    pathID = int(xt.get_node(path_node, "./pi").text)
                    path_name = xt.get_node(path_node, "./pn").text
                    is_enable = xt.get_node(path_node, "./ie").text
                    path_dic[pathID] = {
                        "path_name": path_name,
                        "is_enable": is_enable
                    }
        return path_dic

    def _get_available_engines(self) -> list:
        """获取可用的引擎 ID 列表（使用 session 携带 Cookie）
        
        Returns:
            list: 可用的引擎 ID 列表
        """
        response = self._get_holowan_info()
        engine_nodes = xt.xmlString_to_Object(response)
        idls = []
        for node in engine_nodes.findall("e"):
            each_engine_id = int(xt.get_node(node, "ei").text)
            idls.append(each_engine_id)
        return idls

    def _start_engine_emulation(self) -> HoloWANReturn:
        """开启引擎（使用 session 携带 Cookie）
        
        Returns:
            HoloWANReturn: HoloWAN 返回值
        """
        url = _start_engine_url(self._holowan_ip, self._holowan_port, self._engine_id)
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def _close_engine_emulation(self) -> HoloWANReturn:
        """关闭引擎（使用 session 携带 Cookie）
        
        Returns:
            HoloWANReturn: HoloWAN 返回值
        """
        url = _stop_engine_url(self._holowan_ip, self._holowan_port, self._engine_id)
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def update(self) -> None:
        """从 HoloWAN 获取当前引擎最新的报文分类器和 paths 的配置

        """
        self._login(self._username, self._password)
        self._get_packet_classifier_info()
        self._paths_info = self._get_current_paths_info()
        self._initialize_paths()

    # ==================== classifier start =====================
    @property
    def packet_classifier(self) -> PacketClassifier:
        """获取当前引擎的报文分类器

        Returns:
            PacketClassifier 对象

        """
        return self._classifier

    @packet_classifier.setter
    def packet_classifier(self, value: PacketClassifier) -> None:
        self._classifier = value
        self._classifier.attachTo(self._engine_id)

    def _get_packet_classifier_info(self) -> None:
        # TODO: exception
        url = _clf_cfg_info_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id
        )
        response_str = self.session.get(url, verify=False).text
        self._classifier = PacketClassifier(response_str)

    def apply_classifier_changes(self):
        """应用报文分类器到 HoloWAN。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        self._classifier.update()
        xml_str = self._classifier.xml
        result = HoloWANReturn(mt.post_original_api(
            _emulator_cfg_url(self._holowan_ip, self._holowan_port),
            xml_str,
            self.session
        ))
        self.update()
        return result

    # ==================== classifier end =======================

    # ====================== path start =========================

    def get_path_by_id(self, path_id: PathID) -> Path:
        """通过 id 获取一条 path

        Notes:
            此方法不具备通讯能力，所以获取的时上一次 Engine.update() 后的 path

        Args:
            path_id: 带获取的 path 的 id

        Returns:

        """
        self.update()
        return self._paths.get_item_by_key(str(path_id))

    @check_parameter
    def reset_path(self, path_id: PathID) -> HoloWANReturn:
        """重置指定路径。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            path_id (PathID): 需要重置的path id。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _reset_path_url(
            self._holowan_ip, self._holowan_port, self._engine_id, path_id
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def add_path(self) -> PathID:
        """添加一条 path，该路径为默认配置。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        url = _add_path_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id
        )
        result = HoloWANReturn(self.session.get(url, verify=False).text)
        if result:
            ori_data = eval(result.data)
            return int(ori_data['data']['path_id'])
        return -1

    def remove_path(self, path_id: PathID, path_name: str) -> HoloWANReturn:
        """删除一条path。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            path_id (PathID): 待删除 path 的 path id。
            path_name (str): 待删除 path 的名称。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        self._path_curd.path_id = path_id
        self._path_curd.path_name = path_name
        self._path_curd.modify_switch = 1
        self._path_curd.engine_id = self._engine_id
        self._path_curd.if_enable = 1
        xml_str = self._path_curd.xml
        return HoloWANReturn(mt.post_original_api(
            _emulator_cfg_url(self._holowan_ip, self._holowan_port),
            xml_str,
            self.session
        ))

    def enable_path(self, path_id: PathID, path_name: str) -> HoloWANReturn:
        """启用path。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            path_id (PathID): 待启用 path 的 path id。
            path_name (str): 待启用 path 的名称。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        self._path_curd.modify_switch = 3
        self._path_curd.engine_id = self._engine_id
        self._path_curd.path_id = path_id
        self._path_curd.path_name = path_name
        self._path_curd.if_enable = 2

        xml_str = self._path_curd.xml
        return HoloWANReturn(mt.post_original_api(
            _emulator_cfg_url(self._holowan_ip, self._holowan_port),
            xml_str,
            self.session
        ))

    def disable_path(self, path_id: PathID, path_name: str) -> HoloWANReturn:
        """禁用path。

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            path_id (PathID): 待禁用path 的path id。
            path_name (str): 待禁用path 的名称。

        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        self._path_curd.modify_switch = 3
        self._path_curd.engine_id = self._engine_id
        self._path_curd.path_id = path_id
        self._path_curd.path_name = path_name
        self._path_curd.if_enable = 1

        xml_str = self._path_curd.xml
        return HoloWANReturn(mt.post_original_api(
            _emulator_cfg_url(self._holowan_ip, self._holowan_port),
            xml_str,
            self.session
        ))

    def is_path_enable(self, path_id: PathID) -> bool:
        # only update paths info
        # To know whether a path is enabled or not is not need to update other info.
        self._paths_info = self._get_current_paths_info()
        return self._paths_info[path_id]["is_enable"]

    # def enable_impairment(self,path_id:int,direction:int,impair_type:str,enable:bool)->HoloWANReturn:
    #     path = self.get_path_by_id(path_id)
    #     impair_ls = []
    #     if direction == 1:
    #         impair_ls.append(path.l2r)
    #     elif direction == 2:
    #         impair_ls.append(path.r2l)
    #     elif direction == 3:
    #         impair_ls.append(path.l2r)
    #         impair_ls.append(path.r2l)
    #     else:
    #         raise ValueError("Argument direction must be 1, 2 or 3, got {got!r}, value {value!r}".format(
    #             got=type(direction), value=direction
    #         ))
    #
    #     for impairs in impair_ls:
    #         impairs.enable_impairment(impair_type,enable)
    #
    #     return self.apply_path_configuration(path)

    def apply_path_configuration(self, path: Path) -> HoloWANReturn:
        """应用 id 为 path_id 的 path 配置

        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            path_id: 待应用配置的 path_id

        Returns:
            HoloWANReturn
        """

        path.update()
        xml_str = path.xml
        result = HoloWANReturn(mt.post_original_api(
            _emulator_cfg_url(self._holowan_ip, self._holowan_port),
            xml_str,
            self.session
        ))
        self.update()
        return result

    def _get_path_configuration(self, path_id: PathID) -> str:
        url = _path_cfg_info_url(
            self._holowan_ip, self._holowan_port, self._engine_id, path_id
        )
        response_str = self.session.get(url, verify=False).text
        # self._path = Path(response_str)
        return response_str

    def _initialize_paths(self):
        for id in self._paths_info.keys():
            path = Path(self._get_path_configuration(id))
            self._paths.add_item(str(id), path)

    # ====================== path end ===========================

    # ====================== stats start ========================
    def stats_create_csv(self, path_id: PathID, file_path: str) -> HoloWANReturn:
        self.emulation_off()
        url1 = _csv_data_url(
            self._holowan_ip, self._holowan_port, self._engine_id,
            path_id
        )
        response = HoloWANReturn(self.session.get(url1, verify=False).text)
        if response:
            url2 = _csv_data_url(
                self._holowan_ip, self._holowan_port, self._engine_id,
                path_id
            )
            response2 = self.session.get(url2, verify=False)
            with open(file_path, "w") as f:
                f.write(response2.text)
        return response

    # ====================== stats end ==========================

    # ================== route select start =====================
    def set_route_select(self, path_id: PathID, route_select: RouteSelect):
        path: Path = self._paths.get_item_by_key(str(path_id))
        path.l2r = route_select.l2r
        path.r2l = route_select.r2l
        path.update()
        return self.apply_path_configuration(path)

    # ================== route select end =======================

    def apply_rule_to_classifier(self, rule: Rule, port: int) -> HoloWANReturn:
        """

        Args:
            rule: 单个规则
            port: 待添加规则的 port

        Returns:
            HoloWANReturn
        """
        if port % 2 == 1:  # port 1,3,5,7,...
            self.packet_classifier.port1.append(rule)
        else:  # port 2,4,6,8,...
            self.packet_classifier.port2.append(rule)
        return self.apply_classifier_changes()

    def set_engine_name(self, engine_id: int = 1, engine_name: str = "Engine 1") -> HoloWANReturn:
        """设置引擎名字
        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            holowan_ip: holowan的ip
            holowan_port:  holowan的端口
            engine_id: holowan的engine id
            engine_name: holowan的engine_name 默认长度12
        Examples:
            self.set_engine_name(engine_id=1, engine_name="new Engine 1")
        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        if len(engine_name) <= 12:
            return HoloWANReturn(mt.get_original_api(
                _set_engine_name_url(self._holowan_ip, self._holowan_port, engine_id, engine_name),
            ))
        else:
            return HoloWANReturn('{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"engine_name error!"}')

    def set_path_name(self, path_id: int = 1, path_name: str = "new PATH 1"):
        """设置PATH名字
        Notes:
            此方法具备通讯能力，将改变 HoloWAN 的配置

        Args:
            holowan_ip: holowan的ip
            holowan_port:  holowan的端口
            path_id: holowan的path id
            path_name: 新的path name
        Examples:
            set_holowan_path_name(path_id=1, path_name="new PATH 1")
        Returns:
            HoloWANReturn: HoloWAN 返回值。
        """
        path = self.get_path_by_id(path_id)
        path.path_name = path_name
        return self.apply_path_configuration(path)

    @overload
    def apply_impairment(self, impairment: Bandwidth, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: BackgroundUtilization, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: QueueLimit, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: MTU, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Modify, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: FrameOverhead, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Delay, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Loss, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Reordering, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Corruption, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairment: Duplication, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    @overload
    def apply_impairment(self, impairments: Impairments, path_id: PathID, direction: int) -> HoloWANReturn:
        ...

    def apply_impairment(self, impairment, path_id: PathID, direction: int) -> HoloWANReturn:
        """

        Args:
            impairment: 可以是单个损伤，也可以是一组损伤（Impairments）
            path_id: 待设置损伤的 path ID
            direction: 待设置损伤的 path 的方向

        Returns:
            HoloWANReturn
        """
        path = self.get_path_by_id(path_id)
        if isinstance(impairment, Impairments):
            if direction == 1:
                path.l2r = Impairments(direction=direction).clone_from(src=impairment)
            elif direction == 2:
                path.r2l = Impairments(direction=direction).clone_from(src=impairment)
            elif direction == 3:
                path.l2r = Impairments(direction=1).clone_from(src=impairment)
                path.r2l = Impairments(direction=2).clone_from(src=impairment)
            else:
                raise ValueError("Argument direction must be 1, 2 or 3, got {got!r}, value {value!r}".format(
                    got=type(direction), value=direction
                ))
        else:
            impair_ls = []
            if direction == 1:
                impair_ls.append(path.l2r)
            elif direction == 2:
                impair_ls.append(path.r2l)
            elif direction == 3:
                impair_ls.append(path.l2r)
                impair_ls.append(path.r2l)
            else:
                raise ValueError("Argument direction must be 1, 2 or 3, got {got!r}, value {value!r}".format(
                    got=type(direction), value=direction
                ))

            for impairs in impair_ls:
                if isinstance(impairment, Bandwidth):
                    impairs.bandwidth = impairment
                elif isinstance(impairment, BackgroundUtilization):
                    impairs.background_utilization = impairment
                elif isinstance(impairment, QueueLimit):
                    impairs.queue_limit = impairment
                elif isinstance(impairment, MTU):
                    impairs.mtu = impairment
                elif isinstance(impairment, Modify):
                    impairs.modify = impairment
                elif isinstance(impairment, FrameOverhead):
                    impairs.frame_overhead = impairment
                elif isinstance(impairment, Delay):
                    impairs.delay = impairment
                elif isinstance(impairment, Loss):
                    impairs.loss = impairment
                elif isinstance(impairment, Reordering):
                    impairs.reordering = impairment
                elif isinstance(impairment, Corruption):
                    impairs.corruption = impairment
                elif isinstance(impairment, Duplication):
                    impairs.duplication = impairment
                else:
                    raise ValueError("Unknown impairment type: {0}.".format(type(impairment)))

        return self.apply_path_configuration(path)
