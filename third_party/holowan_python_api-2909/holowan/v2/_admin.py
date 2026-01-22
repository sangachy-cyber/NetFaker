"""
HoloWAN Network Emulator
Python API

HoloWAN: Admin
"""

import requests
import time
import holowan.v2.utils.my_util as mt
from holowan.v2._holowan_types import (
    IPAddress,
    PortNumber,
    check_parameter,
    HoloWANReturn
)
from holowan.v2._network import EthernetStatus, NetworkConfig
from holowan.v2._preference import Preference

from holowan.v2._commons import (
    _preference_get_url, _preference_set_url,
    _work_port_info_url, network_set_url, _network_info_url,
    get_holowan_info, _sync_system_time_url, _get_system_time_url, _reboot_url
)


class HoloWANAdmin(object):
    """HoloWAN 网络损伤仪 Admin 功能部分。
    这个类实现 Admin 的功能。将 Admin 设计为：设备无关、引擎无关。
    所以该类的方法全为静态方法。可以通过 HoloWAN 的 ip 地址和控制端口的端口号对设备进行控制。
    """

    @staticmethod
    def _get_session_with_login(holowan_ip: IPAddress, holowan_port: PortNumber,
                                username: str = "admin", password: str = "holowan") -> requests.Session:
        """创建并登录 Session，获取 Cookie session_id。
        
        Args:
            holowan_ip: HoloWAN 的 ip 地址
            holowan_port: HoloWAN 的控制端口的端口号
            username: 用户名，默认为 "admin"
            password: 密码，默认为 "holowan"
            
        Returns:
            requests.Session: 已登录的 Session 对象，包含 Cookie
        """
        session = requests.Session()
        try:
            from holowan.v2._commons import _api_protocol
            requestURL = _api_protocol(holowan_ip) + "{0}:{1}/login".format(holowan_ip, holowan_port)
            json_data = {"username": username, "password": password}
            session.post(requestURL, json=json_data, verify=False)
            # Session 会自动保存响应中的 Cookie
        except Exception:
            # 如果登录失败，仍然返回 session（可能服务器不需要登录）
            pass
        return session

    @staticmethod
    @check_parameter
    def set_sync_system_time(holowan_ip: IPAddress, holowan_port: PortNumber,
                             username: str = "admin", password: str = "holowan") -> dict:
        """设置时间同步接口

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            dict: 时间同步响应结果

        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        current_time = int(time.time())
        request_url = _sync_system_time_url(holowan_ip, holowan_port, current_time)
        return session.get(request_url, verify=False).json()

    @staticmethod
    @check_parameter
    def get_sync_system_time(holowan_ip: IPAddress, holowan_port: PortNumber,
                             username: str = "admin", password: str = "holowan") -> dict:
        """获取时间同步接口

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            dict: 时间同步响应结果

        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        request_url = _get_system_time_url(holowan_ip, holowan_port)
        return session.get(request_url, verify=False).json()

    @staticmethod
    @check_parameter
    def reboot(holowan_ip: IPAddress, holowan_port: PortNumber,
               username: str = "admin", password: str = "holowan") -> None:
        """重启设备

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        request_url = _reboot_url(holowan_ip, holowan_port)
        session.get(request_url, verify=False)

    @staticmethod
    @check_parameter
    def get_preferences(holowan_ip: IPAddress, holowan_port: PortNumber,
                       username: str = "admin", password: str = "holowan") -> Preference:
        """获取 HoloWAN 的 preference 首选项配置。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            Preference: Preference 对象。包含设备的首选项配置信息。
        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        requestURL = _preference_get_url(holowan_ip, holowan_port)
        return Preference(session.get(requestURL, verify=False).text)

    @staticmethod
    @check_parameter
    def set_preferences(holowan_ip: IPAddress, holowan_port: PortNumber, preference: Preference,
                       username: str = "admin", password: str = "holowan"):
        # TODO:返回值处理
        """Set the custom preference of the HoloWAN network emulator.

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            preference (Preference): Preference对象。包含首选项配置。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            HoloWANReturn: HoloWAN 返回值对象。
        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        preference_xml_str = preference.xml
        url = _preference_set_url(holowan_ip, holowan_port)
        return HoloWANReturn(mt.post_original_api(url, preference_xml_str, session))

    @staticmethod
    @check_parameter
    def get_ethernet_status(holowan_ip: IPAddress, holowan_port: PortNumber,
                           username: str = "admin", password: str = "holowan") -> EthernetStatus:
        """获取 HoloWAN 的 Ethernet 状态信息。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            EthernetStatus: EthernetStatus 对象。包含 Ethernet 状态信息。
        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        url = _work_port_info_url(holowan_ip, holowan_port)
        return EthernetStatus(session.get(url, verify=False).text)

    @staticmethod
    @check_parameter
    def get_network_configuration(holowan_ip: IPAddress, holowan_port: PortNumber,
                                  username: str = "admin", password: str = "holowan") -> NetworkConfig:
        """获取 HoloWAN 的网络配置信息。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            NetworkConfig: NetworkConfig 对象，包含设备网络配置信息。
        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        url = _network_info_url(holowan_ip, holowan_port)
        return NetworkConfig(session.get(url, verify=False).text)

    @staticmethod
    @check_parameter
    def set_network_configuration(holowan_ip: IPAddress, holowan_port: PortNumber, network_config: NetworkConfig,
                                  username: str = "admin", password: str = "holowan"):
        # TODO:返回值处理
        """设置 HoloWAN 的网络配置。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            network_config (NetworkConfig): NetworkConfig 对象。包含网络配置。
            username (str): 登录用户名，默认为 "admin"。
            password (str): 登录密码，默认为 "holowan"。

        Returns:
            HoloWANReturn: A object including the returns info.
        """
        session = HoloWANAdmin._get_session_with_login(holowan_ip, holowan_port, username, password)
        network_cfg_xml_str = network_config.xml
        url = network_set_url(holowan_ip, holowan_port)
        return HoloWANReturn(mt.post_original_api(url, network_cfg_xml_str, session))
