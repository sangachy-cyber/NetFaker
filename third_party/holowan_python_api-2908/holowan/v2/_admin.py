"""
HoloWAN Network Emulator
Python API

HoloWAN: Admin
"""

import time

import requests

import holowan.v2.utils.my_util as mt
from holowan.v2._commons import (
    _network_info_url,
    _preference_get_url,
    _preference_set_url,
    _work_port_info_url,
    network_set_url,
)
from holowan.v2._holowan_types import (
    HoloWANReturn,
    IPAddress,
    PortNumber,
    check_parameter,
)
from holowan.v2._network import EthernetStatus, NetworkConfig
from holowan.v2._preference import Preference


class HoloWANAdmin(object):
    """HoloWAN 网络损伤仪 Admin 功能部分。
    这个类实现 Admin 的功能。将 Admin 设计为：设备无关、引擎无关。
    所以该类的方法全为静态方法。可以通过 HoloWAN 的 ip 地址和控制端口的端口号对设备进行控制。
    """

    @staticmethod
    @check_parameter
    def set_sync_system_time(holowan_ip: IPAddress, holowan_port: PortNumber) -> dict:
        """设置时间同步接口

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        """
        current_time = int(time.time())
        request_url = "http://{0}:{1}/sync_system_time?time={2}".format(holowan_ip, holowan_port, current_time)
        return requests.get(request_url, verify=False).json()

    @staticmethod
    @check_parameter
    def get_sync_system_time(holowan_ip: IPAddress, holowan_port: PortNumber) -> dict:
        """获取时间同步接口

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        """
        request_url = "http://{0}:{1}/get_system_time".format(holowan_ip, holowan_port)
        return requests.get(request_url, verify=False).json()

    @staticmethod
    @check_parameter
    def reboot(holowan_ip: IPAddress, holowan_port: PortNumber) -> None:
        """重启设备

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        """
        request_url = "http://{0}:{1}/reboot".format(holowan_ip, holowan_port)
        requests.get(request_url)

    @staticmethod
    @check_parameter
    def get_preferences(holowan_ip: IPAddress, holowan_port: PortNumber) -> Preference:
        """获取 HoloWAN 的 preference 首选项配置。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        Returns:
            Preference: Preference 对象。包含设备的首选项配置信息。
        """
        requestURL = _preference_get_url(holowan_ip, holowan_port)
        return Preference(requests.get(requestURL).text)

    @staticmethod
    @check_parameter
    def set_preferences(holowan_ip: IPAddress, holowan_port: PortNumber, preference: Preference):
        # TODO:返回值处理
        """Set the custom preference of the HoloWAN network emulator.

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            preference (Preference): Preference对象。包含首选项配置。

        Returns:
            HoloWANReturn: HoloWAN 返回值对象。
        """
        preference_xml_str = preference.xml
        url = _preference_set_url(holowan_ip, holowan_port)
        return HoloWANReturn(mt.post_original_api(url, preference_xml_str))

    @staticmethod
    @check_parameter
    def get_ethernet_status(holowan_ip: IPAddress, holowan_port: PortNumber) -> EthernetStatus:
        """获取 HoloWAN 的 Ethernet 状态信息。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        Returns:
            EthernetStatus: EthernetStatus 对象。包含 Ethernet 状态信息。
        """
        url = _work_port_info_url(holowan_ip, holowan_port)
        return EthernetStatus(requests.get(url).text)

    @staticmethod
    @check_parameter
    def get_network_configuration(holowan_ip: IPAddress, holowan_port: PortNumber) -> NetworkConfig:
        """获取 HoloWAN 的网络配置信息。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。

        Returns:
            NetworkConfig: NetworkConfig 对象，包含设备网络配置信息。
        """
        url = _network_info_url(holowan_ip, holowan_port)
        return NetworkConfig(requests.get(url).text)

    @staticmethod
    @check_parameter
    def set_network_configuration(holowan_ip: IPAddress, holowan_port: PortNumber, network_config: NetworkConfig):
        # TODO:返回值处理
        """设置 HoloWAN 的首选项。

        Args:
            holowan_ip (IPAddress): HoloWAN 的 ip 地址。
            holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
            network_config (NetworkConfig): NetworkConfig 对象。包含网络配置。

        Returns:
            HoloWANReturn: A object including the returns info.
        """
        network_cfg_xml_str = network_config.xml
        url = network_set_url(holowan_ip, holowan_port)
        return HoloWANReturn(mt.post_original_api(url, network_cfg_xml_str))
