"""
HoloWAN Network Emulator
Python API
"""

from typing import List

from holowan.v2._holowan_types import (
    HoloWANConfigXML,
    check_parameter,
    IPv4Address,
    IPv6Address
)
from holowan.v2._xml_holder import XMLHolder
from holowan.v2.utils import xml_util as xt

ETHERNET_STATUS_XML_FORMAT = """
<?xml version="1.0" encoding="utf-8"?>              /* XML 默认Header */
<holowan_admin_ethernet_config_info>                /* Root节点（必须） */
    <port>                                          /* 每一个工作口的状态信息节点 */
        <port_id></port_id>                         /* 工作口ID编号 */
        <speed_duplex>linkdown</speed_duplex>       /* 工作口速率信息 */
        <flow_control>linkdown</flow_control>       /* 工作口流控信息 */
    </port>
</holowan_admin_ethernet_config_info>
"""

# ================== Ethernet Status Key Begin=======================
PORT: str = r"port"
PORT_ID: str = r"port_id"
MAX_SPEED: str = r"max_speed"
SPEED_DUPLEX: str = r"speed_duplex"
FLOW_CONTROL: str = r"flow_control"
# ================== Ethernet Status Key End ========================

NETWORK_CONFIG_XML_FORMAT = """
<?xml version="1.0" encoding="UTF-8"?>
<holowan_admin_network_config_info>
    <network_settings>
        <hostname></hostname>
        <domain></domain>
        <ipaddr></ipaddr>
        <netmask></netmask>
        <ipv6addr>
            <link></link>
            <global></global>
        </ipv6addr>
        <gateway></gateway>
        <dhcp_switch></dhcp_switch>
        <dhcpv6_switch></dhcpv6_switch>
        <dnsserver1></dnsserver1>
        <dnsserver2></dnsserver2>
        <ntpserver1></ntpserver1>
        <ntpserver2></ntpserver2>
        <mac></mac>
    </network_settings>
</holowan_admin_network_config_info>
"""

# ================== Network Config Key Begin =======================
TAG_NAME: str = r"holowan_admin_network_config_info"
NETWORK_SETTINGS_HEADER = r"network_settings/"

HOSTNAME: str = r"hostname"
DOMAIN: str = r"domain"
IPV4_ADDRESS: str = r"ipaddr"
NETMASK: str = r"netmask"
IPV6_LINK: IPv6Address = r"ipv6addr/link"
IPV6_GLOBAL: IPv6Address = r"ipv6addr/global"
GATEWAY: str = r"gateway"
IPV4_DHCP_SWITCH: str = r"dhcp_switch"
IPV6_DHCP_SWITCH: str = r"dhcpv6_switch"
DNS_SERVER1: str = r"dnsserver1"
DNS_SERVER2: str = r"dnsserver2"
NTP_SERVER1: str = r"ntpserver1"
NTP_SERVER2: str = r"ntpserver2"
MACADDRESS: str = r"mac"


# ================== Network Config Key End =========================

class PortStatus(object):
    """网口的状态信息
    此类设计为解析、组织网口的状态信息。用作 EthernetStatus 的组成部分，不涉及任何对设备的控制。
    类中所有的属性均为只读权限。

    Args:
        port_id (int): The port id.
        max_speed (float): The max speed of the traffic through the port.
        speed_duplex (str): The speed duplex.
        flow_control (str): The flow control.

    """
    _status_parameters = {
        PORT_ID: int,
        MAX_SPEED: float,
        SPEED_DUPLEX: str,
        FLOW_CONTROL: str,
    }

    @check_parameter
    def __init__(self, port_id: int, max_speed: float, speed_duplex: str, flow_control: str) -> None:
        self._status_parameters = {}
        self._status_parameters[PORT_ID] = port_id
        self._status_parameters[MAX_SPEED] = max_speed
        self._status_parameters[SPEED_DUPLEX] = speed_duplex
        self._status_parameters[FLOW_CONTROL] = flow_control

    @property
    def port_id(self) -> int:
        """The port id.

        Notes:
            readonly

        """
        return self._status_parameters[PORT_ID]

    @property
    def max_speed(self) -> float:
        """The max speed of the traffic through the port.

        Notes:
            readonly

        """
        return self._status_parameters[MAX_SPEED]

    @property
    def speed_duplex(self) -> str:
        """The speed duplex.

        Notes:
            readonly

        """
        return self._status_parameters[SPEED_DUPLEX]

    @property
    def flow_control(self) -> str:
        """The flow control.

        Notes:
            readonly

        """
        return self._status_parameters[FLOW_CONTROL]

    def __str__(self):
        res = "Port {0}:".format(self._status_parameters[PORT_ID])
        res += self._status_parameters.__str__()
        return res


class EthernetStatus(object):
    """HoloWAN 的 Ethernet 状态信息。
    这个类设计为解析、组织HoloWAN 的 Ethernet 状态信息，不涉及任何对设备的配置。
    EthernetStatus 内包含设备多个网口的状态信息，形式为列表：List[PortStatus]。
    类中属性均为只读权限。

        Args:
            status_xml (HoloWANConfigXML, optional): Ethernet status 的 xml 字符串。默认为 None。
                通过传入的 xml 字符串构造一个 EthernetStatus 对象。

        Attributes:
            port_status: Readonly property. 设备所有网口的状态信息。每个 PortStatus 包含对应网口的状态信息。
    
        Examples::
            >>> status = HoloWANAdmin.get_ethernet_status(holowan_ip=holowan_ip,
                                              holowan_port=holowan_port)
            >>> port1_status = status.port_status[1]
    """
    _status_list: dict
    _status_xml: HoloWANConfigXML = None

    @check_parameter
    def __init__(self, status_xml: HoloWANConfigXML = None):
        if status_xml != None:
            # TODO: 异常处理
            self._status_xml = status_xml
            root = xt.xmlString_to_Object(status_xml)
            port_nodes = root.findall(PORT)
            self._status_list = {}
            for port in port_nodes:
                port_id = int(port.findtext(PORT_ID))
                max_speed = float(port.findtext(MAX_SPEED))
                speed_duplex = str(port.findtext(SPEED_DUPLEX))
                flow_control = str(port.findtext(FLOW_CONTROL))
                self._status_list[port_id] = PortStatus(port_id, max_speed, speed_duplex, flow_control)

    def __len__(self) -> int:
        return len(self._status_list)

    def __str__(self):
        str = ""
        for _, v in self._status_list.items():
            str += v.__str__() + "\n"
        return str

    @property
    def port_status(self) -> dict:
        """设备所有网口的状态信息。
        每个 PortStatus 包含对应网口的状态信息。

        Notes:
            readonly
        """
        return self._status_list


class NetworkConfig(XMLHolder):
    """HoloWAN 的网络配置。
    此类设计为解析、组织 HoloWAN 的网络配置信息。
    并且，可以通过设置该类对象的属性，更改设备的网络配置信息，如 ip 地址。

        Note:
            更改属性并不会立即改变 HoloWAN 的配置，该类不具备与设备进行通讯的能力。

        Args:
            network_cfg_xml (HoloWANConfigXML, optional): 网络配置信息 xml 字符串。默认为 None。

        Attributes:
            hostname(str): The hostname.
            mac(str): The MAC address of the device.
            ipv4_address(IPv4Address): The ipv4 address of the HoloWAN.
            netmask(str): The netmask.
            gateway(str): The gateway.
            ipv4_dhcp_switch(str): The ipv4 DHCP switch.
            ipv6_global_address(str): The ipv6 address.
            ipv6_link_address(str): The ipv6 link address.
            ipv6_dhcp_switch(str): The ipv6 DHCP switch.
        
        EXamples:
            >>> network_config = HoloWANAdmin.get_network_configuration(holowan_ip,holowan_port)
            >>> network_config.ipv4_address = "192.168.1.199"
            >>> result = HoloWANAdmin.set_network_configuration(holowan_ip,holowan_port,network_config)
    """

    @check_parameter
    def __init__(self, network_cfg_xml: HoloWANConfigXML = None) -> None:
        self._network_cfg_parameters = {
            HOSTNAME: str,
            DOMAIN: str,
            IPV4_ADDRESS: IPv4Address,
            NETMASK: str,
            IPV6_LINK: IPv6Address,
            IPV6_GLOBAL: str,
            GATEWAY: str,
            IPV4_DHCP_SWITCH: int,
            IPV6_DHCP_SWITCH: int,
            DNS_SERVER1: str,
            DNS_SERVER2: str,
            NTP_SERVER1: str,
            NTP_SERVER2: str,
            MACADDRESS: str
        }
        if network_cfg_xml != None:
            # TODO: 异常处理
            root = xt.xmlString_to_Object(network_cfg_xml)
            self._network_cfg_parameters[HOSTNAME] = root.findtext(NETWORK_SETTINGS_HEADER + HOSTNAME)
            self._network_cfg_parameters[DOMAIN] = root.findtext(NETWORK_SETTINGS_HEADER + DOMAIN)
            self._network_cfg_parameters[IPV4_ADDRESS] = root.findtext(NETWORK_SETTINGS_HEADER + IPV4_ADDRESS)
            self._network_cfg_parameters[NETMASK] = root.findtext(NETWORK_SETTINGS_HEADER + NETMASK)
            self._network_cfg_parameters[IPV6_LINK] = root.findtext(NETWORK_SETTINGS_HEADER + IPV6_LINK)
            self._network_cfg_parameters[IPV6_GLOBAL] = root.findtext(NETWORK_SETTINGS_HEADER + IPV6_GLOBAL)
            self._network_cfg_parameters[GATEWAY] = root.findtext(NETWORK_SETTINGS_HEADER + GATEWAY)
            self._network_cfg_parameters[IPV4_DHCP_SWITCH] = int(
                root.findtext(NETWORK_SETTINGS_HEADER + IPV4_DHCP_SWITCH))
            self._ipv6_dhcp_switch = int(root.findtext(NETWORK_SETTINGS_HEADER + IPV6_DHCP_SWITCH))
            self._network_cfg_parameters[DNS_SERVER1] = root.findtext(NETWORK_SETTINGS_HEADER + DNS_SERVER1)
            self._network_cfg_parameters[DNS_SERVER2] = root.findtext(NETWORK_SETTINGS_HEADER + DNS_SERVER2)
            self._network_cfg_parameters[NTP_SERVER1] = root.findtext(NETWORK_SETTINGS_HEADER + NTP_SERVER1)
            self._network_cfg_parameters[NTP_SERVER2] = root.findtext(NETWORK_SETTINGS_HEADER + NTP_SERVER2)
            self._network_cfg_parameters[MACADDRESS] = root.findtext(NETWORK_SETTINGS_HEADER + MACADDRESS)
        super().__init__(TAG_NAME, self._network_cfg_parameters)

    # ==================================================================
    @property
    def hostname(self) -> str:
        """The hostname.

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[HOSTNAME]

    @hostname.setter
    @check_parameter
    def hostname(self, value: str) -> None:
        self._network_cfg_parameters[HOSTNAME] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + HOSTNAME, self._network_cfg_parameters[HOSTNAME])

    # ==================================================================
    @property
    def mac(self) -> str:
        """MAC 地址

        Notes:
            可读可写

        """
        return self.mac

    @mac.setter
    @check_parameter
    def mac(self, value: str) -> None:
        self._network_cfg_parameters[MACADDRESS] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + MACADDRESS, self._network_cfg_parameters[MACADDRESS])

    # ==================================================================
    @property
    def ipv4_address(self) -> IPv4Address:
        """ipv4 地址

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[IPV4_ADDRESS]

    @ipv4_address.setter
    @check_parameter
    def ipv4_address(self, value: IPv4Address) -> None:
        self._network_cfg_parameters[IPV4_ADDRESS] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + IPV4_ADDRESS, self._network_cfg_parameters[IPV4_ADDRESS])

    # ==================================================================
    @property
    def netmask(self) -> str:
        """掩码

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[NETMASK]

    @netmask.setter
    @check_parameter
    def netmask(self, value: str) -> None:
        self._network_cfg_parameters[NETMASK] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + NETMASK, self._network_cfg_parameters[NETMASK])

    # ==================================================================
    @property
    def gateway(self) -> str:
        """网关

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[GATEWAY]

    @gateway.setter
    @check_parameter
    def gateway(self, value: str) -> None:
        self._network_cfg_parameters[GATEWAY] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + GATEWAY, self._network_cfg_parameters[GATEWAY])

    # ==================================================================
    @property
    def ipv4_dhcp_switch(self) -> int:
        """ipv4 DHCP

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[IPV4_DHCP_SWITCH]

    @ipv4_dhcp_switch.setter
    @check_parameter
    def ipv4_dhcp_switch(self, value: int):
        self._network_cfg_parameters[IPV4_DHCP_SWITCH] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + IPV4_DHCP_SWITCH, self._network_cfg_parameters[IPV4_DHCP_SWITCH])

    # ==================================================================
    @property
    def ipv6_global_address(self) -> IPv6Address:
        """ipv6 global address

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[IPV6_GLOBAL]

    @ipv6_global_address.setter
    @check_parameter
    def ipv6_global_address(self, value: IPv6Address) -> None:
        self._network_cfg_parameters[IPV6_GLOBAL] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + IPV6_GLOBAL, self._network_cfg_parameters[IPV6_GLOBAL])

    # ==================================================================
    @property
    def ipv6_link_address(self) -> str:
        """ipv6 link address

        Notes:
            可读可写

        """
        return self._network_cfg_parameters[IPV6_LINK]

    @ipv6_link_address.setter
    @check_parameter
    def ipv6_link_address(self, value: str) -> None:
        self._network_cfg_parameters[IPV6_LINK] = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + IPV6_LINK, self._network_cfg_parameters[IPV6_LINK])

    # ==================================================================
    @property
    def ipv6_dhcp_switch(self) -> int:
        """ipv4 DHCP address

        Notes:
            可读可写

        """
        return self._ipv6_dhcp_switch

    @ipv6_dhcp_switch.setter
    @check_parameter
    def ipv6_dhcp_switch(self, value: int) -> None:
        self._ipv6_dhcp_switch = value
        self.set_node_text(NETWORK_SETTINGS_HEADER + IPV6_DHCP_SWITCH, self._ipv6_dhcp_switch)

    # ==================================================================
    def __str__(self):
        res = self.__class__.__name__ + ":\n"
        for k, v in self._network_cfg_parameters.items():
            res += "\t" + str(k).replace("\n", "") + ":" + str(v).replace("\n", "") + "\n"
        return res
