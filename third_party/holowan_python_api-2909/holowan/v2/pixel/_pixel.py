"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET
from typing import Union

import requests

import holowan.v2.utils.my_util as mt
from holowan.v2._commons import (
    _capture_mode_url, _capture_stop_url, _capture_clean_url, _capture_start_url, _capture_filter_url,
    _capture_load_file_url, _capture_delete_file_url, _capture_release_file_url, _capture_pcap_export_url,
    _capture_get_all_file_url, _capture_settings_url,_capture_get_pixel_data,_capture_get_pixel_status
)
from holowan.v2._holowan_types import HoloWANReturn
from holowan.v2._holowan_types import PathID, PortNumber, IPv4Address, EngineID
from holowan.v2._xml_holder import XMLHolder
from holowan.v2.pixel._filter_base import Filter
from holowan.v2.pixel._impair_flag import TypeFilter
from holowan.v2.pixel._ip_filter import IPv4Filter, IPv6Filter
from holowan.v2.pixel._length import LengthFilter
from holowan.v2.pixel._mac_filter import MACFilter
from holowan.v2.pixel._tcp_udp_filter import TCPFilter, UDPFilter, SCTPFilter
from holowan.v2.pixel._vlan_filter import VLANFilter
from holowan.v2.utils import xml_util as xt

_CAPTURE_FILTER: str = r"capture_filter"
_CAPTURE_DIRECTION: str = r"direction"
_CAPTURE_PATH: str = r"pid"


class FilterSettings(XMLHolder):
    """Pixel 过滤器配置

    """

    def __init__(self, path: PathID, direction: int):
        super().__init__(_CAPTURE_FILTER)
        self._basic_param = {}
        self._basic_param[_CAPTURE_PATH] = path
        self._basic_param[_CAPTURE_DIRECTION] = direction

        self._type = TypeFilter(capture_moment=1)
        self._mac = MACFilter(src='any', dst='any', type='any')
        self._mac.disable_filter()
        self._vlan = VLANFilter(fpcp='any', fpid='any')
        self._vlan.enable_second_tag(spid="any", spcp="any")
        self._vlan.disable_filter()
        self._ipv4: IPv4Filter = IPv4Filter(src='any', dst='any', smask=32, dmask=32, tos="any")
        self._ipv4.disable_filter()
        self._ipv6: IPv6Filter = IPv6Filter(src="any", dst="any")
        self._ipv6.disable_filter()
        self._tcp_udp_sctp = TCPFilter(src="any", dst="any", check_version=0)
        self._tcp_udp_sctp.disable_filter()
        self._pk_length = LengthFilter(length_from=0, length_to=1500)
        self._pk_length.disable_filter()

        self._update_settings()
        self.disable_path()
        self.disable_direction()

    def _update_settings(self):
        self.clear_children()
        self.add_children(self._basic_param)
        if self._type != None:
            self.add_child(self._type.node)

        if self._mac != None:
            self.add_child(self._mac.node)

        if self._vlan != None:
            self.add_child(self._vlan.node)

        if self._ipv4 != None:
            self.add_child(self._ipv4.node)

        if self._ipv6 != None:
            self.add_child(self._ipv6.node)

        if self._tcp_udp_sctp != None:
            self.add_child(self._tcp_udp_sctp.node)

        if self._pk_length != None:
            self.add_child(self._pk_length.node)

    @property
    def path(self) -> PathID:
        return self._basic_param[_CAPTURE_PATH]

    @path.setter
    def path(self, value) -> None:
        self._basic_param[_CAPTURE_PATH] = value
        self.set_node_text(_CAPTURE_PATH, self._basic_param[_CAPTURE_PATH])
        self._update_settings()

    @property
    def direction(self) -> int:
        return self._basic_param[_CAPTURE_DIRECTION]

    @direction.setter
    def direction(self, value: int) -> None:
        self._basic_param[_CAPTURE_DIRECTION] = value
        self.set_node_text(_CAPTURE_DIRECTION, self._basic_param[_CAPTURE_DIRECTION])
        self._update_settings()

    def enable_path(self):
        self.set_property(_CAPTURE_PATH, "enable", "1")
        self._update_settings()

    def disable_path(self):
        self.set_property(_CAPTURE_PATH, "enable", "0")
        self._update_settings()

    def enable_direction(self):
        self.set_property(_CAPTURE_DIRECTION, "enable", "1")
        self._update_settings()

    def disable_direction(self):
        self.set_property(_CAPTURE_DIRECTION, "enable", "0")
        self._update_settings()

    @property
    def type(self) -> TypeFilter:
        return self._type

    @type.setter
    def type(self, value: TypeFilter) -> None:
        self._type = value
        self._update_settings()

    @property
    def mac(self) -> MACFilter:
        return self._mac

    @mac.setter
    def mac(self, value: MACFilter) -> None:
        self._mac = value
        self._update_settings()

    @property
    def vlan(self) -> VLANFilter:
        return self._vlan

    @vlan.setter
    def vlan(self, value: VLANFilter) -> None:
        self._vlan = value
        self._update_settings()

    @property
    def ipv4(self) -> IPv4Filter:
        return self._ipv4

    @ipv4.setter
    def ipv4(self, value: IPv4Filter) -> None:
        self._ipv4 = value
        self._update_settings()

    @property
    def ipv6(self) -> IPv6Filter:
        return self._ipv6

    @ipv6.setter
    def ipv6(self, value: IPv6Filter) -> None:
        self._ipv6 = value
        self._update_settings()

    @property
    def tcp_udp_sctp(self) -> Union[TCPFilter, UDPFilter, SCTPFilter]:
        return self._tcp_udp_sctp

    @tcp_udp_sctp.setter
    def tcp_udp_sctp(self, value: Union[TCPFilter, UDPFilter, SCTPFilter]) -> None:
        self._tcp_udp_sctp = value
        self._update_settings()

    @property
    def packet_length(self) -> LengthFilter:
        return self._pk_length

    @packet_length.setter
    def packet_length(self, value: LengthFilter) -> None:
        self._pk_length = value
        self._update_settings()

    def reset(self):
        self.disable_path()
        self.disable_direction()
        self._type = TypeFilter(capture_moment=1)
        self._mac = MACFilter(src='any', dst='any', type='any')
        self._mac.disable_filter()
        self._vlan = VLANFilter(fpcp='any', fpid='any')
        self._vlan.enable_second_tag(spid="any", spcp="any")
        self._vlan.disable_filter()
        self._ipv4: IPv4Filter = IPv4Filter(src='any', dst='any', smask=32, dmask=32, tos="any")
        self._ipv4.disable_filter()
        self._ipv6: IPv6Filter = IPv6Filter(src="any", dst="any")
        self._ipv6.disable_filter()
        self._tcp_udp_sctp = TCPFilter(src="any", dst="any", check_version=0)
        self._tcp_udp_sctp.disable_filter()
        self._pk_length = LengthFilter(length_from=0, length_to=1500)
        self._pk_length.disable_filter()
        self._update_settings()

    def __str__(self):
        res = self.__class__.__name__ + ":\n"
        res += "\tPath:{0}\n".format(self._basic_param[_CAPTURE_PATH])
        res += "\tDirection:{0}\n".format(self._basic_param[_CAPTURE_DIRECTION])
        res += "\t" + self._mac.__str__() + "\n"
        res += "\t" + self._vlan.__str__() + "\n"
        res += "\t" + self._ipv4.__str__() + "\n"
        res += "\t" + self._ipv6.__str__() + "\n"
        res += "\t" + self._tcp_udp_sctp.__str__() + "\n"
        res += "\t" + self._pk_length.__str__()
        return res

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'FilterSettings':
        path_id = int(node.findtext(_CAPTURE_PATH))
        direction = int(node.findtext(_CAPTURE_DIRECTION))
        fs = FilterSettings(path=path_id, direction=direction)

        for n in list(node):
            filter_name = n.tag
            if n.tag == "pid" or n.tag == "direction":
                path_enable = True if node.find(_CAPTURE_PATH).get("enable") == "1" else False
                if path_enable:
                    fs.enable_path()
            elif filter_name == "direction":
                dir_enable = True if node.find(_CAPTURE_DIRECTION).get("enable") == "1" else False
                if dir_enable:
                    fs.enable_direction()
            elif filter_name == "mac":
                fs.mac = MACFilter.construct_from_node(n)
            elif filter_name == "ipv4":
                fs.ipv4 = IPv4Filter.construct_from_node(n)
            elif filter_name == "ipv6":
                fs.ipv6 = IPv6Filter.construct_from_node(n)
            elif filter_name == "tcp_udp":
                _t = int(n.findtext("type"))
                if _t == 1:
                    _f = TCPFilter.construct_from_node(n)
                elif _t == 2:
                    _f = UDPFilter.construct_from_node(n)
                else:
                    _f = SCTPFilter.construct_from_node(n)
                fs.tcp_udp_sctp = _f
            elif filter_name == "impair_flag":
                fs.type = TypeFilter.construct_from_node(n)
            elif filter_name == "length":
                fs.packet_length = LengthFilter.construct_from_node(n)
            elif filter_name == "vlan":
                fs.vlan = VLANFilter.construct_from_node(n)
            else:
                raise TypeError("Unknown filter type, got {got!r}".format(
                    got=filter_name
                ))
        return fs


class PixelMode:
    ALL_PACKETS: str = r"all_pkt"
    ALL_PATH: str = r"all_path"
    ONE_PATH: str = r"one_path"


class Pixel(object):
    """Pixel

    Args:
    holowan_ip (IPAddress): HoloWAN 的 ip 地址。
    holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
    engine_id (EngineID): 引擎 id。
    username (str): 登录用户名，默认为 "admin"。
    password (str): 登录密码，默认为 "holowan"。

    Notes:
        构造函数会自动执行登录操作，获取 Cookie session_id。
        所有后续的 API 调用都会自动携带该 Cookie。

    """

    def __init__(self, holowan_ip: IPv4Address, holowan_port: PortNumber, engine_id: EngineID,
                 username: str = "admin", password: str = "holowan") -> None:
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port
        self._engine_id = engine_id
        self._filter_settings = None
        
        # Session 对象用于保存 Cookie
        self.session = requests.Session()
        
        # 在调用任何 API 之前先执行登录
        self._login(username, password)

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

    def set_filter_settings(self, settings: FilterSettings) -> HoloWANReturn:
        xml_str = settings.xml

        result = mt.post_original_api(
            _capture_filter_url(self._holowan_ip, self._holowan_port, self._engine_id),
            xml_str,
            self.session
        )
        return HoloWANReturn(result)

    def get_filter_settings(self):
        try:
            url = _capture_settings_url(self._holowan_ip, self._holowan_port, self._engine_id)
            xml_str = self.session.get(url, verify=False).text
            return FilterSettings.construct_from_node(xt.xmlString_to_Object(xml_str))
        except Exception:
            return FilterSettings(path=1, direction=0)

    def reset_filter_settings(self) -> HoloWANReturn:
        fs = FilterSettings(path=1,direction=0)
        fs.reset()
        return self.set_filter_settings(fs)

    def set_filter(self, filter: Filter) -> HoloWANReturn:
        fs = self.get_filter_settings()
        if isinstance(filter, MACFilter):
            fs.mac = filter
        elif isinstance(filter, TypeFilter):
            fs.type = filter
        elif isinstance(filter, VLANFilter):
            fs.vlan = filter
        elif isinstance(filter, IPv4Filter):
            fs.ipv4 = filter
        elif isinstance(filter, IPv6Filter):
            fs.ipv6 = filter
        elif isinstance(filter, TCPFilter) or isinstance(filter, UDPFilter) or isinstance(filter, SCTPFilter):
            fs.tcp_udp_sctp = filter
        elif isinstance(filter, LengthFilter):
            fs.packet_length = filter
        else:
            raise TypeError("Argument filter is not a Filter, got {got!r}, value {value!r}".format(
                got=type(filter), value=filter
            ))
        return self.set_filter_settings(fs)

    def set_filter_path(self, path_id: PathID) -> HoloWANReturn:
        fs = self.get_filter_settings()
        fs.path = path_id
        fs.enable_path()
        return self.set_filter_settings(fs)

    def set_filter_direction(self, direction: int) -> HoloWANReturn:
        if direction not in [0, 1]:
            raise ValueError("Argument direction must be 0 or 1, got {got!r}, value {value!r}".format(
                got=type(direction), value=direction
            ))
        fs = self.get_filter_settings()
        fs.direction = direction
        fs.enable_direction()
        return self.set_filter_settings(fs)

    def start_capture(self) -> HoloWANReturn:
        url = _capture_start_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def stop_capture(self):
        url = _capture_stop_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def clean_capture(self):
        url = _capture_clean_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def export_pcap(self, file_path: str):
        url = _capture_pcap_export_url(
            self._holowan_ip, self._holowan_port, self._engine_id
        )
        return mt.create_and_write_file(file_path, self.session.get(url, verify=False).content)

    def set_capture_mode(self, mode, path_id: PathID = None):
        if mode == PixelMode.ONE_PATH:
            if path_id == None:
                raise ValueError(
                    "When pixel capture mode is PixelMode.ONE_PATH, the argument 'path_id' can not be None.")
        else:
            path_id = 1
        url = _capture_mode_url(
            self._holowan_ip, self._holowan_port, self._engine_id,
            path_id, mode
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def get_all_pixel_file_info(self):
        url = _capture_get_all_file_url(
            self._holowan_ip, self._holowan_port
        )
        return self.session.get(url, verify=False).text

    def load_pixel_file(self, file_path: str):
        url = _capture_load_file_url(
            self._holowan_ip, self._holowan_port, file_path
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def release_pixel_file(self, file_name: str):
        url = _capture_release_file_url(
            self._holowan_ip, self._holowan_port,
            self._engine_id, file_name
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)

    def delete_pixel_file(self, file_name: str):
        url = _capture_delete_file_url(
            self._holowan_ip, self._holowan_port, file_name
        )
        return HoloWANReturn(self.session.get(url, verify=False).text)


    def get_pixel_capture_pkt_count(self)->int:
        """

        Returns:
            int, 返回数据包总数
            -1， 无数据包或错误
        """
        url = _capture_get_pixel_status(
            self._holowan_ip,self._holowan_port
        )
        try:
            data =eval(self.session.get(url, verify=False).text)
            for cdata in data["data"]:
                if cdata["cid"] == self._engine_id:
                    return int(cdata["pkt_count"])
            return -1
        except:
            return -1

    def get_pixel_capture_pkt_data(self,offset:int,count:int):
        url = _capture_get_pixel_data(
            self._holowan_ip,self._holowan_port,self._engine_id,offset,count
        )
        return self.session.get(url, verify=False).text


if __name__ == '__main__':
    fs = FilterSettings(path=1, direction=0)
    print(fs.xml)
    fs.ipv4 = IPv4Filter(src="any", smask=32, dst="any", dmask=32, tos="any")
    print(fs.xml)
    holowan_ip = "192.168.1.111"
    holowan_port = "8080"
    engine_id = 2
    pixel = Pixel(holowan_ip, holowan_port, engine_id)
    pixel.set_filter_settings(fs)
    res = pixel.set_capture_mode(PixelMode.ALL_PATH)
    print(res)
