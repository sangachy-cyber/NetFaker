"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

import holowan.v2.utils.xml_util as xt
from holowan.v2._holowan_types import (
    IPv6Address, IPv4Address
)
from holowan.v2.pixel._filter_base import Filter

_IPV4_FILTER_NAME: str = r"ipv4"
_TUNNEL: str = r"tunnel"
_SRC_MASK: str = r"smask"
_DST_MASK: str = r"dmask"
_TOS: str = r"tos"
_IPV6_FILTER_NAME: str = r"ipv6"
_ANY: str = r"any"
_SRC: str = r"src"
_DST: str = r"dst"
_TYPE: str = r"type"

_ip_tag2str_dict = {
    _SRC: "src",
    _DST: "dst",
    _SRC_MASK: "smask",
    _DST_MASK: "dmask",
    _TOS: "tos",
}


class IPv4Filter(Filter):
    """IPv4分类规则。

    Note:
        强制关键字传参。

    Args:
        src (IPv4Address): 源 ipv4 地址。
        smask(IPv4Address): 源 ipv4 掩码。
        dst(IPv4Address): 目的 ipv4 地址。
        dmask(IPv4Address): 目的 ipv4 掩码。
        tos(str): TOS。

    Examples:
        >>> ipv4 = IPv4Filter(src="any", smask=32, dst="any", dmask=32, tos="any")

    """

    def __init__(self, *, src: IPv4Address, smask: int, dst: IPv4Address, dmask: int, tos: str):
        super().__init__(_IPV4_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_SRC] = src
        self.filter_parameters[_DST] = dst
        self.filter_parameters[_SRC_MASK] = smask
        self.filter_parameters[_DST_MASK] = dmask
        self.filter_parameters[_TOS] = tos
        self._param_tag2str = _ip_tag2str_dict
        self._update_filter()

    def _update_filter(self):
        self.clear_children()
        children_node_Map = {
            _SRC: "",
            _SRC_MASK: self.filter_parameters[_SRC_MASK],
            _DST: "",
            _DST_MASK: self.filter_parameters[_DST_MASK],
            _TOS: "",
        }
        properties = {}
        # ===============src================== #
        if self.filter_parameters[_SRC] == _ANY:
            properties[_SRC] = {_ANY: 1}
            children_node_Map[_SRC] = self.filter_parameters[_SRC]
        else:
            children_node_Map[_SRC] = self.filter_parameters[_SRC]
        # ===============dst================== #
        if self.filter_parameters[_DST] == _ANY:
            properties[_DST] = {_ANY: 1}
            children_node_Map[_DST] = self.filter_parameters[_DST]
        else:
            children_node_Map[_DST] = self.filter_parameters[_DST]
        # ===============tos================== #
        if self.filter_parameters[_TOS] == _ANY:
            properties[_TOS] = {_ANY: 1}
        else:
            children_node_Map[_TOS] = self.filter_parameters[_SRC]
        # ===============Action================== #
        # TODO: has path
        xt.add_children(self, children_node_Map)
        xt.add_properties(self, properties)

    @staticmethod
    def construct_from_node(node: ET.Element):
        src_node = node.find(_SRC)
        if src_node.get(_ANY) == "1":
            src = _ANY
        else:
            src = node.findtext(_SRC)

        dst_node = node.find(_DST)
        if dst_node.get(_ANY) == "1":
            dst = _ANY
        else:
            dst = node.findtext(_DST)

        tos_node = node.find(_TOS)
        if tos_node.get(_ANY) == "1":
            tos = _ANY
        else:
            tos = node.findtext(_TOS)

        enable = True if node.get("enable") == "1" else False
        ipv4 = IPv4Filter(
            src=src, smask=int(node.findtext(_SRC_MASK)),
            dst=dst, dmask=int(node.findtext(_DST_MASK)),
            tos=tos
        )
        if not enable:
            ipv4.disable_filter()
        return ipv4


class IPv6Filter(Filter):
    """IPv6分类规则。

    Note:
        强制关键字传参。

    Args:
        src (IPv6Address): 源 ipv6 地址。
        dst(IPv6Address): 目的 ipv6 地址。

    Examples:
        >>> ipv6 = IPv6Filter(src="any", dst="any")

    """

    def __init__(self, *, src: IPv6Address, dst: IPv6Address):
        super().__init__(_IPV6_FILTER_NAME)
        super().__setattr__("filter_parameters", {})
        self.filter_parameters[_SRC] = src
        self.filter_parameters[_DST] = dst
        self._param_tag2str = _ip_tag2str_dict
        self._update_filter()

    def _update_filter(self):
        self.clear_children()
        properties = {}
        # ===============src================== #
        if self.filter_parameters[_SRC] == _ANY:
            properties[_SRC] = {_ANY: 1}

        # ===============dst================== #
        if self.filter_parameters[_DST] == _ANY:
            properties[_DST] = {_ANY: 1}

        # ===============Action================== #
        # TODO: has path
        self.add_children(self.filter_parameters)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element):
        src_node = node.find(_SRC)
        if src_node.get(_ANY) == "1":
            src = _ANY
        else:
            src = node.findtext(_SRC)

        dst_node = node.find(_DST)
        if dst_node.get(_ANY) == "1":
            dst = _ANY
        else:
            dst = node.findtext(_DST)

        enable = True if node.get("enable") == "1" else False
        ipv6 = IPv6Filter(
            src=src, dst=dst
        )
        if not enable:
            ipv6.disable_filter()

        return ipv6


if __name__ == '__main__':
    ipv6 = IPv6Filter(src="any", dst="any")
    print(ipv6)
    print(ipv6.xml)

    ipv4 = IPv4Filter(src="any", smask=32, dst="any", dmask=32, tos="any")
    print(ipv4)
    print(ipv4.xml)
