"""
HoloWAN Network Emulator
Python API

Rule
"""

import xml.etree.cElementTree as ET
from abc import abstractmethod
from typing import Union, List

from holowan.v2._xml_holder import XMLHolder


class HoloWANRuleTag:
    IPv4 = r"ipv4"
    IPv6 = r"ipv6"
    MAC = r"mac"
    MPLS = r"mpls"
    PPPoE = r"pppoe"
    RawByte = r"raw_group"
    TCP_UDP_SCTP = r"tcp_udp"
    VLAN = r"vlan"
    COMBINATION = r"comb"


_RULE_ACTION: str = r"path_id"


class Rule(XMLHolder):
    """报文分类器分类规则基类
    所有的规则继承自此基类。此基类实现了由参数构造 xml 。

    """

    def __init__(self, rule_name: str, rule_parameters: Union[dict, List['Rule']] = None):
        super().__init__(rule_name, rule_parameters)
        self.rule_parameters = rule_parameters
        self._param_tag2str = {}

    @property
    def rule_name(self):
        return self.tag

    def set_custom_name(self, name: str):
        self.set_property(self.tag, "label", name)

    def _update_rule(self):  # TODO: 抽象到基类来实现
        """
        适用于没有嵌套子节点, 没有 property 的更新
        如果有嵌套子节点和 property, 子类需重写此方法
        """
        self.clear_children()
        self.add_children(self.rule_parameters)

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element):
        """由 xml 节点构造 Rule 对象
        Note:
            抽象方法。子类必须实现此方法。
        Args:
            node:

        Returns:

        """
        ...

    def __str__(self) -> str:
        res = self.__class__.__name__ + ":{ "
        for k, v in self.rule_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res
