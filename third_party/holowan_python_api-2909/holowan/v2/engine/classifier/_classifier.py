"""
HoloWAN Network Emulator
Python API

PacketClassifier
"""

import xml.etree.cElementTree as ET

import holowan.v2.utils.xml_util as xt
from holowan.v2.engine.classifier._combination_rule import CombinationRule
from holowan.v2.engine._container import Sequential
from holowan.v2.engine.classifier._ip_rule import IPv4Rule, IPv6Rule
from holowan.v2.engine.classifier._mac_rule import MACRule
from holowan.v2.engine.classifier._mpls_rule import MPLSRule
from holowan.v2.engine.classifier._pppoe_rule import PPPoERule
from holowan.v2.engine.classifier._raw_byte_rule import RawByteRule
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._tcp_udp_sctp_rule import UDPRule, TCPRule, SCTPRule
from holowan.v2.engine.classifier._vlan_rule import VLANRule
from holowan.v2._holowan_types import (
    check_parameter,
    HoloWANConfigXML,
    EngineID
)
from holowan.v2._xml_holder import XMLHolder

_CLF_XML_FORMAT = """<?xml version="1.0" encoding="utf-8"?>
<clc>                                   
    <engine_id></engine_id>            
    <port>                             
        <port_id></port_id>            
    </port>
    <port>                             
        <port_id></port_id>            
    </port>
</clc>
"""

_PORT_ID = r"port_id"
_ENGINE_ID = r"engine_id"
_CLF_TAG_NAME = r"clc"


class PacketClassifier(XMLHolder):
    """HoloWAN 报文分类器
    此类设计为解析、维护 HoloWAN 报文分类器的 xml 配置。设计为：设备无关，引擎无关。
    
    Note:
        此类不具备与 HoloWAN 设备的通讯能力，对此类的属性的设置不会改变 HoloWAN 的配置。

    Examples:
        >>> engine = Engine(holowan_ip, holowan_port, engine_id)
        >>> classifier = engine.packet_classifier
        >>> classifier.port1 = Sequential(
        >>>    rule1,rul2,rule3,rule4,...
        >>> )

    """

    @check_parameter
    def __init__(self, clf_xml: HoloWANConfigXML = None) -> None:
        if clf_xml == None:
            clf_xml = _CLF_XML_FORMAT
        root = xt.xmlString_to_Object(clf_xml)
        super().__init__(_CLF_TAG_NAME, list(root))
        port_nodes = xt.get_nodes(root, "port")
        self._port1_root, self._port2_root = port_nodes[0], port_nodes[1]
        self._port1 = Sequential()
        self._port2 = Sequential()
        self._init_from_xml()

    @property
    def port1(self) -> Sequential:
        """报文分类器的 port1 部分。
        
        Note:
            可读可写。报文分类器的一个 port 为序列容器 Sequential，规则在该容器中为有序的。
            可以使用 Sequential 类的方法对其中的元素进行操作。

        """
        return self._port1

    @port1.setter
    def port1(self, value) -> None:
        if isinstance(value, Rule):
            self._port1 = Sequential(value)
        elif isinstance(value, Sequential):
            self._port1 = value
        else:
            raise ValueError("Argument must be Rule or Sequential, but got {got!r}, value {value!r}".format(
                got=type(value), value=value
            ))

        self.update()

    @property
    def port2(self) -> Sequential:
        """报文分类器的 port2 部分。
        
        Note:
            可读可写。报文分类器的一个 port 为序列容器 Sequential，规则在该容器中为有序的。
            可以使用 Sequential 类的方法对其中的元素进行操作。

        """
        return self._port2

    def add_rule(self, port: int, rule: Rule) -> None:
        if not isinstance(rule, Rule):
            raise ValueError("Argument must be Rule, but got {got!r}, value {value!r}".format(
                got=type(rule), value=rule
            ))
        if port % 2 == 1:
            rules = self._port1
        else:
            rules = self._port2
        rules.append(rule)

    def insert_rule(self, port: int, index: int, rule: Rule):
        if not isinstance(rule, Rule):
            raise ValueError("Argument must be Rule, but got {got!r}, value {value!r}".format(
                got=type(rule), value=rule
            ))
        if port % 2 == 1:
            rules = self._port1
        else:
            rules = self._port2
        rules.insert(index, rule)

    def modify_rule_by_idx(self, port: int, index: int, rule: Rule) -> None:
        if not isinstance(rule, Rule):
            raise ValueError("Argument must be Rule, but got {got!r}, value {value!r}".format(
                got=type(rule), value=rule
            ))

        if port % 2 == 1:
            rules = self._port1
        else:
            rules = self._port2

        rules.remove_item_by_idx(index)
        rules.insert(index, rule)

    def rearrange_rules(self, port: int, order: list):
        if port % 2 == 1:
            rules = self._port1
        else:
            rules = self._port2
        rules.rearrange(order)

    def remove_rule(self, port: int, index: int) -> None:
        if port % 2 == 1:
            rules = self._port1
        else:
            rules = self._port2

        rules.remove_item_by_idx(index)

    @port2.setter
    def port2(self, value) -> None:
        if isinstance(value, Rule):
            self._port2 = Sequential(value)
        elif isinstance(value, Sequential):
            self._port2 = value
        else:
            raise ValueError("Argument must be Rule or Sequential, but got {got!r}, value {value!r}".format(
                got=type(value), value=value
            ))

        self.update()

    def update(self):
        """更新报文分类器。该方法控制报文分类器属性发生更改时的更新行为。

        """
        for node in list(self._port1_root):
            if node.tag != _PORT_ID:
                self._port1_root.remove(node)
        for node in list(self._port2_root):
            if node.tag != _PORT_ID:
                self._port2_root.remove(node)

        if self._port1 is not None:
            for node in self._port1.nodes:
                self._port1_root.append(node)

        if self._port2 is not None:
            for node in self._port2.nodes:
                self._port2_root.append(node)

        # remove old
        for node in self.findall("port"):
            self.remove(node)

        self.add_child(self._port1_root)
        self.add_child(self._port2_root)

    @check_parameter
    def attachTo(self, engine: EngineID) -> None:
        """将当前报文分类器配置到指定的引擎。

        Args:
            engine(EngineID): 目标引擎的 id。

        """
        self.set_node_text(_ENGINE_ID, engine)
        node = self._port1_root.find(_PORT_ID)
        node.text = str(2 * engine - 1)
        node = self._port2_root.find(_PORT_ID)
        node.text = str(2 * engine)

    def reset(self) -> None:
        """重置报文分类器。清空所有规则。
        
        """
        self._port1 = Sequential()
        self._port2 = Sequential()
        self.update()

    def _init_from_xml(self):
        """从 xml 字符串构造规则。

        """
        for idx, item in enumerate(list(self._port1_root)):
            if item.tag == _PORT_ID:
                continue
            rule = rule_factory(item)
            if rule != None:
                self._port1.add_item(str(idx), rule)

        for idx, item in enumerate(list(self._port2_root)):
            if item.tag == _PORT_ID:
                continue
            rule = rule_factory(item)
            if rule != None:
                self._port2.add_item(str(idx), rule)

    def __str__(self):
        res = self.__class__.__name__ + ":\n"
        res += "port1:\n"
        res += self._port1.__str__()
        res += "\n"
        res += "port2:\n"
        res += self._port2.__str__()
        return res


def rule_factory(node: ET.Element) -> Rule:
    rule_name = node.tag
    if rule_name == HoloWANRuleTag.MAC:
        return MACRule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.TCP_UDP_SCTP:
        rule_type = int(node.findtext("type"))
        if rule_type == 1:
            return TCPRule.construct_from_node(node)
        elif rule_type == 2:
            return UDPRule.construct_from_node(node)
        else:
            return SCTPRule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.MPLS:
        return MPLSRule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.PPPoE:
        return PPPoERule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.RawByte:
        return RawByteRule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.VLAN:
        return VLANRule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.IPv4:
        return IPv4Rule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.IPv6:
        return IPv6Rule.construct_from_node(node)
    elif rule_name == HoloWANRuleTag.COMBINATION:
        return CombinationRule.construct_from_node(node)
    else:
        raise ValueError("Unknown rule type: {0}.".format(rule_name))


if __name__ == "__main__":
    clf = PacketClassifier()
    clf.port1 = Sequential(
        TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
        SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
        MACRule(src="any", dst="any", type="any", action=1),
    )
    clf.attachTo(1)
    print(clf.xml)
