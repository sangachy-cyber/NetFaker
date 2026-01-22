"""
HoloWAN Network Emulator
Python API

Packet Classifier: VLAN Rule
"""

import xml.etree.cElementTree as ET

from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION
from holowan.v2._holowan_types import (
    PathID
)
from holowan.v2._holowan_types import check_parameter

_ANY: str = r"any"
_RULE_NAME: str = HoloWANRuleTag.VLAN
_FPCP: str = r"fpcp"
_FPID: str = r"fpid"
_ENABLE_STAG: str = r"enable_stag"
_SPCP: str = r"spcp"
_SPID: str = r"spid"
_ACTION: str = _RULE_ACTION

_VLAN_DEFAULT_CUSTOM_NAME: str = r"VLAN: pcp {} id {}"


class VLANRule(Rule):
    """VLAN 分类规则。

    Note:
        强制关键字传参。

    Args:
        fpcp (str): VLAN 的 fpcp。
        fpid(str): VLAN 的 fpid。
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> vlan = VLANRule(fpcp="any", fpid="any", action=1)
        >>> vlan.enable_second_tag(spcp="any", spid="any")

    """

    def __init__(self, *, fpcp: str, fpid: str, action: PathID):
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_FPCP] = fpcp
        self.rule_parameters[_FPID] = fpid
        self.rule_parameters[_ENABLE_STAG] = 0
        self.rule_parameters[_SPCP] = _ANY
        self.rule_parameters[_SPID] = _ANY
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = {
            _FPCP: "src",
            _FPID: "dst",
            _ACTION: "action",
            _ENABLE_STAG: "enable_second_tag",
            _SPCP: "spcp",
            _SPID: "spid"
        }
        self._update_rule()
        self.set_custom_name(_VLAN_DEFAULT_CUSTOM_NAME.format(fpcp, fpid))

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。

        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @check_parameter
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def fpcp(self) -> str:
        return self.rule_parameters[_FPCP]

    @fpcp.setter
    def fpcp(self, value: str) -> None:
        self.rule_parameters[_FPID] = value
        self._update_rule()

    def enable_second_tag(self, spcp: str = _ANY, spid: str = _ANY):
        """启用 VLAN 分类规则的 second tag。

        Args:
            spcp: Second tag 的 spcp。
            spid: Second tag 的 spid。

        """
        self.rule_parameters[_ENABLE_STAG] = 1
        self.rule_parameters[_SPCP] = spcp
        self.rule_parameters[_SPID] = spid
        self._update_rule()

    def _update_rule(self):
        self.clear_children()
        children_node_Map = {
            _FPCP: "",
            _FPID: "",
            _ENABLE_STAG: self.rule_parameters[_ENABLE_STAG],
            _ACTION: self.rule_parameters[_ACTION]
        }
        properties = {}

        if self.rule_parameters[_FPCP] == _ANY:
            properties[_FPCP] = {_ANY: 1}
        else:
            children_node_Map[_FPCP] = self.rule_parameters[_FPCP]

        if self.rule_parameters[_FPID] == _ANY:
            properties[_FPID] = {_ANY: 1}
        else:
            children_node_Map[_FPID] = self.rule_parameters[_FPID]

        if self.rule_parameters[_ENABLE_STAG] == 1:
            if self.rule_parameters[_SPCP] == _ANY:
                children_node_Map[_SPCP] = ""
                properties[_SPCP] = {_ANY: 1}
            else:
                children_node_Map[_SPCP] = self.rule_parameters[_SPCP]

            if self.rule_parameters[_SPID] == _ANY:
                children_node_Map[_SPID] = ""
                properties[_SPID] = {_ANY: 1}
            else:
                children_node_Map[_SPID] = self.rule_parameters[_SPID]

        self.add_children(children_node_Map)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'VLANRule':
        """由一个 VLAN 分类规则的 xml 节点构造 VLANRule 对象。

        Args:
            node (ET.Element): VLAN 分类规则的 xml 节点。

        Returns:
            VLANRule 对象。
        """
        if node.find(_FPCP).get(_ANY) == "1":
            fpcp = _ANY
        else:
            fpcp = node.findtext(_FPCP)

        if node.find(_FPID).get(_ANY) == "1":
            fpid = _ANY
        else:
            fpid = node.findtext(_FPID)

        enable_stag = node.findtext(_ENABLE_STAG)
        action = int(node.findtext(_ACTION))
        vlan = VLANRule(
            fpcp=fpcp, fpid=fpid, action=action
        )
        if enable_stag == 1:
            if node.find(_SPCP).get(_ANY) == "1":
                spcp = _ANY
            else:
                spcp = node.findtext(_SPCP)

            if node.find(_SPID).get(_ANY) == "1":
                spid = _ANY
            else:
                spid = node.findtext(_SPID)
            vlan.enable_second_tag(spcp, spid)

        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        vlan.attrib = node.attrib
        return vlan

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.rule_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res
