"""
HoloWAN Network Emulator
Python API

Packet Classifier: MPLS Rule
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import PathID, check_parameter
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION

_RULE_NAME = HoloWANRuleTag.MPLS
_ANY: str = r"any"
_LABEL: str = r"label"
_ACTION: str = _RULE_ACTION

_MPLS_DEFAULT_CUSTOM_NAME: str = r"MPLS: label range {}"


class MPLSRule(Rule):
    """MPLS分类规则。

    Note:
        强制关键字传参。

    Args:
        label (str): MPLS 标签。
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> mpls = MPLSRule(label="any", action=1)

    """

    def __init__(self, *, label: str, action: PathID) -> None:
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_LABEL] = label
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = {
            _LABEL: "label",
            _ACTION: "action",
        }
        self._update_rule()
        self.set_custom_name(_MPLS_DEFAULT_CUSTOM_NAME.format(label))

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
    def label(self) -> str:
        return self.rule_parameters[_LABEL]

    @label.setter
    def label(self, value: str) -> None:
        self.rule_parameters[_LABEL] = value
        self._update_rule()

    @staticmethod
    @check_parameter
    def construct_from_node(node: ET.Element) -> 'MPLSRule':
        """由一个 MPLS 分类规则的 xml 节点构造 MPLSRule 对象。

        Args:
            node (ET.Element): MPLS 分类规则的 xml 节点。

        Returns:
            MPLSRule 对象。
        """
        label = node.findtext(_LABEL)
        if label == None and node.find(_LABEL).get(_ANY) == "1":
            label = _ANY

        rule = MPLSRule(
            label=label, action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule
