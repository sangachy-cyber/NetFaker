"""
HoloWAN Network Emulator
Python API

Packet Classifier: PPPoE Rule
"""

import xml.etree.cElementTree as ET

import holowan.v2.utils.xml_util as xt
from holowan.v2._holowan_types import (
    PathID,
    check_parameter
)
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION

_RULE_NAME = HoloWANRuleTag.PPPoE
_ANY: str = r"any"
_SID: str = r"sid"
_CODE: str = r"code"
_ACTION: str = _RULE_ACTION

_PPPOE_XML_FORMAT = """
<pppoe>                 /* 关于PPPoE层的报文分类 */
    <sid any="1"/>      /* 匹配的PPPoE的Session ID值，若为任意值，则属性any值为1*/
    <code any="1"/>     /* 匹配的PPPoE的Code值，若为任意值，则属性any值为1 */
    <path_id></path_id> /* 命中分类后进入的虚拟链路ID，取值范围：1-15编号 */
</pppoe>
"""


class PPPoERule(Rule):
    """MPLS分类规则。

    Note:
        强制关键字传参。

    Args:
        sid (str): sid
        code(PathID): code
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> pppoe = PPPoERule(sid="any", code="any", action=1)

    """

    def __init__(self, *, sid: str, code: str, action: PathID):
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_SID] = sid
        self.rule_parameters[_CODE] = code
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = {
            _SID: "sid",
            _CODE: "code",
            _ACTION: "action",
        }
        self._update_rule()

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
    def sid(self) -> str:
        return self.rule_parameters[_SID]

    @sid.setter
    def sid(self, value: str) -> None:
        self.rule_parameters[_SID] = value
        self._update_rule()

    @property
    def code(self) -> str:
        return self.rule_parameters[_CODE]

    @code.setter
    def code(self, value: str) -> None:
        self.rule_parameters[_CODE] = value
        self._update_rule()

    def _update_rule(self):
        self.clear_children()
        children_node_Map = {
            _SID: "",
            _CODE: "",
            _ACTION: self.rule_parameters[_ACTION]}
        properties = {}

        if self.rule_parameters[_SID] == _ANY:
            properties[_SID] = {_ANY: 1}
        else:
            children_node_Map[_SID] = self.rule_parameters[_SID]

        if self.rule_parameters[_CODE] == _ANY:
            properties[_CODE] = {_ANY: 1}
        else:
            children_node_Map[_CODE] = self.rule_parameters[_CODE]

        # TODO: has path
        xt.add_children(self, children_node_Map)
        xt.add_properties(self, properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'PPPoERule':
        """由一个 PPPoE 分类规则的 xml 节点构造 PPPoERule 对象。

        Args:
            node (ET.Element): PPPoE 分类规则的 xml 节点。

        Returns:
            PPPoERule 对象。
        """
        rule = PPPoERule(
            sid=node.findtext(_SID),
            code=node.findtext(_CODE),
            action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule
