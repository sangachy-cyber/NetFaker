"""
HoloWAN Network Emulator
Python API

Packet Classifier: Combination Rule
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import (
    PathID
)
from holowan.v2.engine.classifier._ip_rule import IPv4Rule, IPv6Rule
from holowan.v2.engine.classifier._mac_rule import MACRule
from holowan.v2.engine.classifier._mpls_rule import MPLSRule
from holowan.v2.engine.classifier._pppoe_rule import PPPoERule
from holowan.v2.engine.classifier._raw_byte_rule import RawByteRule
from holowan.v2.engine.classifier._rule_base import Rule, _RULE_ACTION
from holowan.v2.engine.classifier._tcp_udp_sctp_rule import UDPRule, TCPRule, SCTPRule
from holowan.v2.engine.classifier._vlan_rule import VLANRule
from holowan.v2._holowan_types import check_parameter

_RULE_NAME: str = r"comb"
_ACTION: str = _RULE_ACTION


class CombinationRule(Rule):
    """组合分类规则。

    Note:
        强制关键字传参。

    Args:
        action (PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> comb = CombinationRule(action=1)
        >>> comb.add_rule(
        >>>     TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
        >>>     SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
        >>>     MACRule(src="any", dst="any", type="any", action=1),
        >>>     PPPoERule(sid="any", code="any", action=1),
        >>>     MPLSRule(label="any", action=1)
        >>> )
    """

    def __init__(self, *, action: PathID) -> None:
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self._rule_ls = []
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = {
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
        # TODO: update
        self.rule_parameters[_ACTION] = value

    def _update_rule(self):
        self.clear_children()
        self.add_children(self.rule_parameters)
        for node in self._rule_ls:
            self.add_child(node)

    def add_rule(self, *args):
        """添加组合分类规则的子规则。

        Note:
            待添加的规则必须是 Rule 的子类。
        
        Args:
            *args: 可以一次性传入多个规则。
        """
        if len(args) == 1 and isinstance(args, Rule):
            args.remove(args.find(_ACTION))
            self._rule_ls.append(args)
        else:
            for value in args:
                if isinstance(value, Rule):
                    value.remove(value.find(_ACTION))
                    self._rule_ls.append(value)
                else:
                    raise ValueError("Argument must be Rule, but got {got!r}, value {value!r}".format(
                        got=type(value), value=value
                    ))

        self._update_rule()

    def construct_from_node(node: ET.Element) -> 'CombinationRule':
        """由一个组合分类规则的 xml 节点构造 CombinationRule 对象。

        Args:
            node (ET.Element): 组合分类规则的 xml 节点。

        Returns:
            CombinationRule 对象。
        """
        action = int(node.findtext(_ACTION))
        comb = CombinationRule(action=action)
        for sub_rule in list(node):
            rule_name = sub_rule.tag
            sub_rule.append(node.find(_ACTION))
            if rule_name == "mac":
                comb.add_rule(MACRule.construct_from_node(sub_rule))
            elif rule_name == "tcp_udp":
                rule_type = int(sub_rule.findtext("type"))
                if rule_type == 1:
                    comb.add_rule(TCPRule.construct_from_node(sub_rule))
                elif rule_type == 2:
                    comb.add_rule(UDPRule.construct_from_node(sub_rule))
                else:
                    comb.add_rule(SCTPRule.construct_from_node(sub_rule))
            elif rule_name == "mpls":
                comb.add_rule(MPLSRule.construct_from_node(sub_rule))
            elif rule_name == "pppoe":
                comb.add_rule(PPPoERule.construct_from_node(sub_rule))
            elif rule_name == "raw_group":
                comb.add_rule(RawByteRule.construct_from_node(sub_rule))
            elif rule_name == "vlan":
                comb.add_rule(VLANRule.construct_from_node(sub_rule))
            elif rule_name == "ipv4":
                comb.add_rule(IPv4Rule.construct_from_node(sub_rule))
            elif rule_name == "ipv6":
                comb.add_rule(IPv6Rule.construct_from_node(sub_rule))
            elif rule_name == _ACTION:
                pass
            else:
                raise ValueError("Unknown rule type: {0}.".format(rule_name))
        return comb

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.rule_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        for rule in self._rule_ls:
            res += rule.__str__()
            res += " "

        res += "}"
        return res
