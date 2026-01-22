"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

from holowan.v2._holowan_types import (
    PathID,
    check_parameter
)
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION

_RULE_NAME = r"raw"
_TYPE: str = r"type"
_LAYER: str = r"layer"
_OFFSET: str = r"offset"
_MASK: str = r"mask"
_VALUE: str = r"value"
_ACTION: str = _RULE_ACTION

_RAW_GROUP_TAG: str = HoloWANRuleTag.RawByte


class RawByteInternal(Rule):
    def __init__(self, type: int, layer: int, offset: int, mask: str, value: str):
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_TYPE] = type
        self.rule_parameters[_LAYER] = layer
        self.rule_parameters[_OFFSET] = offset
        self.rule_parameters[_MASK] = mask
        self.rule_parameters[_VALUE] = value
        self._param_tag2str = {
            _LAYER: "layer",
            _OFFSET: "offset",
            _MASK: "mask",
            _VALUE: "value"
        }
        self._update_rule()

    def _update_rule(self):
        self.clear_children()
        self.add_children(self.rule_parameters)

    def __str__(self):
        res = "{ "
        for k, v in self.rule_parameters.items():
            if k == _TYPE:
                continue
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)
        res += "}"
        return res


class RawByteRule(Rule):
    """Raw Byte分类规则。

    Note:
        强制关键字传参。

    Args:
        type (str): 1为 Raw Byte 1; 4为 Raw Byte 4。
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> raw1 = RawByteRule(type=1, action=1)
        >>> raw2 = RawByteRule(type=2, action=1)

    """

    @check_parameter
    def __init__(self, *, type: int, action: PathID):
        super(RawByteRule, self).__init__(_RAW_GROUP_TAG)
        super().__setattr__("rule_parameters", {})
        self._type = type
        self.rule_parameters[_ACTION] = action
        self._raw_byte_internal = []
        self._param_tag2str = {
            _ACTION: "action"
        }
        self._update_rule()

    @check_parameter
    def add_raw_byte(self, *, layer: int, offset: int, mask: str, value: str) -> None:
        """为分类规则添加一条匹配项

        Args:
            layer: 网络层
            offset: 偏移量
            mask: 掩码
            value: 待匹配的值

        """
        self._raw_byte_internal.append(RawByteInternal(self._type, layer, offset, mask, value))

    @property
    def node(self) -> ET.Element:
        """返回自身规则的 xml 节点。

        Returns:
            ET.Element
        """
        for rule in self._raw_byte_internal:
            self.add_child(rule.node)
        return self

    @property
    def type(self) -> int:
        """返回类型

        Returns:
            1为 Raw Byte 1; 4为 Raw Byte 4。
        """
        return self._type

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

    def construct_from_node(node: ET.Element):
        """由一个 Raw Byte 分类规则的 xml 节点构造 RawByteRule 对象。

        Args:
            node (ET.Element): Raw Byte 分类规则的 xml 节点。

        Returns:
            RawByteRule 对象。
        """
        internal_raw = node.findall(_RULE_NAME)
        type = int(internal_raw[0].findtext(_TYPE))
        raw_byte_rule = RawByteRule(
            type=type, action=int(node.findtext(_ACTION))
        )
        for raw in internal_raw:
            raw_byte_rule.append(RawByteInternal(
                type=type, layer=int(raw.findtext(_LAYER)),
                offset=int(raw.findtext(_OFFSET)),
                mask=raw.findtext(_MASK),
                value=raw.findtext(_VALUE)
            ))
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        raw_byte_rule.attrib = node.attrib
        return raw_byte_rule

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        res += r"{0}:{1} ".format("type", self._type)
        for k, v in self.rule_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += r"offsets:["
        for interval in self._raw_byte_internal:
            res += interval.__str__()
        res += "]"
        res += "}"
        return res
