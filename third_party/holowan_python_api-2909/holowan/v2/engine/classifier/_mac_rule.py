"""
HoloWAN Network Emulator
Python API

Packet Classifier: MAC Rule
"""

import functools
import inspect
import xml.etree.cElementTree as ET
from typing import Callable, Union

import holowan.v2.utils.my_util as mt
from holowan.v2._holowan_types import (
    PathID, MACAddress, MACAddressList)
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION

_RULE_NAME = HoloWANRuleTag.MAC
_ANY: str = r"any"
_SRC: str = r"src"
_DST: str = r"dst"
_TYPE: str = r"type"
_ACTION: str = _RULE_ACTION

_MAC_DEFAULT_CUSTOM_NAME: str = r"Mac: {} to {}"


# 定制装饰器, 检查 mac 的参数
def _check_mac_rule_parameters(function: Callable):
    msg = 'Argument {argument} must be {expected!r}, but got {got!r}, value {value!r}'

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        new_args = list()
        new_kwargs = {}
        sig = inspect.signature(function)  # 提取函数签名
        params = sig.parameters
        va = list(params.values())
        for arg, param in zip(args, va):
            if type(arg) == int and param.annotation == float:
                arg = float(arg)
            if param.annotation != inspect._empty and not isinstance(arg, param.annotation):
                error = msg.format(argument=param.name, expected=param.annotation.__name__,
                                   got=type(arg), value=arg)
                raise TypeError(error)
            new_args.append(arg)
        for k, v in kwargs.items():
            if k == _SRC or k == _DST:
                if isinstance(v, str):
                    if v != _ANY and mt.isMac(v) == False:
                        raise ValueError(r"Argument {argument!r} is not a valid MACAddress, got {got!r}".format(
                            argument=k, got=v
                        ))
                    else:
                        # int, valid
                        new_kwargs[k] = v
                elif isinstance(v, list):
                    for idx, pt in enumerate(set(v)):
                        if isinstance(pt, str) and mt.isMac(pt):
                            new_kwargs[k] = v
                        else:
                            if pt == _ANY:
                                new_kwargs[k] = v
                            else:
                                raise TypeError(("The value:{value!r} in the {argument!r} MACAddress list "
                                                 "is not a valid MACAddress").format(argument=k, value=pt))
                else:
                    raise TypeError('Argument {argument} must be {expected!r},but got {got!r},value {value!r}'.format(
                        argument=k, expected="MACAddress",
                        got=type(v), value=v
                    ))
            else:
                # TODO:has_path
                if params[k].annotation != inspect._empty and not isinstance(v, params[k].annotation):
                    error = msg.format(argument=params[k].name, expected=params[k].annotation.__name__,
                                       got=type(v), value=v)
                    raise TypeError(error)
                new_kwargs[k] = v
        return function(*tuple(new_args), **dict(new_kwargs))

    return wrapper


class MACRule(Rule):
    """MAC分类规则。

    Note:
        强制关键字传参。

    Args:
        src (IPv6Address): 源 mac 地址。
        dst(IPv6Address): 目的 mac 地址。
        type(str): EtherType类型
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> mac = MACRule(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any", action=1)

    """

    @_check_mac_rule_parameters
    def __init__(self, *, src: Union[MACAddress, MACAddressList],
                 dst: Union[MACAddress, MACAddressList],
                 type: str, action: PathID):
        super().__init__(_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_SRC] = src
        self.rule_parameters[_DST] = dst
        self.rule_parameters[_ACTION] = action
        self.rule_parameters[_TYPE] = type
        self._param_tag2str = {
            _SRC: "src",
            _DST: "dst",
            _ACTION: "action",
            _TYPE: "type"
        }
        self._update_rule()
        self.set_custom_name(_MAC_DEFAULT_CUSTOM_NAME.format(src, dst))

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。

        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @_check_mac_rule_parameters
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> Union[MACAddress, MACAddressList]:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: Union[MACAddress, MACAddressList]) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> Union[MACAddress, MACAddressList]:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: Union[MACAddress, MACAddressList]) -> None:
        self.rule_parameters[_DST] = value
        self._update_rule()

    @property
    def type(self) -> str:
        return self.rule_parameters[_TYPE]

    @type.setter
    def type(self, value: str) -> None:
        self.rule_parameters[_TYPE] = value
        self._update_rule()

    def _update_rule(self):
        self.clear_children()
        properties = {}
        # ===============src================== #
        if self.rule_parameters[_SRC] == _ANY:
            properties[_SRC] = {_ANY: 1}
        # ===============dst================== #
        if self.rule_parameters[_DST] == _ANY:
            properties[_DST] = {_ANY: 1}
        # ===============EtherType================== #
        if self.rule_parameters[_TYPE] == _ANY:
            properties[_TYPE] = {_ANY: 1}
        # ===============Action================== #
        # TODO: has path
        self.add_children(self.rule_parameters)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'MACRule':
        """由一个 MAC 分类规则的 xml 节点构造 MACRule 对象。

        Args:
            node (ET.Element): MAC 分类规则的 xml 节点。

        Returns:
            MACRule 对象。
        """
        if node.find(_SRC).get(_ANY) == "1":
            src = _ANY
        else:
            src = node.findtext(_SRC)
        if node.find(_DST).get(_ANY) == "1":
            dst = _ANY
        else:
            dst = node.findtext(_DST)
        if node.find(_TYPE).get(_ANY) == "1":
            type = _ANY
        else:
            type = node.findtext(_TYPE)
        rule = MACRule(
            src=src,
            dst=dst,
            type=type,
            action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule


if __name__ == '__main__':
    mac = MACRule(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any", action=1)
    mac.action = 1
    print(mac)
