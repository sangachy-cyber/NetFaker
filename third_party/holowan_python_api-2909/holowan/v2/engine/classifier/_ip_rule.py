"""
HoloWAN Network Emulator
Python API

Packet Classifier: IPv4 & IPv6 Rule
"""

import functools
import inspect
import xml.etree.cElementTree as ET
from typing import Callable

import holowan.v2.utils.xml_util as xt
from holowan.v2._holowan_types import (
    PathID,
    IPv6Address, IPv4Address
)
from holowan.v2.engine.classifier._rule_base import Rule, HoloWANRuleTag
from holowan.v2.engine.classifier._rule_base import _RULE_ACTION
from holowan.v2.utils import my_util as mt

_IPV4_RULE_NAME: str = HoloWANRuleTag.IPv4
_TUNNEL: str = r"tunnel"
_SRC_MASK: str = r"smask"
_DST_MASK: str = r"dmask"
_TOS: str = r"tos"
_IPV6_RULE_NAME: str = HoloWANRuleTag.IPv6
_ANY: str = r"any"
_SRC: str = r"src"
_DST: str = r"dst"
_TYPE: str = r"type"
_ACTION: str = _RULE_ACTION

_IP_DEFAULT_CUSTOM_NAME: str = r"IPv{}: {} to {}"

_ip_tag2str_dict = {
    _SRC: "src",
    _DST: "dst",
    _SRC_MASK: "smask",
    _DST_MASK: "dmask",
    _TOS: "tos",
    _ACTION: "action",
}


def _check_ip_params(function: Callable):
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
                if params[k].annotation == IPv4Address:
                    if v != _ANY and mt.isIPV4(v) == False:
                        raise ValueError(r"Argument {argument!r} is not a valid IPv4 address, got {got!r}".format(
                            argument=k, got=v
                        ))
                    else:
                        new_kwargs[k] = v
                elif params[k].annotation == IPv6Address:
                    if v != _ANY and mt.isIPV6(v) == False:
                        raise ValueError(r"Argument {argument!r} is not a valid IPv6 address, got {got!r}".format(
                            argument=k, got=v
                        ))
                    else:
                        new_kwargs[k] = v
                else:
                    raise TypeError(
                        msg.format(
                            argument=k, expected="IP address",
                            got=type(v), value=v
                        ))
            else:
                if params[k].annotation != inspect._empty and not isinstance(v, params[k].annotation):
                    error = msg.format(argument=params[k].name, expected=params[k].annotation.__name__,
                                       got=type(v), value=v)
                    raise TypeError(error)

            new_kwargs[k] = v

        return function(*tuple(new_args), **dict(new_kwargs))

    return wrapper


class IPv4Rule(Rule):
    """IPv4分类规则。

    Note:
        强制关键字传参。

    Args:
        src (IPv4Address): 源 ipv4 地址。
        smask(IPv4Address): 源 ipv4 掩码。
        dst(IPv4Address): 目的 ipv4 地址。
        dmask(IPv4Address): 目的 ipv4 掩码。
        tos(str): TOS。
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> ipv4 = IPv4Rule(src="any", smask=32, dst="any", dmask=32, tos="any", action=1)

    """

    @_check_ip_params
    def __init__(self, *, src: IPv4Address, smask: int, dst: IPv4Address, dmask: int, tos: str, action: PathID):
        super().__init__(_IPV4_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_SRC] = src
        self.rule_parameters[_DST] = dst
        self.rule_parameters[_SRC_MASK] = smask
        self.rule_parameters[_DST_MASK] = dmask
        self.rule_parameters[_TOS] = tos
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = _ip_tag2str_dict
        self._update_rule()
        self.set_custom_name(_IP_DEFAULT_CUSTOM_NAME.format(4, src, dst))

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。
        
        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @_check_ip_params
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> IPv4Address:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: IPv4Address) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> IPv4Address:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: IPv4Address) -> None:
        self.rule_parameters[_DST] = value
        self._update_rule()

    @property
    def smask(self) -> int:
        return self.rule_parameters[_SRC_MASK]

    @smask.setter
    def smask(self, value: int) -> None:
        self.rule_parameters[_SRC_MASK] = value
        self._update_rule()

    @property
    def dmask(self) -> int:
        return self.rule_parameters[_DST_MASK]

    @dmask.setter
    def dmask(self, value: int) -> None:
        self.rule_parameters[_DST_MASK] = value
        self._update_rule()

    @property
    def tos(self) -> str:
        return self.rule_parameters[_TOS]

    @tos.setter
    def tos(self, value: str) -> None:
        self.rule_parameters[_TOS] = value
        self._update_rule()

    def _update_rule(self):
        self.clear_children()
        children_node_Map = {
            _SRC: "",
            _SRC_MASK: self.rule_parameters[_SRC_MASK],
            _DST: "",
            _DST_MASK: self.rule_parameters[_DST_MASK],
            _TOS: "",
            _ACTION: self.rule_parameters[_ACTION]
        }
        properties = {}
        # ===============src================== #
        if self.rule_parameters[_SRC] == _ANY:
            properties[_SRC] = {_ANY: 1}
        else:
            children_node_Map[_SRC] = self.rule_parameters[_SRC]
        # ===============dst================== #
        if self.rule_parameters[_DST] == _ANY:
            properties[_DST] = {_ANY: 1}
        else:
            children_node_Map[_DST] = self.rule_parameters[_DST]
        # ===============tos================== #
        if self.rule_parameters[_TOS] == _ANY:
            properties[_TOS] = {_ANY: 1}
        else:
            children_node_Map[_TOS] = self.rule_parameters[_SRC]
        # ===============Action================== #
        # TODO: has path
        xt.add_children(self, children_node_Map)
        xt.add_properties(self, properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'IPv4Rule':
        """由一个组合分类规则的 xml 节点构造 IPv4Rule 对象。

        Args:
            node (ET.Element): IPv4 分类规则的 xml 节点。

        Returns:
            IPv4Rule 对象。
        """
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

        rule = IPv4Rule(
            src=src, smask=int(node.findtext(_SRC_MASK)),
            dst=dst, dmask=int(node.findtext(_DST_MASK)),
            tos=tos, action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule


class IPv6Rule(Rule):
    """IPv6分类规则。

    Note:
        强制关键字传参。

    Args:
        src (IPv6Address): 源 ipv6 地址。
        dst(IPv6Address): 目的 ipv6 地址。
        action(PathID): 被此规则匹配的报文将被转发到的path id。

    Examples:
        >>> ipv6 = IPv6Rule(src="any", dst="any", action=1)

    """

    @_check_ip_params
    def __init__(self, *, src: IPv6Address, dst: IPv6Address, action: PathID):
        super().__init__(_IPV6_RULE_NAME)
        super().__setattr__("rule_parameters", {})
        self.rule_parameters[_SRC] = src
        self.rule_parameters[_DST] = dst
        self.rule_parameters[_ACTION] = action
        self._param_tag2str = _ip_tag2str_dict
        self._update_rule()
        self.set_custom_name(_IP_DEFAULT_CUSTOM_NAME.format(6, src, dst))

    @property
    def action(self) -> PathID:
        """被此规则匹配的报文将被转发到的path id。
        
        Note:
            可读可写。

        """
        return self.rule_parameters[_ACTION]

    @action.setter
    @_check_ip_params
    def action(self, value: PathID) -> None:
        self.rule_parameters[_ACTION] = value
        self._update_rule()

    @property
    def src(self) -> IPv6Address:
        return self.rule_parameters[_SRC]

    @src.setter
    def src(self, value: IPv6Address) -> None:
        self.rule_parameters[_SRC] = value
        self._update_rule()

    @property
    def dst(self) -> IPv6Address:
        return self.rule_parameters[_DST]

    @dst.setter
    def dst(self, value: IPv6Address) -> None:
        self.rule_parameters[_DST] = value
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

        # ===============Action================== #
        # TODO: has path
        self.add_children(self.rule_parameters)
        self.set_properties(properties)

    @staticmethod
    def construct_from_node(node: ET.Element) -> 'IPv6Rule':
        """由一个组合分类规则的 xml 节点构造 IPv6Rule 对象。

        Args:
            node (ET.Element): IPv6 分类规则的 xml 节点。

        Returns:
            IPv6Rule 对象。
        """
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

        rule = IPv6Rule(
            src=src, dst=dst, action=int(node.findtext(_ACTION))
        )
        # 由于 WEB GUI 2.0 支持设置分类规则的 label(是 xml 节点的一个属性)
        # 所以此处处理 xml 节点中的属性, 直接拿过来就行
        # 如果用户想要通过 API 去改规则的名字, 通过新增的 set_custom_name (实现在规则基类 Rule)
        rule.attrib = node.attrib
        return rule


if __name__ == '__main__':
    ip4 = IPv4Rule(src="any", smask=32, dst="any", dmask=32, tos="any", action=1)
    ip4.action = "1"
    print(ip4)
