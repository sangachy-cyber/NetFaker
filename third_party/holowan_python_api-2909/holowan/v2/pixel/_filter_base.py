"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET
from abc import abstractmethod
from typing import List
from typing import Union

from holowan.v2.engine.classifier._rule_base import Rule
from holowan.v2._xml_holder import XMLHolder

_FILTER_ENABLE_PROPERTY: str = r"enable"


class Filter(XMLHolder):
    '''
        继承自 XMLHolder, 父类已实现对子节点的增删改查

        pixel filter 和 holowan packet classifier的 Rule 基本一致, 只有微小的区别
        为保持二者的独立性, 这里重新实现一套 filter

        filter 默认 enable 状态 （enable = "1"）
    '''

    def __init__(self, filter_name: str, filter_parameters: Union[dict, List['Rule']] = None):
        super().__init__(filter_name, filter_parameters)
        self.filter_parameters = filter_parameters
        self._param_tag2str = {}
        self.enable_filter()

    @property
    def enable(self) -> bool:
        flag = self.get(_FILTER_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_filter(self):
        self.set_property(self.tag, _FILTER_ENABLE_PROPERTY, "1")

    def disable_filter(self):
        self.set_property(self.tag, _FILTER_ENABLE_PROPERTY, "0")

    @property
    def filter_name(self):
        return self.tag

    def _update_filter(self):  # TODO: 抽象到基类来实现
        """
        适用于没有嵌套子节点, 没有 property 的更新
        如果有嵌套子节点和 property, 子类需重写此方法
        :return:
        """
        self.clear_children()
        self.add_children(self.filter_parameters)

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element):
        ...

    def __str__(self) -> str:
        res = self.__class__.__name__ + ":{ "
        for k, v in self.filter_parameters.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res
