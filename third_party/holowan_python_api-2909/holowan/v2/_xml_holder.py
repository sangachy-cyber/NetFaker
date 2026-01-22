"""
HoloWAN Network Emulator
Python API
"""
import xml.etree.cElementTree as ET
from abc import abstractmethod
from typing import Any
from typing import Union, List

from holowan.v2.utils import xml_util as xt


class XMLHolder(ET.Element):
    """XML 基本单元基类
    HoloWAN 的配置通过 xml 格式。各个功能的表现为 xml 配置文件中的一个子节点。
    各个功能的参数为功能节点下的子节点。该类设计为解析、组织 HoloWAN 的 xml 配置文件。
    每一个 XMLHolder只描述、处理父子节点的关系。形如：
         <tag>
             <child1>text<child1>
             ...
             <childn>text<childn>
         <tag>

    XMLHolder 可以进行嵌套。但类中实现的方法为对<tag>下子节点的增删改查(只对子节点)。
    
    """

    def __init__(self, tag_name: str, children: Union[dict, List[ET.Element]] = None) -> None:
        super().__init__(tag_name)
        # TODO: 异常处理
        self.tag = tag_name
        if children != None:
            self.add_children(children)

    def set_node_text(self, target_node: str, value: Any) -> None:
        node = self.find(target_node)
        if isinstance(value, str):
            node.text = value
        else:
            node.text = str(value)

    def remove_child(self, node: ET.Element):
        self.remove(node)

    def clear_children(self):
        for node in list(self):
            self.remove_child(node)

    def set_properties(self, properties: dict):
        xt.add_properties(self, properties)

    def set_property(self, node_path: str, property_name: str, value: Any):
        if node_path == self.tag:
            self.set(property_name, value)
        else:
            self.find(node_path).set(property_name, value)

    def remove_property(self, node: str, property_name: str):
        del self.find(node).attrib[property_name]

    @abstractmethod
    def __str__(self):
        ...

    @property
    def xml(self) -> str:
        return xt.xmlObject_to_string(self)

    def add_child(self, node: ET.Element):
        self.append(node)

    def add_children(self, children: Union[dict, List[ET.Element]]):
        if isinstance(children, dict):
            xt.add_children(self, children)
        elif isinstance(children, list) and len(children) > 0 and isinstance(children[0], ET.Element):
            for node in children:
                self.add_child(node)

    @property
    def node(self) -> ET.Element:
        return self
