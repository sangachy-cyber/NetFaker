import xml.etree.cElementTree as ET
from holowan.myUtils import MyUtil as mt
import sys

# xml字符串 -> xml对象
def xmlString_to_Object(xml: str) -> ET.Element:
    return ET.fromstring(xml)


# xml对象 -> xml字符串
def xmlObject_to_string(xmlObject: ET.Element) -> str:
    encoding = 'utf-8'
    if sys.version_info >= (3, 8):
        # xml_declaration 在 python >= 3.8 才有
        return ET.tostring(element=xmlObject, encoding=encoding, method="xml", xml_declaration=True).decode()
    else:
        # 如果 python < 3.8, 但又必须要有 xml_declaration, 走下列代码
        declaration = '<?xml version="1.0" encoding="{}"?>\n'.format(encoding)
        return declaration + ET.tostring(element=xmlObject, encoding=encoding, method="xml").decode()


# xml文件 -> xml对象
def xmlFile_to_Object(xmlFile):
    tree = ET.ElementTree()
    tree.parse(xmlFile)
    return tree

# 创建节点对象
def create_node(tag, propertyMap, text):
    node = ET.Element(tag, propertyMap)
    node.text = text
    return node

# 获取节点对象
def get_node(root, path):
    return root.find(path)

# 获取节点对象（当有多个同名节点时）
def get_nodes(root, path):
    return root.findall(path)


# 删除父节点下的所有子节点
def remove_children(parent):
    children = list(parent)
    for child in children:
        parent.remove(child)
    return parent


# 给父节点添加单个子节点
def add_child(parent, child):
    parent.append(child)


# 给父节点添加多个子节点
def add_children(parent, childrenMap):
    for tag in childrenMap:
        if type(childrenMap[tag]) is str:
            childnode = create_node(tag, {}, childrenMap[tag])
            add_child(parent, childnode)
        elif type(childrenMap[tag]) is int or type(childrenMap[tag]) is float:
            childnode = create_node(tag, {}, str(childrenMap[tag]))
            add_child(parent, childnode)
        elif type(childrenMap[tag]) is dict:
            childnode = create_node(tag, {}, "")
            childnode = add_children(childnode, childrenMap[tag])
            add_child(parent, childnode)
    return parent


# 设置节点文本
def set_node_text(node, text):
    if type(text) is not str:
        node.text = str(text)
    else:
        node.text = text


# 给节点添加属性值
def add_properties(node, propertiesMap):
    enable = node.get("enable")
    if enable != "1":
        node.set("enable", "1")
    for key in propertiesMap:
        key_node = get_node(node, key)
        properties = propertiesMap.get(key)
        for property in properties:
            key_node.set(property, str(properties[property]))


# 对比Classifier节点
def classifier_equal(node1, node2):
    dic1 = {}
    dic2 = {}
    for n1 in node1:
        dic1[n1.tag] = xmlObject_to_string(n1)
    for n2 in node2:
        dic2[n2.tag] = xmlObject_to_string(n2)
    return dic1 == dic2


# 获取xml字符串下特定标签的内容
def getStringTagText(xmlString: str, tagPath: str, condition: dict) -> None:
    """
    :param xmlString: xml数据
    :param tagPath: tag路径
    :param condition: 匹配条件。若有多个匹配结果，则根据匹配条件来匹配，eg.匹配第一个engine的状态{"e/ei": "1"}
    :return: 返回标签内容
    """
    xmlObject = xmlString_to_Object(xmlString)
    # xmlObjectList = []
    # 匹配结果为1
    if len(condition) == 0:
        return get_node(xmlObject, tagPath).text
    # 匹配结果为n
    elif len(condition) != 0:
        # 获取所有条件的路径
        conditionKeyList = []
        for c in condition.keys():
            conditionKeyList.append(c.split("/"))
        # 给list排序
        conditionKeyList = mt.sortListbyLen(conditionKeyList)
        conditionKeyList.append(tagPath.split("/"))

        # 获取每一级的路径list和条件list
        mydict = mt.ObjectPathAndConditionDict(conditionKeyList)
        objectPathList = mydict.get("pathList")
        objectConditionList = mydict.get("conditionList")

        # 获取每一级的条件list
        for index in range(len(objectPathList)):
            # if index < len(objectPathList):
            xmlObjectList = get_nodes(xmlObject, objectPathList[index])
            for xo in xmlObjectList:
                x = xmlObject_to_string(xo)
                a = get_node(xo, objectConditionList[index]).text
                b = str(list(condition.values())[index])
                if get_node(xo, objectConditionList[index]).text == str(list(condition.values())[index]):
                    xmlObject = xo
        node = get_node(xmlObject, objectConditionList[-1])
        if node is None:
            return None
        else:
            return node.text



if __name__ == '__main__':
    xmlString = "<ic><eeq>3</eeq><e><ei>1</ei><es>2</es><epq>1</epq><ep><p><pi>1</pi><pn>PATH 1</pn><ie>2</ie><l>1000.0Mbps/0.0ms/0.0%</l><r>1000.0Mbps/0.0ms/0.0%</r><re>0</re><rd>0</rd></p><p><pi>2</pi><pn>PATH 2</pn><ie>2</ie><l>1000.0Mbps/0.0ms/0.0%</l><r>1000.0Mbps/0.0ms/0.0%</r><re>0</re><rd>0</rd></p></ep><p1 unit='Mbps'>10000</p1><p2 unit='Mbps'>10000</p2></e><e><ei>2</ei><es>2</es><epq>1</epq><ep><p><pi>1</pi><pn>PATH 1</pn><ie>2</ie><l>1000.0Mbps/0.0~50.0ms/0.0%</l><r>1000.0Mbps/0.0ms/0.0%</r><re>0</re><rd>0</rd></p></ep><p1 unit='Mbps'>1000</p1><p2 unit='Mbps'>1000</p2></e><e><ei>3</ei><es>2</es><epq>1</epq><ep><p><pi>1</pi><pn>PATH 1</pn><ie>2</ie><l>1000.0Mbps/0.0ms/0.0%</l><r>1000.0Mbps/0.0ms/0.0%</r><re>0</re><rd>0</rd></p></ep><p1 unit='Mbps'>1000</p1><p2 unit='Mbps'>1000</p2></e></ic>"
    tagPath = "e/ep/p/pn"
    condition = {"e/ei": 1, "e/ep/p/pi": 1}
    result = getStringTagText(xmlString, tagPath, condition)
    print(result)
    # xmlObject = xmlString_to_Object(xmlString)
    # node = get_node(xmlObject, "./e")
    # print(node.text)

