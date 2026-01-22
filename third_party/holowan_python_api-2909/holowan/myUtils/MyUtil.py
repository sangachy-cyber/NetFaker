import hashlib
import re
import requests
import os
from holowan.myUtils import XmlUtil as xt
from configparser import ConfigParser
import socket

MyUtilProxy = {}

def proxy_init(proxy):
    global MyUtilProxy
    MyUtilProxy = proxy


# md5加密
def md5_encryption(raw_data):
    m = hashlib.md5()
    data = raw_data.encode(encoding='utf-8')
    m.update(data)
    data_md5 = m.hexdigest()
    return data_md5


# 压缩xml
def compress_xml(xml):
    return re.sub('>\s+<', "><", xml)


# post调用原始api
def post_original_api(url, body, session=None):
    if url == "" or url is None:
        raise RuntimeError("url为空")
    if body == "" or body is None:
        raise RuntimeError("body为空")
    body = compress_xml(body)
    url_md5 = md5_encryption("/" + url.split("/")[-1])
    body_md5 = md5_encryption(body)
    headers = {
        "URL-MD5": url_md5,
        "Body-MD5": body_md5,
        "Content-Type": "text/xml;charset=UTF-8"
    }
    # 如果提供了 session，使用 session 发送请求（会自动携带 Cookie）
    if session is not None:
        response = session.post(url, data=body, headers=headers, verify=False)
    else:
        # response = requests.post(url, data=body, headers=headers)
        response = requests.post(url, data=body, headers=headers, proxies=MyUtilProxy,verify=False)
    response.encoding = "utf-8"
    return response.text


# get调用原始api
def get_original_api(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    return response.text


# 发送xml数据（xml数据在文件中）
def send_xml_infile(api, xml_file):
    body_data = open(xml_file, encoding='utf-8').read()
    return post_original_api(api, body_data)


# 发送xml数据
def send_xml(api, xml):
    return post_original_api(api, xml)


# 发送get请求，获取数据
def get(api):
    return get_original_api(api)


# 判断文件是否存在
def file_exists(file_dir):
    return os.path.exists(file_dir)


# 删除文件
def del_file(file_dir):
    if file_exists(file_dir):
        os.remove(file_dir)


# 创建文件， 并写入内容
def create_and_write_file(file_dir, text):
    f = open(file_dir, "wb+")
    f.write(text)
    f.close()


# 判断字符串是否为ip地址
def isIP(ipStr: str) -> bool:
    if isIPV4(ipStr) or isIPV6(ipStr):
        return True
    else:
        return False


# 判断字符串是否为IPV4地址
def isIPV4(ipStr: str) -> bool:
    # if re.compile("^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$").match(ipStr):
    #     return True
    # else:
    #     return False
    try:
        socket.inet_pton(socket.AF_INET, ipStr)
    except socket.error:
        return False
    return True


# 判断字符串是否为IPV6地址
def isIPV6(ipStr: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET6, ipStr)
    except socket.error:
        return False
    return True


# 判断字符串是否为端口号
def isPort(portStr: str) -> bool:
    if portStr.isdigit() != True:
        return False
    if int(portStr) >= 1 and int(portStr) <= 65535:
        return True
    else:
        return False


# 判断字符串为MAC地址
def isMac(macStr: str) -> bool:
    if re.compile("^([A-Fa-f0-9]{2}[-,:]){5}[A-Fa-f0-9]{2}$").match(macStr):
        return True
    else:
        return False


# 判断字符串是否为两位十六进制数
def isDoubleHexadecimal(theStr):
    if len(theStr) == 2:
        pattern = "[0-9a-fA-F]{2}"
        p = re.compile(pattern)
        if p.match(theStr):
            return True
        else:
            return False
    else:
        return False


# 判断字符串是否为符合大小的十六进制数
def isHexadecimal(theStr, size):
    if len(theStr) == size * 2:
        pattern = "[0-9a-fA-F]{2}"
        p = re.compile(pattern)
        if p.match(theStr):
            return True
        else:
            return False
    else:
        return False


# 获取HoloWAN_parameter.xml数据
def getProperties(nodeTag):
    project_path = os.path.dirname(os.path.dirname(__file__))
    fileName = r"{}\holowan\resources\HoloWAN_parameter.xml".format(project_path)
    tree = xt.xmlFile_to_Object(fileName)
    return xt.get_node(tree, r"./{}".format(nodeTag)).text


# 判断四位整数
def isFourInteger(num: int) -> bool:
    if isinstance(num, int):
        if len(str(num)) == 4:
            return True
        else:
            return False
    else:
        return False


# 判断是整数
def isInteger(theNum: int) -> bool:
    if isinstance(theNum, int):
        return True
    else:
        raise RuntimeError('{"errCode":"-1","errMsg":"ERROE","errReason":"Parameter type error"}')


# 读取ini文件
def open_ini(filePath):
    config = ConfigParser()
    config.read(filePath)
    return config


# 读取ini文件
def open_ini2(filePath, section):
    config = MyConfigParser()
    config.setSection(section)
    config.read(filePath)
    return config


class MyConfigParser(ConfigParser):
    def __int__(self, section: str):
        ConfigParser.__init__(self)
        self.section = section

    def setSection(self, section: str):
        self.section = section

    def getOption(self, option: str):
        return super(ConfigParser, self).get(self.section, option)


# 通过list里的list元素的长度来排序，输入格式[[],[],...]
def sortListbyLen(doubleList: list) -> list:
    mydict = {}
    returnList = []
    for singleList in doubleList:
        mydict[len(singleList)] = singleList
    while len(mydict) != 0:
        minkey = -1
        for key in mydict.keys():
            if minkey == -1:
                minkey = key
            elif minkey > key:
                minkey = key
        if minkey != -1:
            returnList.append(mydict.get(minkey))
            mydict.pop(minkey)
    return returnList


# 返回包含每一级的路径list和条件list的dict
def ObjectPathAndConditionDict(doubleList: list) -> dict:
    dListLen = len(doubleList)
    pathList = []
    conditionList = []
    lastIndex = 0
    maxIndex = -1
    for index in range(len(doubleList)):
        if index < dListLen - 1:
            maxIndex = indexofMaxSameValue(doubleList[index], doubleList[index + 1])
            pathList.append("/".join(doubleList[index][lastIndex: maxIndex + 1]))
            # if index == 0:
            #     conditionList.append("/".join(doubleList[index][-1]))
            # conditionList.append("/".join(doubleList[index+1][-1]))
            if index == 0:
                conditionList.append(doubleList[index][-1])
            conditionList.append(doubleList[index + 1][-1])
            lastIndex = maxIndex + 1
    return {"pathList": pathList, "conditionList": conditionList}


# 返回两个一维list的从左开始往右算，值相同的最大的index，若遇到值不相同的，则停止
def indexofMaxSameValue(l1: list, l2: list) -> int:
    minLen = -1
    maxSameValueIndex = -1
    if len(l1) < len(l2):
        minLen = l1
    else:
        minLen = l2
    for i in range(len(minLen)):
        l1Value = l1[i]
        l2Value = l2[i]
        if l1Value == l2Value:
            maxSameValueIndex = i
    return maxSameValueIndex


# 获取文件类型
def getFileType(fileName: str) -> str:
    return fileName.split(".")[-1]


# # db格式转换成csv格式
# def dbToCsv(dbFilePath):
#     conn = sqlite3.connect(dbFilePath)
#     cursor = conn.cursor()
#     sql = '''select * from'''

# 验证list中的dict类型是否符合
def validate_list(check_list, rule_list):
    for item in check_list:
        # 验证类型
        if type(item) != dict:
            return False
        # 验证键
        if set(item.keys()) != set(rule_list[0].keys()):
            return False
        # 验证值类型
        for key, val in item.items():
            if type(val) != type(rule_list[0][key]):
                return False
    return True

if __name__ == '__main__':
    s = "ping.db"
    print(getFileType(s))
