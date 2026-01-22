import binascii
import string
import time
import requests
from holowan.myUtils import XmlUtil as xt, MyUtil as mt
from holowan.myUtils.MyUtil import proxy_init
import os
import json
from functools import wraps
import inspect
import ntpath
import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore",category=InsecureRequestWarning)

# 检测参数类型的类装饰器
class checkParameter(object):
    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            newArgs = list()
            newKwargs = dict()
            sig = inspect.signature(function)  # 提取函数签名
            params = sig.parameters
            va = list(params.values())
            for arg, param in zip(args, va):
                if type(arg) == int and param.annotation == float:
                    arg = float(arg)
                if param.annotation != inspect._empty and not isinstance(arg, param.annotation):
                    raise TypeError("{} parameter type error".format(param.name))
                if param.name == "holowan_ip":
                    # 剥去 https://
                    ipaddr = arg.split("//")[-1]
                    # 去掉 IPv6 地址可能的中括号
                    if ipaddr.startswith("[") and ipaddr.endswith("]"):
                        ipaddr = ipaddr[1:-1]
                    if mt.isIP(ipaddr) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error IP address"}'
                if param.name == "holowan_port":
                    if mt.isPort(arg) is False:
                        return '{"errCode":"-501","errMsg":"Parameter ERROR","errReason":"Error Port"}'
                newArgs.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, int) and params[k].annotation == float:
                    v = float(v)
                if params[k].annotation != inspect._empty and not isinstance(v, params[k].annotation):
                    raise TypeError("{} parameter type error".format(params[k].name))
                if k == "holowan_ip":
                    ipaddr = v.split("//")[-1]
                    # 去掉 IPv6 地址可能的中括号
                    if ipaddr.startswith("[") and ipaddr.endswith("]"):
                        ipaddr = ipaddr[1:-1]
                    if mt.isIP(ipaddr) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error IP address"}'
                if k == "holowan_port":
                    if mt.isPort(v) is False:
                        return '{"errCode":"-501","errMsg":"Parameter ERROR","errReason":"Error Port"}'
                newKwargs[k] = v
            # return function(*args, **kwargs)
            return function(*tuple(newArgs), **dict(newKwargs))
        return wrapper

# Modify参数检测的类装饰器
class checkModify(object):
    project_path = os.path.abspath(os.path.dirname(__file__))
    ini = mt.open_ini(project_path + r"/resources/HoloWAN.ini")
    matchTypeList = ini.get("modify", "matchType").split(", ")
    modifyTypeList = ini.get("modify", "modifyType").split(", ")

    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            global matchSize, modifySize
            sig = inspect.signature(function)  # 提取函数签名
            params = sig.parameters
            va = list(params.values())
            for arg, param in zip(args, va):
                if param.name == "matchSize":
                    matchSize = arg
                if param.name == "modifySize":
                    modifySize = arg
                if param.name == "matchType":
                    if str(arg) not in self.matchTypeList:
                        return '{"errCode":"-43","errMsg":"ERROR","errReason":"Error parameter values ' + param.name + '"}'
                if param.name == "modifyType":
                    if str(arg) not in self.modifyTypeList:
                        return '{"errCode":"-51","errMsg":"ERROR","errReason":"Error parameter values ' + param.name + '"}'
                if param.name == "matchValue":
                    if mt.isHexadecimal(arg, matchSize) is False:
                        return '{"errCode":"-49","errMsg":"ERROR","errReason":"Error parameter values ' + param.name + '"}'
                if param.name == "modifyValue":
                    if mt.isHexadecimal(arg, modifySize) is False:
                        return '{"errCode":"-57","errMsg":"ERROR","errReason":"Error parameter values ' + param.name + '"}'
            for k, v in kwargs.items():
                if k == "matchSize":
                    matchSize = v
                if k == "modifySize":
                    modifySize = v
                if k == "matchType":
                    if str(v) not in self.matchTypeList:
                        return '{"errCode":"-43","errMsg":"ERROR","errReason":"Error parameter values ' + k + '"}'
                if k == "modifyType":
                    if str(v) not in self.modifyTypeList:
                        return '{"errCode":"-51","errMsg":"ERROR","errReason":"Error parameter values ' + k + '"}'
                if k == "matchValue":
                    if mt.isHexadecimal(v, matchSize) is False:
                        return '{"errCode":"-49","errMsg":"ERROR","errReason":"Error parameter values ' + k + '"}'
                if k == "modifyValue":
                    if mt.isHexadecimal(v, modifySize) is False:
                        return '{"errCode":"-57","errMsg":"ERROR","errReason":"Error parameter values ' + k + '"}'
            return function(*args, **kwargs)
        return wrapper


# Pixel过滤器参数的类装饰器
class checkPixelSettings(object):
    project_path = os.path.abspath(os.path.dirname(__file__))
    ini = mt.open_ini(project_path + r"/resources/HoloWAN.ini")
    enableList = ini.get("PixelSettingsEnable", "enable").split(", ")
    pathIDList = ini.get("PixelSettingsPathID", "pathID").split(", ")
    directionList = ini.get("PixelSettingsDirection", "direction").split(", ")
    typeBeforeList = ini.get("PixelSettingsType", "before").split(", ")
    typeAfterList = ini.get("PixelSettingsType", "after").split(", ")
    typeCls_dropList = ini.get("PixelSettingsType", "cls_drop").split(", ")
    typeAllList = ini.get("PixelSettingsType", "all").split(", ")
    typeNo_impairList = ini.get("PixelSettingsType", "no_impair").split(", ")
    typeModList = ini.get("PixelSettingsType", "mod").split(", ")
    typeFrg_afList = ini.get("PixelSettingsType", "frg_af").split(", ")
    typeFrg_beList = ini.get("PixelSettingsType", "frg_be").split(", ")
    typeDupList = ini.get("PixelSettingsType", "dup").split(", ")
    typeBw_dropList = ini.get("PixelSettingsType", "bw_drop").split(", ")
    typeLosList = ini.get("PixelSettingsType", "los").split(", ")
    typeBerList = ini.get("PixelSettingsType", "ber").split(", ")
    typeReoList = ini.get("PixelSettingsType", "reo").split(", ")
    typeBypassList = ini.get("PixelSettingsType", "bypass").split(", ")
    VLANEnable_stagList = ini.get("PixelSettingsVLAN", "enable_stag").split(", ")
    VLANFpcpList = ini.get("PixelSettingsVLAN", "fpcp").split(", ")
    VLANSpcpList = ini.get("PixelSettingsVLAN", "spcp").split(", ")
    IPv4SmarkList = ini.get("PixelSettingsIPv4", "smark").split(", ")
    IPv4DmarkList = ini.get("PixelSettingsIPv4", "dmark").split(", ")
    TCP_UDPTypeList = ini.get("PixelSettingsTCP/UDP", "type").split(", ")
    TCP_UDPCheckList = ini.get("PixelSettingsTCP/UDP", "check").split(", ")

    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(function)   # 提取函数签名
            params = sig.parameters
            va = list(params.values())
            for arg, param in zip(args, va):
                if param.name == "enable":
                    if str(arg) not in self.enableList:
                        return '{"errCode":"-502","errMsg":"ERROR","errReason": "The value of "enable" can only be 0 or 1"}'
                if param.name == "pathID":
                    if str(arg) not in self.pathIDList:
                        return '{"errCode":"-503","errMsg":"ERROR","errReason": "The value range of "pathID" is 1 to 15"}'
                if param.name == "direction":
                    if str(arg) not in self.directionList:
                        return '{"errCode":"-504","errMsg":"ERROR","errReason": "The value of "direction" can only be 0 or 1"}'
                if param.name == "before":
                    if str(arg) not in self.typeBeforeList:
                        return '{"errCode":"-505","errMsg":"ERROR","errReason": "The value of "before" can only be 0 or 1"}'
                if param.name == "after":
                    if str(arg) not in self.typeAfterList:
                        return '{"errCode":"-506","errMsg":"ERROR","errReason": "The value of "after" can only be 0 or 1"}'
                if param.name == "cls_drop":
                    if str(arg) not in self.typeCls_dropList:
                        return '{"errCode":"-507","errMsg":"ERROR","errReason": "The value of "cls_drop" can only be 0 or 1"}'
                if param.name == "all":
                    if str(arg) not in self.typeAllList:
                        return '{"errCode":"-508","errMsg":"ERROR","errReason": "The value of "all" can only be 0 or 1"}'
                if param.name == "no_impair":
                    if str(arg) not in self.typeNo_impairList:
                        return '{"errCode":"-509","errMsg":"ERROR","errReason": "The value of "no_impair" can only be 0 or 1"}'
                if param.name == "mod":
                    if str(arg) not in self.typeModList:
                        return '{"errCode":"-510","errMsg":"ERROR","errReason": "The value of "mod" can only be 0 or 1"}'
                if param.name == "frg_af":
                    if str(arg) not in self.typeFrg_afList:
                        return '{"errCode":"-511","errMsg":"ERROR","errReason": "The value of "frg_af" can only be 0 or 1"}'
                if param.name == "frg_be":
                    if str(arg) not in self.typeFrg_beList:
                        return '{"errCode":"-512","errMsg":"ERROR","errReason": "The value of "frg_be" can only be 0 or 1"}'
                if param.name == "dup":
                    if str(arg) not in self.typeDupList:
                        return '{"errCode":"-513","errMsg":"ERROR","errReason": "The value of "dup" can only be 0 or 1"}'
                if param.name == "bw_drop":
                    if str(arg) not in self.typeBw_dropList:
                        return '{"errCode":"-514","errMsg":"ERROR","errReason": "The value of "bw_drop" can only be 0 or 1"}'
                if param.name == "los":
                    if str(arg) not in self.typeLosList:
                        return '{"errCode":"-515","errMsg":"ERROR","errReason": "The value of "los" can only be 0 or 1"}'
                if param.name == "ber":
                    if str(arg) not in self.typeBerList:
                        return '{"errCode":"-516","errMsg":"ERROR","errReason": "The value of "ber" can only be 0 or 1"}'
                if param.name == "reo":
                    if str(arg) not in self.typeReoList:
                        return '{"errCode":"-517","errMsg":"ERROR","errReason": "The value of "reo" can only be 0 or 1"}'
                if param.name == "bypass":
                    if str(arg) not in self.typeBypassList:
                        return '{"errCode":"-518","errMsg":"ERROR","errReason": "The value of "bypass" can only be 0 or 1"}'
                if param.name == "enable_stag":
                    if str(arg) not in self.VLANEnable_stagList:
                        return '{"errCode":"-519","errMsg":"ERROR","errReason": "The value of "enable_stag" can only be 0 or 1"}'
                if param.name == "fpcp":
                    if str(arg) not in self.VLANFpcpList:
                        return '{"errCode":"-520","errMsg":"ERROR","errReason": "The value range of "fpcp" is 0 to 7"}'
                if param.name == "spcp":
                    if str(arg) not in self.VLANSpcpList:
                        return '{"errCode":"-521","errMsg":"ERROR","errReason": "The value range of "spcp" is 0 to 7"}'
                if param.name == "smark":
                    if str(arg) not in self.IPv4SmarkList:
                        return '{"errCode":"-522","errMsg":"ERROR","errReason": "The value of "smark" can only be 8, 16, 24, 32"}'
                if param.name == "dmark":
                    if str(arg) not in self.IPv4DmarkList:
                        return '{"errCode":"-523","errMsg":"ERROR","errReason": "The value of "dmark" can only be 8, 16, 24, 32"}'
                if param.name == "type":
                    if str(arg) not in self.TCP_UDPTypeList:
                        return '{"errCode":"-524","errMsg":"ERROR","errReason": "The value of "type" can only be 1, 2, 3"}'
                if param.name == "check":
                    if str(arg) not in self.TCP_UDPCheckList:
                        return '{"errCode":"-525","errMsg":"ERROR","errReason": "The value of "check" can only be 0, 4, 6"}'
            for k, v in kwargs.items():
                if k == "enable":
                    if str(v) not in self.enableList:
                        return '{"errCode":"-502","errMsg":"ERROR","errReason": "The value of "enable" can only be 0 or 1"}'
                if k == "pathID":
                    if str(v) not in self.pathIDList:
                        return '{"errCode":"-503","errMsg":"ERROR","errReason": "The value range of "pathID" is 1 to 15"}'
                if k == "direction":
                    if str(v) not in self.directionList:
                        return '{"errCode":"-504","errMsg":"ERROR","errReason": "The value of "direction" can only be 0 or 1"}'
                if k == "before":
                    if str(v) not in self.typeBeforeList:
                        return '{"errCode":"-505","errMsg":"ERROR","errReason": "The value of "before" can only be 0 or 1"}'
                if k == "after":
                    if str(v) not in self.typeAfterList:
                        return '{"errCode":"-506","errMsg":"ERROR","errReason": "The value of "after" can only be 0 or 1"}'
                if k == "cls_drop":
                    if str(v) not in self.typeCls_dropList:
                        return '{"errCode":"-507","errMsg":"ERROR","errReason": "The value of "cls_drop" can only be 0 or 1"}'
                if k == "all":
                    if str(v) not in self.typeAllList:
                        return '{"errCode":"-508","errMsg":"ERROR","errReason": "The value of "all" can only be 0 or 1"}'
                if k == "no_impair":
                    if str(v) not in self.typeNo_impairList:
                        return '{"errCode":"-509","errMsg":"ERROR","errReason": "The value of "no_impair" can only be 0 or 1"}'
                if k == "mod":
                    if str(v) not in self.typeModList:
                        return '{"errCode":"-510","errMsg":"ERROR","errReason": "The value of "mod" can only be 0 or 1"}'
                if k == "frg_af":
                    if str(v) not in self.typeFrg_afList:
                        return '{"errCode":"-511","errMsg":"ERROR","errReason": "The value of "frg_af" can only be 0 or 1"}'
                if k == "frg_be":
                    if str(v) not in self.typeFrg_beList:
                        return '{"errCode":"-512","errMsg":"ERROR","errReason": "The value of "frgbe" can only be 0 or 1"}'
                if k == "dup":
                    if str(v) not in self.typeDupList:
                        return '{"errCode":"-513","errMsg":"ERROR","errReason": "The value of "dup" can only be 0 or 1"}'
                if k == "bw_drop":
                    if str(v) not in self.typeBw_dropList:
                        return '{"errCode":"-514","errMsg":"ERROR","errReason": "The value of "bw_drop" can only be 0 or 1"}'
                if k == "los":
                    if str(v) not in self.typeLosList:
                        return '{"errCode":"-515","errMsg":"ERROR","errReason": "The value of "los" can only be 0 or 1"}'
                if k == "ber":
                    if str(v) not in self.typeBerList:
                        return '{"errCode":"-516","errMsg":"ERROR","errReason": "The value of "ber" can only be 0 or 1"}'
                if k == "reo":
                    if str(v) not in self.typeReoList:
                        return '{"errCode":"-517","errMsg":"ERROR","errReason": "The value of "reo" can only be 0 or 1"}'
                if k == "bypass":
                    if str(v) not in self.typeBypassList:
                        return '{"errCode":"-518","errMsg":"ERROR","errReason": "The value of "bypass" can only be 0 or 1"}'
                if k == "enable_stag":
                    if str(v) not in self.VLANEnable_stagList:
                        return '{"errCode":"-519","errMsg":"ERROR","errReason": "The value of "enable_stag" can only be 0 or 1"}'
                if k == "fpcp":
                    if str(v) not in self.VLANFpcpList:
                        return '{"errCode":"-520","errMsg":"ERROR","errReason": "The value range of "fpcp" is 0 to 7"}'
                if k == "spcp":
                    if str(v) not in self.VLANSpcpList:
                        return '{"errCode":"-521","errMsg":"ERROR","errReason": "The value range of "spcp" is 0 to 7"}'
                if k == "smark":
                    if str(v) not in self.IPv4SmarkList:
                        return '{"errCode":"-522","errMsg":"ERROR","errReason": "The value of "smark" can only be 8, 16, 24, 32"}'
                if k == "dmark":
                    if str(v) not in self.IPv4DmarkList:
                        return '{"errCode":"-523","errMsg":"ERROR","errReason": "The value of "dmark" can only be 8, 16, 24, 32"}'
                if k == "type":
                    if str(v) not in self.TCP_UDPTypeList:
                        return '{"errCode":"-524","errMsg":"ERROR","errReason": "The value of "type" can only be 1, 2, 3"}'
                if k == "check":
                    if str(v) not in self.TCP_UDPCheckList:
                        return '{"errCode":"-525","errMsg":"ERROR","errReason": "The value of "check" can only be 0, 4, 6"}'
            return function(*args, **kwargs)
        return wrapper



class HoloWAN:

    def __init__(self, proxy=None):
        if proxy is None:
            proxy = {}
        self.project_path = os.path.abspath(os.path.dirname(__file__))
        self.path_crud_fileName = r"{}/resources/path_crud.xml".format(self.project_path)
        self.path_config_fileName = r"{}/resources/path_config.xml".format(self.project_path)
        self.recorder_config_fileName = r"{}/resources/recorder_config.xml".format(self.project_path)
        self.get_HoloWAN_information_api = "statistics_information"
        self.emulator_config_api = "emulator_config"
        self.history_data_count = "history_data_count"
        self.recorder_config_api = "recorder_config"

        # 初始化配置信息
        self.ini = mt.open_ini(self.project_path + r"/resources/HoloWAN.ini")
        self.matchTypeList = self.ini.get("modify", "matchType").split(", ")
        self.modifyTypeList = self.ini.get("modify", "modifyType").split(", ")

        self.modifyRandomRateDecimal = self.ini.get("modify", "modifyRandomRateDecimal")
        self.constantDelayDecimal = self.ini.get("delay", "constantDelayDecimal")
        self.uniformMinimumDecimal = self.ini.get("delay", "uniformMinimumDecimal")
        self.uniformMaximumDecimal = self.ini.get("delay", "uniformMaximumDecimal")
        self.normalMinDecimal = self.ini.get("delay", "normalMinDecimal")
        self.normalMeanDecimal = self.ini.get("delay", "normalMeanDecimal")
        self.normalStdDeviationDecimal = self.ini.get("delay", "normalStdDeviationDecimal")
        self.normalAdvancedPeriodDecimal = self.ini.get("delay", "normalAdvancedPeriodDecimal")
        self.normalAdvancedDurationDecimal = self.ini.get("delay", "normalAdvancedDurationDecimal")
        self.normalAdvancedMinDecimal = self.ini.get("delay", "normalAdvancedMinDecimal")
        self.normalAdvancedMaxDecimal = self.ini.get("delay", "normalAdvancedMaxDecimal")
        self.customMeanDelayDecimal = self.ini.get("delay", "customMeanDelayDecimal")
        self.customMinDelayDecimal = self.ini.get("delay", "customMinDelayDecimal")
        self.customMaxDelayDecimal = self.ini.get("delay", "customMaxDelayDecimal")
        self.customPositiveDeltaDecimal = self.ini.get("delay", "customPositiveDeltaDecimal")
        self.customNegativeDeltaDecimal = self.ini.get("delay", "customNegativeDeltaDecimal")
        self.customSpreadDeltaDecimal = self.ini.get("delay", "customSpreadDeltaDecimal")
        self.jitterMeanDelayDecimal = self.ini.get("delay", "jitterMeanDelayDecimal")
        self.jitterMeanJitterDecimal = self.ini.get("delay", "jitterMeanJitterDecimal")
        self.randomRateDecimal = self.ini.get("loss", "randomRateDecimal")
        self.burstProbabilityDecimal = self.ini.get("loss", "burstProbabilityDecimal")
        self.dualGoodStateLossDecimal = self.ini.get("loss", "dualGoodStateLossDecimal")
        self.dualGoodToBadProbabilityDecimal = self.ini.get("loss", "dualGoodToBadProbabilityDecimal")
        self.dualBadStateLossDecimal = self.ini.get("loss", "dualBadStateLossDecimal")
        self.dualBadToGoodProbabilityDecimal = self.ini.get("loss", "dualBadToGoodProbabilityDecimal")
        self.markovP13Decimal = self.ini.get("loss", "markovP13Decimal")
        self.markovP31Decimal = self.ini.get("loss", "markovP31Decimal")
        self.markovP32Decimal = self.ini.get("loss", "markovP32Decimal")
        self.markovP23Decimal = self.ini.get("loss", "markovP23Decimal")
        self.markovP14Decimal = self.ini.get("loss", "markovP14Decimal")
        self.markovP13Max = self.ini.get("loss", "markovP13Max")
        self.markovP13Min = self.ini.get("loss", "markovP13Min")
        self.markovP31Max = self.ini.get("loss", "markovP31Max")
        self.markovP31Min = self.ini.get("loss", "markovP31Min")
        self.markovP32Max = self.ini.get("loss", "markovP32Max")
        self.markovP32Min = self.ini.get("loss", "markovP32Min")
        self.markovP23Max = self.ini.get("loss", "markovP23Max")
        self.markovP23Min = self.ini.get("loss", "markovP23Min")
        self.markovP14Max = self.ini.get("loss", "markovP14Max")
        self.markovP14Min = self.ini.get("loss", "markovP14Min")
        self.normalProbabilityDecimal = self.ini.get("recorder", "normalProbabilityDecimal")
        self.normalDelayMinDecimal = self.ini.get("recorder", "normalDelayMinDecimal")
        self.normalDelayMaxDecimal = self.ini.get("recorder", "normalDelayMaxDecimal")
        self.duplicationNormalProbabilityDecimal = self.ini.get("duplication", "duplicationNormalProbabilityDecimal")
        self.etherType =self.ini.get("mac", "etherType")

        # recorder配置参数
        self.packetLossCycleList = self.ini.get("recorder", "packetLossCycle").split(", ")

        self.IPV4 = 0
        self.IPV6 = 1
        self.MAC = 2
        self.TCP = 3
        self.UDP = 4
        self.IPV4ANDTCP = 5
        self.IPV4ANDUDP = 6
        self.RAW1BYTE = 7
        self.RAW4BYTE = 8
        self.SCTP = 9
        self.IPV4ANDSCTP = 10
        self.IPV4ANDSCTP = 11
        self.TCPRANGE = 12
        self.UDPRANGE = 13


        self.ADD = 0
        self.REMOVE = 1
        self.REMOVEALL = 2

        self.REMOVEPATH = 1
        self.OPENORCLOSEPATH = 3
        self.CLOSEPATH = 1
        self.OPENPATH = 2

        # 错误码[链路增删配置]
        self.ini2 = mt.open_ini2(self.project_path + r"/resources/HoloWAN.ini", "PathOperateErrorCode")
        self.POengineIDError = self.ini2.getOption("engineIDError")
        self.POpathIDError = self.ini2.getOption("pathIDError")
        self.POpathNameError = self.ini2.getOption("pathIDError")
        self.POremovePathNotFound = self.ini2.getOption("removePathNotFound")
        self.POpathExist = self.ini2.getOption("pathExist")
        self.POopenPathNotFound = self.ini2.getOption("openPathNotFound")
        self.POclosePathNotFound = self.ini2.getOption("closePathNotFound")
        self.POpathIsOpening = self.ini2.getOption("pathIsOpening")

        # 错误码[链路损伤配置]
        self.ini2.setSection("PathConfigErrorCode")
        self.PCengineIDError = self.ini2.getOption("engineIDError")
        self.PCpathIDError = self.ini2.getOption("pathIDError")
        self.PCpathNotFound = self.ini2.getOption("pathNotFound")
        self.PCpathNameTooLong = self.ini2.getOption("pathNameTooLong")
        self.PCpathDirectionError = self.ini2.getOption("pathDirectionError")

        # 代理配置
        self.http_proxy = proxy
        proxy_init(proxy)
        
        # Session 对象用于保存 Cookie
        self.session = requests.Session()
        if proxy:
            self.session.proxies = proxy

    ''' =============================================身份验证相关接口================================================='''
    # 引擎锁
    @checkParameter()
    def hold_engine(self, holowan_ip: str, holowan_port: str, engineID: int, password: int, new_password: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param password: 原密码
        :param new_password: 新密码
        :return: 功能：1.检测锁状态 输入[password=0, new_password=0] return [0:无锁, -4:锁]
                      2.开     锁 输入[password=0或xxxx, new_password=xxxx] return [0:成功, -4:password错误]
                      3.设置新密码 输入[password=0或xxxx, new_password=xxxx] return [0:成功, -4:password错误]
                      4.清除 密码  输入[password=4123456789, new_password=4123456789] return [0:成功, -4:password错误]

        '''
        try:
            isCheck = (password == 0 and new_password == 0)
            isOpen = (mt.isFourInteger(password) and new_password == 0)
            isSet = (mt.isFourInteger(password) or password == 0) and mt.isFourInteger(new_password)
            isClean = (password == 4123456789 and new_password == 4123456789)
            if (isCheck or isOpen or isSet or isClean) is False:
                raise RuntimeError(r'{"errCode":"-4","errMsg":"error","errReason":"password error"}')
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}//hold_engine?engine={2}&passwd={3}&new_passwd={4}".format(holowan_ip, holowan_port, engineID, password, new_password)
            return self.session.get(requestURL, verify=False).text
        except requests.exceptions.ConnectionError:
            return r'{"errCode":"-200","errMsg":"ConnectionError","errReason":"Failed to establish a new connection"}'
        except RuntimeError as r:
            return r

    # 登录接口
    @checkParameter()
    def login(self, holowan_ip: str, holowan_port: str, username: str = "admin", password: str = "holowan"):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param username: 用户名，默认为 "admin"
        :param password: 密码，默认为 "holowan"
        :return: 登录响应结果，成功后会保存 Cookie session_id 到 Session 中
        '''
        try:
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/login".format(holowan_ip, holowan_port)
            json_data = {"username": username, "password": password}
            response = self.session.post(requestURL, json=json_data, verify=False)
            # Session 会自动保存响应中的 Cookie
            return response.text
        except requests.exceptions.ConnectionError:
            return r'{"errCode":"-200","errMsg":"ConnectionError","errReason":"Failed to establish a new connection"}'
        except Exception as e:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"' + str(e) + '"}'

    ''' ===============================================损伤参数相关接口===================================================='''
    # 获取当前HoloWAN设备xml信息
    @checkParameter()
    def get_HoloWAN_information(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: 当前HoloWAN设备xml信息，详见附录一（1.1设备概述信息XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.get_HoloWAN_information_api)
        return self.session.get(requestURL, verify=False).text

    # 新增虚拟链路path
    @checkParameter()
    def add_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathName: str = "PATH"):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathName: 链路名
        :return: json 正确：errCode 等于 0，错误：errCode小于0（详见附录二（1.1修改虚拟链路配置错误编号））
        '''
        if pathName == "PATH":
            pathName = pathName+" "+str(pathID)
        children_node_Map = {"modify_switch": "2", "engine_id": engineID, "path_id": pathID,
                             "path_name": pathName, "if_enable": "1"}
        node = xt.get_node(xt.xmlFile_to_Object(self.path_crud_fileName), ".")
        xt.remove_children(node)
        xt.add_children(node, children_node_Map)
        XMLStr = xt.xmlObject_to_string(node)
        add_path_return = mt.post_original_api(self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api), XMLStr, self.session)
        if json.loads(add_path_return)["errCode"] == "0":
            return self.init_path(holowan_ip, holowan_port, engineID, pathID, pathName)
        else:
            return add_path_return

    # 新增虚拟链路path并enable
    @checkParameter()
    def add_path_and_enable(self, holowan_ip: str, holowan_port: str, engineID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :return: json 正确：code 等于 0，错误：code 小于 0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/add_path?eid={2}".format(holowan_ip, holowan_port, engineID)
        return self.session.get(requestURL, verify=False).text

    # 初始化链路, 这个函数只在add_path函数中调用
    @checkParameter()
    def init_path(self, holowan_ip:str, holowan_port: str, engineID: int, pathID: int, pathName: str):
        root = xt.get_node(xt.xmlFile_to_Object(self.path_config_fileName), ".")
        xt.add_children(root, {"version": "0"})
        xt.set_node_text(xt.get_node(root, "./eid"), engineID)
        xt.set_node_text(xt.get_node(root, "./pid"), pathID)
        xt.set_node_text(xt.get_node(root, "./pn"), pathName)
        pathXMLStr = xt.xmlObject_to_string(root)
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/emulator_config.xml".format(holowan_ip, holowan_port)
        return mt.post_original_api(requestURL, pathXMLStr, self.session)

    # 删除虚拟链路path
    @checkParameter()
    def remove_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathName: str,
                    force: bool = False):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathName: 链路名
        :param force: 是否强制删除
        :return: json 正确：errCode 等于 0，错误：errCode小于0（详见附录二（1.1修改虚拟链路配置错误编号））
        '''
        children_node_Map = {"modify_switch": "1", "engine_id": engineID, "path_id": pathID, "path_name": pathName,
                             "if_enable": "1"}
        if self.has_engine(holowan_ip, holowan_port, engineID) is False:
            return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE is not found"}}'.format(self.POengineIDError)
        elif self.has_path(holowan_ip, holowan_port, engineID, pathID) is False:
            if pathID > 15 or pathID <= 0:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH Error"}}'.format(self.POpathIDError)
            else:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH is not found"}}'.format(self.POremovePathNotFound)
        if force is False:
            if self.path_is_open(holowan_ip, holowan_port, engineID, pathID):
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH is enabled, can not be deleted"}}'.format(self.POpathIsOpening)
        return self.remove_or_open_or_close_path(holowan_ip, holowan_port, engineID, pathID, pathName,
                                                 children_node_Map)

    # 开启虚拟链路path
    @checkParameter()
    def open_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathName: str):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathName: 链路名
        :return: json 正确：errCode 等于 0，错误：errCode小于0（详见附录二（1.1修改虚拟链路配置错误编号））
        """
        if self.has_engine(holowan_ip, holowan_port, engineID) is False:
            return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE is not found"}}'.format(self.POengineIDError)
        elif self.has_path(holowan_ip, holowan_port, engineID, pathID) is False:
            if pathID > 15 or pathID <= 0:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH Error"}}'.format(self.POpathIDError)
            else:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH is not found"}}'.format(self.POopenPathNotFound)
        if pathName == "PATH" or pathName == None:
            pathName = self.get_path_Name(holowan_ip, holowan_port, engineID, pathID)
        children_node_Map = {"modify_switch": "3", "engine_id": engineID, "path_id": pathID, "path_name": pathName,
                             "if_enable": "2"}
        return self.remove_or_open_or_close_path(holowan_ip, holowan_port, engineID, pathID, pathName,
                                                 children_node_Map)

    # 关闭虚拟链路path
    @checkParameter()
    def close_path(self, holowan_ip:str, holowan_port:str, engineID:int, pathID:int, pathName: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathName: 链路名
        :return: json 正确：errCode 等于 0，错误：errCode小于0（详见附录二（1.1修改虚拟链路配置错误编号））
        '''
        if self.has_engine(holowan_ip, holowan_port, engineID) is False:
            return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE is not found"}}'.format(self.POengineIDError)
        elif self.has_path(holowan_ip, holowan_port, engineID, pathID) is False:
            if pathID > 15 or pathID <= 0:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH Error"}}'.format(self.POpathIDError)
            else:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH is not found"}}'.format(self.POclosePathNotFound)
        if pathName == "PATH" or pathName == None:
            pathName = self.get_path_Name(holowan_ip, holowan_port, engineID, pathID)
        children_node_Map = {"modify_switch": "3", "engine_id": engineID, "path_id": pathID, "path_name": pathName,
                             "if_enable": "1"}
        return self.remove_or_open_or_close_path(holowan_ip, holowan_port, engineID, pathID, pathName,
                                                 children_node_Map)

    # 删除/开启/关闭虚拟链路path共同代码
    def remove_or_open_or_close_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathName, children_node_Map):
        root = xt.xmlFile_to_Object(self.path_crud_fileName)
        node = xt.get_node(root, ".")
        xt.remove_children(node)
        xt.add_children(node, children_node_Map)
        XMLStr = xt.xmlObject_to_string(node)
        return mt.post_original_api(self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api), XMLStr, self.session)

    # 设置虚拟链路（PATH）上下行方向
    @checkParameter()
    def set_path_direction(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1. 仅损伤下行，2. 仅损伤上行，3. 损伤上下行]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        tagPath = "pd"
        tagText = pathDirection
        return self.set_one_tag_text(holowan_ip, holowan_port, engineID, pathID, tagPath, tagText)

    # 设置path带宽，Fixed正常模式
    @checkParameter()
    def set_path_Bandwidth_Fixed(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                 pathDirection: int, rateValue: float, rateUnit: int):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1. 仅损伤下行，2. 仅损伤上行]
        :param rateValue: 带宽限制值
        :param rateUnit: 带宽单位  [1.bps, 2.Kbps, 3.Mbps]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "bd"
        children_node_Map = {"s": 1, "fi": {"r": rateValue, "t": rateUnit}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置path带宽，Token Bucket模式
    @checkParameter()
    def set_path_Bandwidth_Token_Bucket(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                          pathDirection: int, bucket_type: int, air: float, air_t: int,
                                          abs: float, abs_t: int, bir: float, bir_t: int, bbs: float, bbs_t: int):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1. 仅损伤下行，2. 仅损伤上行]
        :param bucket_type: 令牌桶算法类型 [1. 单令牌桶算法，2. 双令牌桶(srTCM)算法，3. 双令牌桶(trTCM)算法]
        :param air: A桶令牌的产生速率
        :param air_t: A桶令牌的产生速率的单位  [1.bps, 2.Kbps, 3.Mbps]
        :param abs: A桶可容纳的最大令牌数
        :param abs_t: A桶可容纳的最大令牌数的单位，[1. B，2. KB，3. MB]
        :param bir: B桶令牌的产生速率
        :param bir_t: B桶令牌的产生速率的单位  [1.bps, 2.Kbps, 3.Mbps]
        :param bbs: B桶可容纳的最大令牌数
        :param bbs_t: B桶可容纳的最大令牌数的单位，[1. B，2. KB，3. MB]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "bd"
        # 需要判断类型, 然后children_mode_Map再作出相应的调整
        if bucket_type == 1:
            children_node_Map = {"s": 3, "tb": {"s": bucket_type, "air": air, "air_t": air_t, "abs": abs,
                                                "abs_t": abs_t}}
        elif bucket_type == 2:
            children_node_Map = {"s": 3, "tb": {"s": bucket_type, "air": air, "air_t": air_t, "abs": abs,
                                                "abs_t": abs_t, "bbs": bbs, "bbs_t": bbs_t}}
        else:
            children_node_Map = {"s": 3,
                                 "tb": {"s": bucket_type, "air": air, "air_t": air_t, "abs": abs,"abs_t": abs_t,
                                        "bir": bir, "bir_t": bir_t, "bbs": bbs, "bbs_t": bbs_t}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    def set_path_Bandwidth_Jitter(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, rate_type: int, MAX: float, MIN: float, jitter_type: int,
                                  Phase: int = 0, Period: int = 60,
                                  Rise: int = 30, Fall: int = 20, Section: int = 50):
        """
        :param Section:
        :param Fall:
        :param Rise:
        :param Period:
        :param Phase:
        :param MAX:2
        :param MIN:
        :param jitter_type:
        :param rate_type:
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1. 仅损伤下行，2. 仅损伤上行]
        """

        if rate_type < 1 or rate_type > 3:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"rate type error"}'

        parent_node_ID = "bd"
        children_node_Map = {"s": 2, "ji": {"t": rate_type, "shake": {"max": MAX, "min": MIN, "cycle": Period}}}

        properties_Map = {"ji/shake": {"type_id": jitter_type}}

        # 需要判断类型, 然后children_mode_Map再作出相应的调整
        if jitter_type == 1:
            children_node_Map["ji"]["shake"]["phase"] = float(Phase / 100)
        elif jitter_type == 2:
            children_node_Map["ji"]["shake"]["proportion_up"] = float(Rise / 100)
            children_node_Map["ji"]["shake"]["proportion_down"] = float(Fall / 100)
        # elif jitter_type == 3 or jitter_type == 4:
            #  3 和 4啥都不用做
        elif jitter_type == 5 or jitter_type == 6:
            children_node_Map["ji"]["shake"]["proportion"] = float(Section / 100)

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    #  关闭背景流量 [s=1: 关闭]
    @checkParameter()
    def close_Background_Utilization(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                     pathDirection: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        try:
            return self.close_damage_config(holowan_ip, holowan_port, engineID, pathID, pathDirection, "/bg/s")
        except RuntimeError as e:
            return e

    # 设置背景流量
    @checkParameter()
    def set_Background_Utilization(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                   pathDirection: int, rate: int, burst: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param rate: 背景流量带宽占比例
        :param burst: 背景流量报文大小
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "bg"
        children_node_Map = {"s": 2, "lu": rate, "bs": burst}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置队列深度
    @checkParameter()
    def set_Queue_Limit_Drop_Tail(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, queueDepthValue: int,
                                  queueDepthType: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向  [1.仅损伤下行，2.仅损伤上行]
        :param queueDepthValue: 队列深度值大小
        :param queueDepthType: 队列深度值类型  [1.报文格式， 2.内存大小， 3.时间ms]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "ql"
        children_node_Map = {"s": 1, "dt": {"qd": queueDepthValue, "qdt": queueDepthType}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置RED队列
    @checkParameter()
    def set_Queue_Limit_RED(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, weight: float, minQueueThreshold: int,  maxQueueThreshold: int, maxDropProbability: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向  [1.仅损伤下行，2.仅损伤上行]
        :param weight: 权重
        :param minQueueThreshold: 最小队列长度阈值
        :param maxQueueThreshold: 最大队列长度阈值
        :param maxDropProbability: 最大丢弃概率
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "ql"
        children_node_Map = {"s": 2, "red": {"w": weight, "math": maxQueueThreshold, "mith": minQueueThreshold, "madp": maxDropProbability}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)
    # 关闭报文配置
    @checkParameter()
    def close_message_Modify(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        try:
            return self.close_damage_config(holowan_ip, holowan_port, engineID, pathID, pathDirection, "/md/cs")
        except RuntimeError as e:
            return e


    # 设置报文配置 Normal模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Normal(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, matchType: int,
                                  matchOffset: int, matchSize: int, matchValue: str,
                                  modifyType: int, modifyOffset: int, modifySize: int, modifyValue: str,
                                  crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.Valn, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param modifyType: 修改报文修改起始位置类型,??????????????????????????????????????????????????????
        :param modifyOffset: 修改报文修改位置距离起始位置的偏移量
        :param modifySize: 修改报文新的值大小,[1,2,4,6,8],单位Byte
        :param(str) modifyValue:修改报文新的值
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "md"
        children_node_Map = {"cs": "1", "pt": matchType, "po": matchOffset, "ps": matchSize,
                             "pv": "0x" + matchValue,"fs":matchSwitch,
                             "mt": modifyType, "mo": modifyOffset,
                             "ms": modifySize, "mv": "0x" + modifyValue, "crc": crc, "checksum": checksum}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置报文配置 Cycle模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Cycle(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                 pathDirection: int, matchType: int, matchOffset: int, matchSize: int,
                                 matchValue: str, modifyType: int, modifyOffset: int, modifySize: int,
                                 modifyValue: str, modifyCyclePeriod: int, modifyCycleBurst: int,
                                 crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.Valn, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset:修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param modifyType: 修改报文修改起始位置类型,??????????????????????????????????????????????????????
        :param modifyOffset: 修改报文修改位置距离起始位置的偏移量
        :param modifySize: 修改报文新的值大小,[1,2,4,6,8],单位Byte
        :param(str) modifyValue: 修改报文新的值
        :param modifyCyclePeriod: ??????????????????????????????????????????????????????
        :param modifyCycleBurst: ??????????????????????????????????????????????????????
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """

        parent_node_ID = "md"
        children_node_Map = {"cs": "2", "pt": matchType, "po": matchOffset, "ps": matchSize,
                             "pv": "0x" + matchValue,"fs":matchSwitch,
                             "mt": modifyType,
                             "mo": modifyOffset, "ms": modifySize, "mv": "0x" + modifyValue,
                             "mcp": modifyCyclePeriod,
                             "mcb": modifyCycleBurst,
                             "crc": crc,
                             "checksum": checksum}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置报文配置 Random模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Random(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                 pathDirection: int, matchType: int, matchOffset: int, matchSize: int,
                                 matchValue: str, modifyType: int, modifyOffset: int, modifySize: int,
                                 modifyValue: str, modifyRandomRate: float, crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.Valn, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset:修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param modifyType: 修改报文修改起始位置类型,??????????????????????????????????????????????????????
        :param modifyOffset: 修改报文修改位置距离起始位置的偏移量
        :param modifySize: 修改报文新的值大小,[1,2,4,6,8],单位Byte
        :param(str) modifyValue: 修改报文新的值
        :param modifyRandomRate: ??????????????????????????????????????????????????????
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if len(str(modifyRandomRate).split('.')[1]) > int(self.modifyRandomRateDecimal):
            return r'{"errCode":"-64","errMsg":"ERROR","errReason":"Error parameter values modifyRandomRate"}'

        parent_node_ID = "md"
        children_node_Map = {"cs": "3", "pt": matchType, "po": matchOffset, "ps": matchSize,
                             "pv": "0x" + matchValue,"fs":matchSwitch,
                             "mt": modifyType,
                             "mo": modifyOffset, "ms": modifySize, "mv": "0x" + modifyValue,
                             "mrr": modifyRandomRate, "crc": crc, "checksum": checksum}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置修改报文 Range模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Range(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, matchType: int,
                                  matchOffset: int, matchSize: int, matchValue: str,
                                  ranges: list, crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.vlan, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param ranges: 修改报文的范围, 包括value, offset以及size
            [value为修改的内容, 使用不带0x的十六进制数字符串代表一个字节, 使用逗号(,)分隔
            offset为范围修改的偏移量
            size为修改的长度, 需要与value的长度对应]
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "md"
        children_node_Map = {"cs": "4", "pt": matchType, "po": matchOffset, "ps": matchSize,"fs":matchSwitch,
                             "pv": "0x" + matchValue,
                             "crc": crc, "checksum": checksum, "ra": {"nb": ranges.__len__()}}

        for v in ranges:
            if v.__len__() != 3 or v["value"] == "" or v["offset"] == "" or v["size"] == "":
                return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"ranges error!"}'
            str_list = v["value"].split(',')
            if len(str_list) != v["size"]:
                return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"size error!"}'
            for s in str_list:
                if int(s, 16) > 0xff or int(s, 16) < 0:
                    return r'{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"value error!"}'

        for k, v in enumerate(ranges):
            children_node_Map["ra"]["range{}".format(k)] = v

        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)


    # 设置修改报文 Insert模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Insert(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, matchType: int,
                                  matchOffset: int, matchSize: int, matchValue: str,
                                  value: str, size: int, offset: int,
                                  crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.vlan, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param value: 插入报文的内容, 使用不带0x的十六进制数字符串代表一个字节, 使用逗号(,)分隔
        :param size: 修改的长度, 需要与value的长度对应
        :param offset: 插入位置的偏移量
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        str_list = value.split(',')
        if len(str_list) != size:
            return r'{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"size error!"}'
        for s in str_list:
            if int(s, 16) > 0xff or int(s, 16) < 0:
                return r'{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"value error!"}'

        parent_node_ID = "md"
        children_node_Map = {"cs": "5", "pt": matchType, "po": matchOffset, "ps": matchSize,
                             "pv": "0x" + matchValue, "ms": size, "ins": value, "mo": offset,
                             "crc": crc, "checksum": checksum,"fs":matchSwitch}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)

    @checkParameter()
    @checkModify()
    def set_message_Modify_Delete(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                  pathDirection: int, matchType: int,
                                  matchOffset: int, matchSize: int, matchValue: str,
                                  offset: int,size:int,crc:int=1,checksum:int=1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.vlan, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param size: 修改的长度, 需要与value的长度对应
        :param offset: 插入位置的偏移量
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param matchSwitch: match开关
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """

        parent_node_ID = "md"
        children_node_Map = {"cs": "8", "pt": matchType, "po": matchOffset, "ps": matchSize,
                             "pv": "0x" + matchValue,'fs':matchSwitch,
                             "crc": crc, "checksum": checksum,
                             "de":{"offset":offset,"size":size}
                             }
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)

    # 设置修改报文 Exchage模式
    @checkParameter()
    @checkModify()
    def set_message_Modify_Exchage(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                   pathDirection: int, matchType: int,
                                   matchOffset: int, matchSize: int, matchValue: str,
                                   offsetPrev: int, sizePrev: int, offsetNext: int, sizeNext: int,
                                   crc: int = 1, checksum: int = 1,matchSwitch:int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.vlan, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param offsetPrev: 报文交换前一段内容的位移
        :param sizePrev: 报文交换前一段内容的长度
        :param offsetNext: 报文交换后一段内容的位移
        :param sizeNext: 报文交换后一段内容的长度
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "md"
        children_node_Map = {"cs": "6", "pt": matchType, "po": matchOffset, "ps": matchSize,"fs":matchSwitch,
                             "pv": "0x" + matchValue, "ex": {"prev": "", "next": ""},
                             "crc": crc, "checksum": checksum}
        properties_Map = {"./ex/prev": {"offset": offsetPrev, "size": sizePrev},
                          "./ex/next": {"offset": offsetNext, "size": sizeNext}}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)

    @checkParameter()
    @checkModify()
    def set_message_Modify_Count(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                   pathDirection: int, matchType: int,
                                   matchOffset: int, matchSize: int, matchValue: str,
                                   ranges:list,modifyOffset:int,modifyCount:int,
                                   crc: int = 1, checksum: int = 1, matchSwitch: int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param matchType: 修改报文二次匹配起始位置类型,[1.Ether, 4.vlan, 5.Vlan Stack, 6.Mpls, 7.PPPoE, 8.IPv4, 12.IPv6, 15.TCP, 17.UDP]
        :param matchOffset: 修改报文二次匹配距离起始位置偏移量
        :param matchSize: 修改报文二次匹配值大小,[1,2,4,6,8],单位Byte
        :param(str) matchValue: 修改报文二次匹配值
        :param offsetPrev: 报文交换前一段内容的位移
        :param sizePrev: 报文交换前一段内容的长度
        :param offsetNext: 报文交换后一段内容的位移
        :param sizeNext: 报文交换后一段内容的长度
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :param checksum: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        parent_node_ID = "md"
        children_node_Map = {"cs": "7", "pt": matchType, "po": matchOffset, "ps": matchSize, "fs": matchSwitch,
                             "pv": "0x" + matchValue, "crc": crc, "checksum": checksum,
                             "ra":{"nb":ranges.__len__()}
                             }

        for v in ranges:
            if v.__len__() != 3 or v["value"] == "" or v["offset"] == "" or v["size"] == "":
                return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"ranges error!"}'
            str_list = v["value"].split(',')
            if len(str_list) != v["size"]:
                return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"size error!"}'
            for s in str_list:
                if int(s, 16) > 0xff or int(s, 16) < 0:
                    return r'{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"value error!"}'

        for k, v in enumerate(ranges):
            children_node_Map["ra"]["range{}".format(k)] = v

        children_node_Map["ra"]["start_offset"] = modifyOffset
        children_node_Map["ra"]["modify_count"] = modifyCount
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)

    # 设置path重定向
    @checkParameter()
    def set_path_redirect(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                   pathDirection: int, redirect: int):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路重定向方向 [1.下行，2.上行]
        :param redirect: 重定向到达的pathID, 当设置为0的时候关闭重定向
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if pathDirection == 1:
            path_tag = "./pltr"
        elif pathDirection == 2:
            path_tag = "./prtl"
        else:
            return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"Error pathDirection"}}'.format(
                self.PCpathDirectionError)

        return self.set_one_tag_text(holowan_ip, holowan_port, engineID, pathID, tagPath=path_tag + "/rd",
                                     tagText=redirect)


    # 关闭MTU限制配置
    @checkParameter()
    def close_MTU_Limit(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        try:
            return self.close_damage_config(holowan_ip, holowan_port, engineID, pathID, pathDirection, "/m/s")
        except RuntimeError as e:
            return e

        # 设置path重定向

    @checkParameter()
    def set_path_redirect(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                          pathDirection: int, redirect: int):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路重定向方向 [1.下行，2.上行]
        :param redirect: 重定向到达的pathID, 当设置为0的时候关闭重定向
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if pathDirection == 1:
            path_tag = "./pltr"
        elif pathDirection == 2:
            path_tag = "./prtl"
        else:
            return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"Error pathDirection"}}'.format(
                self.PCpathDirectionError)

        return self.set_one_tag_text(holowan_ip, holowan_port, engineID, pathID, tagPath=path_tag + "/rd",
                                     tagText=redirect)

    # 设置MTU限制配置
    @checkParameter()
    def set_MTU_Limit(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                      limitValue: int,drop_oversize:int = 1,df:int = 0):
        '''

        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param limitValue:
        :param drop_oversize
        :param df
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if drop_oversize + df > 1:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(drop_oversize) or values(df)"}'
        parent_node_ID = "m"
        children_node_Map = {"s": "2", "n": limitValue,"ov":drop_oversize,"df":df}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置以太网间隙占用配置
    @checkParameter()
    def set_Frame_Overhead(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                           type: int, rate: int):
        '''

        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param type: 帧间隙占用类型,[1.默认以太网24, 2.最小4, 3.自定义值]
        :param rate: 帧间隙大小值
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "fo"
        if type == 1:
            children_node_Map = {"t": 1, "r": 24}
        elif type == 2:
            children_node_Map = {"t": 2, "r": 4}
        elif type == 3:
            children_node_Map = {"t": 3, "r": rate}
            if rate < 0 or rate > 64:
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"rate ranges from 1 to 64"}'
        else:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(type)"}'
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  常量时延(Constant)
    @checkParameter()
    def set_Delay_Constant(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                           delay: float):
        '''

        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param(float) delay: 时延值 [>0.1] 单位ms
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if len(str(delay).split(".")[1]) > int(self.constantDelayDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(delay)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "1", "co": {"de": delay}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  平均分布时延(Uniform)  无高级设置
    @checkParameter()
    def set_Delay_Uniform(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                          minimum: float, maximum: float,
                          enableReordering: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param(float) minimum: 平均分布最小值
        :param(float) maximum: 平均分布最大值
        :param enableReordering: 是否允许时延乱序 [0.关闭, 1.开启]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if len(str(minimum).split(".")[1]) > int(self.uniformMinimumDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(minimum)"}'

        if len(str(maximum).split(".")[1]) > int(self.uniformMaximumDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(maximum)"}'

        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "2", "un": {"dmi": minimum, "dma": maximum, "reo": enableReordering, "shake": ""}}
        properties_Map = {"un/shake": {"type_id": "0"}}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  平均分布时延(Uniform) 高级设置
    @checkParameter()
    def set_Delay_Uniform_Advanced(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                   pathDirection: int,
                                   minimum: float, maximum: float,
                                   enableReordering: int,
                                   jitter_type: int, Advanced_MAX: int, Advanced_MIN: int,
                                   Phase: int = 0, Period: int = 60,
                                   Rise: int = 30, Fall: int = 20, Section: int = 50
                                   ):
        """
        :param Section:
        :param Fall:
        :param Rise:
        :param Period:
        :param Phase:
        :param Advanced_MIN:
        :param Advanced_MAX:
        :param jitter_type: 高级功能抖动
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param(float) minimum: 平均分布最小值
        :param(float) maximum: 平均分布最大值
        :param enableReordering: 是否允许时延乱序 [0.关闭, 1.开启]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if len(str(minimum).split(".")[1]) > int(self.uniformMinimumDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(minimum)"}'

        if len(str(maximum).split(".")[1]) > int(self.uniformMaximumDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(maximum)"}'

        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "2", "un": {"dmi": minimum, "dma": maximum, "reo": enableReordering,
                                              "shake": {"max": Advanced_MAX, "min": Advanced_MIN, "cycle": Period}}}
        properties_Map = {"un/shake": {"type_id": jitter_type}}

        if jitter_type == 1:
            children_node_Map["un"]["shake"]["phase"] = float(Phase / 100)
        elif jitter_type == 2:
            children_node_Map["un"]["shake"]["proportion_up"] = float(Rise / 100)
            children_node_Map["un"]["shake"]["proportion_down"] = float(Fall / 100)
        elif jitter_type == 5 or jitter_type == 6:
            children_node_Map["un"]["shake"]["proportion"] = float(Section / 100)

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  正太分布时延(Uniform)  无高级设置
    @checkParameter()
    def set_Delay_Normal(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                         min: float, mean: float, stdDeviation: float, enableReordering: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param(float) min: 最小截断值
        :param(float) mean: 期望值
        :param stdDeviation: 标准差
        :param enableReordering: 是否允许时延乱序 [0.关闭, 1.开启]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''

        if (len(str(min).split(".")[1])) > int(self.normalMinDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(min)"}'

        if (len(str(mean).split(".")[1])) > int(self.normalMeanDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(mean)"}'

        if (len(str(stdDeviation).split(".")[1])) > int(self.normalStdDeviationDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(stdDeviation)"}'

        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "3",
                             "no": {"de": min, "me": mean, "sd": stdDeviation, "reo": enableReordering, "b": ""}}
        properties_Map = {"./no/b": {"e": "0"}}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  正太分布时延(Uniform)  高级设置
    @checkParameter()
    def set_Delay_Normal_AdvancedSetup(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                         min: float, mean: float, stdDeviation: float, enableReordering: int, advancedPeriod: float,
                                       advancedDuration: float, advancedMin: float, advancedMax: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param(float) min: 最小截断值
        :param(float) mean: 期望值
        :param stdDeviation: 标准差
        :param enableReordering: 是否允许时延乱序 [0.关闭, 1.开启]
        :param advancedPeriod: ???????????????????????????????????????????????
        :param advancedDuration: ???????????????????????????????????????????????
        :param advancedMin: ???????????????????????????????????????????????
        :param advancedMax: ???????????????????????????????????????????????
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''

        if (len(str(min).split(".")[1])) > int(self.normalMinDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(min)"}'

        if (len(str(mean).split(".")[1])) > int(self.normalMeanDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(mean)"}'

        if (len(str(stdDeviation).split(".")[1])) > int(self.normalStdDeviationDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(stdDeviation)"}'

        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        if (len(str(advancedPeriod).split(".")[1])) > int(self.normalAdvancedPeriodDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(advancedPeriod)"}'

        if (len(str(advancedDuration).split(".")[1])) > int(self.normalAdvancedDurationDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(advancedDuration)"}'

        if (len(str(advancedMax).split(".")[1])) > int(self.normalAdvancedMaxDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(advancedMax)"}'

        if (len(str(advancedMin).split(".")[1])) > int(self.normalAdvancedMinDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(advancedMin)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "3", "no": {"de": min, "me": mean, "sd": stdDeviation, "reo": enableReordering,
                                              "b": {"p": advancedPeriod, "d": advancedDuration, "mi": advancedMin,
                                                    "ma": advancedMax}}}
        properties_Map = {"./no/b": {"e": "1"}}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置时延配置  可自定义的正太分布时延(Custom)
    @checkParameter()
    def set_Delay_Custom(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                         meanDelay: float, minDelay: float, maxDelay: float, positiveDelta: float,
                         negativeDelta: float, spread: float, enableReordering: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param meanDelay: 平均时延
        :param minDelay: 最小时延
        :param maxDelay: 最大时延
        :param positiveDelta: 最大正向变化值
        :param negativeDelta: 最大负向变化值
        :param spread: Spread值
        :param enableReordering: 是否允许时延乱序 [0.关闭, 1.开启]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''

        if (len(str(meanDelay).split(".")[1])) > int(self.customMeanDelayDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(meanDelay)"}'
        if (len(str(minDelay).split(".")[1])) > int(self.customMinDelayDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(minDelay)"}'
        if (len(str(maxDelay).split(".")[1])) > int(self.customMaxDelayDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(maxDelay)"}'
        if (len(str(positiveDelta).split(".")[1])) > int(self.customPositiveDeltaDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(positiveDelta)"}'
        if (len(str(negativeDelta).split(".")[1])) > int(self.customNegativeDeltaDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(negativeDelta)"}'
        if (len(str(spread).split(".")[1])) > int(self.customSpreadDeltaDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(spread)"}'
        if meanDelay >= maxDelay:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"meanDelay must be less than maxDelay"}'

        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "4", "cu": {"made": maxDelay, "mede": meanDelay, "mide": minDelay, "pd": positiveDelta,
                                              "nd": negativeDelta, "spd": spread, "reo": enableReordering}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    # Delay - Jitter
    @checkParameter()
    def set_Delay_Jitter(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                         delay: float, jitter: float, jitterType: str, enableReordering: int):

        if (len(str(delay).split(".")[1])) > int(self.jitterMeanDelayDecimal or delay < 0 or delay > 10000):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(delay)"}'
        if (len(str(jitter).split(".")[1])) > int(self.jitterMeanJitterDecimal or jitter < 0 or jitter > 10000):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(jitter)"}'
        if (delay - jitter < 0):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Parameters must satisfy: delay - jitter ≥ 0"}'
        if jitterType == 'constant':
            typeNum = 1
        elif jitterType == 'uniform':
            typeNum = 2
        elif jitterType == 'normal':
            typeNum = 3
        else:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(jitterType)"}'
        if enableReordering not in [0, 1]:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(enableReordering)"}'

        parent_node_ID = "d"
        children_node_Map = {"s": "5", "ji": {"md": delay, "j": jitter, "s": typeNum, "reo": enableReordering}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    # Delay - Gamma
    @checkParameter()
    def set_Delay_Gamma(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                        shape: float, scale: float, enableReordering: int):
        parent_node_ID = "d"
        children_node_Map = {"s": "6", "ga": {"shape": shape, "scale": scale, "reo": enableReordering}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    # Delay - Accumulate&Burst
    @checkParameter()
    def set_Delay_Accumulate_And_Burst(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int,
                                       pathDirection: int,
                                       delay: float, extra_delay: float, count: int):
        parent_node_ID = "d"
        children_node_Map = {"s": "7", "acc": {"delay": delay, "extra_delay": extra_delay, "count": count}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    # 设置丢包配置  Random普通概率模式
    @checkParameter()
    def set_Loss_Random(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int, rate: float):
        '''

        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param rate: 普通概率丢包概率值
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if (len(str(rate).split(".")[1])) > int(self.randomRateDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(rate)"}'

        parent_node_ID = "l"
        children_node_Map = {"s": "1", "ra": {"r": rate}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置丢包配置  Cycle周期丢包模式
    @checkParameter()
    def set_Loss_Cycle(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                       period: int, burst: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param period: 周期丢包周期值，单位为报文个数
        :param burst: 周期丢包数量，单位为报文个数
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "l"
        children_node_Map = {"s": "2", "cy": {"a": period, "l": burst}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置丢包配置  Burst突发丢包模式
    @checkParameter()
    def set_Loss_Burst(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                       probability: float, minimum: int, maximum: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param probability: 突发丢包概率值
        :param minimum: 突发丢包最小丢包数量
        :param maximum: 突发丢包最大丢包数量
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''

        if (len(str(probability).split(".")[1])) > int(self.burstProbabilityDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(probability)"}'

        parent_node_ID = "l"
        children_node_Map = {"s": "3", "bu": {"l": probability, "mi": minimum, "ma": maximum}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置丢包配置  Dual双通道丢包模式
    @checkParameter()
    def set_Loss_Dual(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                      goodStateLoss: float, goodToBadProbability: float, badStateLoss: float,
                      badToGoodProbability: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param goodStateLoss: 好通道丢包概率
        :param goodToBadProbability: 好通道转换到坏通道的转换概率
        :param badStateLoss: 坏通道丢包概率
        :param badToGoodProbability: 坏通道转换到好通道的转换概率
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if (len(str(goodStateLoss).split(".")[1])) > int(self.dualGoodStateLossDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(goodStateLoss)"}'
        if (len(str(goodToBadProbability).split(".")[1])) > int(self.dualGoodToBadProbabilityDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(goodToBadProbability)"}'
        if (len(str(badStateLoss).split(".")[1])) > int(self.dualBadStateLossDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(badStateLoss)"}'
        if (len(str(badToGoodProbability).split(".")[1])) > int(self.dualBadToGoodProbabilityDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(badToGoodProbability)"}'

        parent_node_ID = "l"
        children_node_Map = {"s": "4", "du": {"g": goodStateLoss, "b": badStateLoss, "gtb": goodToBadProbability,
                                              "btg": badToGoodProbability}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置丢包配置  Markov马尔可夫模型
    def set_Loss_Markov(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                        p13: float, p31: float, p32: float, p23: float, p14: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param p13: Received Successfully 状态切换到 Lost Within a BURST 状态的概率
        :param p31: Lost Within a BURST 状态切换到 Received Successfully 状态的概率
        :param p32: Lost Within a BURST 状态切换到 Received Within a BURST 状态的概率
        :param p23: Received Within a BURST 状态切换到 Lost Within a BURST 状态的概率
        :param p14: Received Successfully 状态切换到 Isolated Lost Within a Gap 状态的概率
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if ("." in str(p13)):
            if (len(str(p13).split(".")[1])) > int(self.markovP13Decimal):
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p13)"}'
        if ("." in str(p31)):
            if (len(str(p31).split(".")[1])) > int(self.markovP31Decimal):
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p31)"}'
        if ("." in str(p32)):
            if (len(str(p32).split(".")[1])) > int(self.markovP32Decimal):
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p32)"}'
        if ("." in str(p23)):
            if (len(str(p23).split(".")[1])) > int(self.markovP23Decimal):
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p23)"}'
        if ("." in str(p14)):
            if (len(str(p14).split(".")[1])) > int(self.markovP14Decimal):
                return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p14)"}'

        if p13 > int(self.markovP13Max) or p13 < float(self.markovP13Min):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p13)"}'
        if p31 > int(self.markovP31Max) or p31 < float(self.markovP31Min):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p31)"}'
        if p32 > int(self.markovP32Max) or p32 < float(self.markovP32Min):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p32)"}'
        if p23 > int(self.markovP23Max) or p23 < float(self.markovP23Min):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p23)"}'
        if p14 > int(self.markovP14Max) or p14 < float(self.markovP14Min):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(p14)"}'
        parent_node_ID = "l"
        children_node_Map = {"s": "6", "mkv": {"p13": p13*0.01, "p31": p31*0.01, "p32": p32*0.01, "p23": p23*0.01, "p14": p14*0.01}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    @checkParameter()
    def set_Loss_Jitter(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                        jitter_type: int, MAX: float, MIN: float,
                        Phase: int = 0, Period: int = 60,
                        Rise: int = 30, Fall: int = 20, Section: int = 50):
        parent_node_ID = "l"
        children_node_Map = {"s": 5, "shake": {"cycle": Period, "max": MAX, "min": MIN}}
        properties_Map = {"shake": {"type_id": jitter_type}}

        if jitter_type == 1:
            children_node_Map["shake"]["phase"] = float(Phase / 100)
        elif jitter_type == 2:
            children_node_Map["shake"]["proportion_up"] = float(Rise / 100)
            children_node_Map["shake"]["proportion_down"] = float(Fall / 100)
        elif jitter_type == 5 or jitter_type == 6:
            children_node_Map["shake"]["proportion"] = float(Section / 100)

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID, children_node_Map, properties_Map)


    # 设置BER Normal配置
    @checkParameter()
    def set_BER_Normal(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                bitErrorValue: int, bitErrorIndex: int, crc: int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param bitErrorValue: BER值
        :param bitErrorIndex: BER指数值
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if not self.right_path(holowan_ip,holowan_port,engineID,pathID):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(pathID)"}'

        if bitErrorValue < 0 or bitErrorValue > 9:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(bitErrorValue)"}'

        if bitErrorIndex > -3 or bitErrorIndex < -14:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(bitErrorIndex)"}'

        bitErrorIndex = abs(bitErrorIndex)
        parent_node_ID = "cor"
        children_node_Map = {"s": "1", "ber": bitErrorValue, "beri": bitErrorIndex, "crc": crc}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置BER Range配置
    @checkParameter()
    def set_BER_Range(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                bitErrorValue: int, bitErrorIndex: int, bitErrorRangeList: list, crc: int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param bitErrorValue: BER值
        :param bitErrorIndex: BER指数值
        :param bitErrorRangeList: BER范围列表
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if not self.right_path(holowan_ip, holowan_port, engineID, pathID):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(pathID)"}'

        if bitErrorValue < 0 or bitErrorValue > 9:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(bitErrorValue)"}'

        if bitErrorIndex > -3 or bitErrorIndex < -14:
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(bitErrorIndex)"}'
        bitErrorIndex = abs(bitErrorIndex)
        parent_node_ID = "cor"
        children_node_Map = {"s": "2", "ber": bitErrorValue, "beri": bitErrorIndex, "range": {}, "crc": crc}
        properties_Map = {}
        # 例：bitErrorRangeList为[[0, 10], [20, 30]]
        # 对应的properties_Map为{"range/r1": {"start": 0, "end": 10}, "range/r2": {"start": 20, "end": 30}}
        max_pos = 0
        if len(bitErrorRangeList) < 1 or len(bitErrorRangeList) > 10:
            return '{"errCode":"-500","errMsg":"ERROR","errReason":"bitErrorRangeList的长度必须大于0，且小于等于10"}'
        for i in range(len(bitErrorRangeList)):
            r = bitErrorRangeList[i]
            if len(r) != 2:
                return '{"errCode":"-500","errMsg":"ERROR","errReason":"bitErrorRangeList中的每个range必须含有2个整数，分别代表start和end的字节位置"}'
            start = int(r[0])
            end = int(r[1])
            if start < 0 or end < 0 or start > 1500 or end > 1500:
                return '{"errCode":"-500","errMsg":"ERROR","errReason":"start或end超出范围"}'
            if start >= end:
                return '{"errCode":"-500","errMsg":"ERROR","errReason":"start必须小于end"}'
            if start < max_pos:
                return '{"errCode":"-500","errMsg":"ERROR","errReason":"bitErrorRangeList中的range必须是有序的"}'
            max_pos = end
            children_node_Map["range"]["r" + str(i + 1)] = ""
            properties_Map["range/r" + str(i + 1)] = {"start": str(start), "end": str(end)}

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)

    # 设置BER packet配置
    @checkParameter()
    def set_BER_Pactet(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                       Probability: float, bitErrorNum: int, crc: int = 1):
        """
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param Probability: 错包的概率, 范围[0.01~100]
        :param bitErrorNum: 每个报文中有多少个位错误, 范围:[1~20]
        :param crc: 是否重新计算crc[0.不重新计算, 1.重新计算]
        :return: Json [正确：errCode等于0， 错误：errCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        """
        if not self.right_path(holowan_ip,holowan_port,engineID,pathID):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(pathID)"}'

        parent_node_ID = "cor"
        children_node_Map = {"s": 3, "prob": Probability, "nb": bitErrorNum, "crc": crc}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)

    # 设置报文乱序配置  默认类型
    @checkParameter()
    def set_Reordering_Normal(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                              probability: float, delayMin: float, delayMax: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param probability: 报文乱序概率
        :param delayMin: 报文乱序延时最小值
        :param delayMax: 报文乱序延时最大值
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if (len(str(probability).split(".")[1])) > int(self.normalProbabilityDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(probability)"}'
        if (len(str(delayMin).split(".")[1])) > int(self.normalDelayMinDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(delayMin)"}'
        if (len(str(delayMax).split(".")[1])) > int(self.normalDelayMaxDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(delayMax)"}'

        parent_node_ID = "reo"
        children_node_Map = {"s": "1", "no": {"p": probability, "dmi": delayMin, "dma": delayMax}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)


    # 设置报文乱序配置 曲线控制乱序概率变化
    @checkParameter()
    def set_Reordering_Jitter(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                              delayMin: float, delayMax: float, MAX: float, MIN: float, jitter_type: int,
                              Phase: int = 0, Period: int = 60,
                              Rise: int = 30, Fall: int = 20, Section: int = 50):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param delayMin: 报文乱序延时最小值
        :param delayMax: 报文乱序延时最大值
        :param MAX: 最大乱序概率
        :param MIN: 最小乱序概率
        :param jitter_type: 抖动图像类型
        :param Phase: 曲线初始位置
        :param Period: 抖动的周期(最大3600秒)
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "reo"
        children_node_Map = {"s": 2, "ji": {"dmi": delayMin, "dma": delayMax,
                                            "shake": {"max": MAX, "min": MIN, "cycle": Period, "phase": Phase}}}
        properties_Map = {"ji/shake": {"type_id": jitter_type}}

        # 需要判断类型, 然后children_mode_Map再作出相应的调整
        if jitter_type == 1:
            children_node_Map["ji"]["shake"]["phase"] = float(Phase / 100)
        elif jitter_type == 2:
            children_node_Map["ji"]["shake"]["proportion_up"] = float(Rise / 100)
            children_node_Map["ji"]["shake"]["proportion_down"] = float(Fall / 100)
        # elif jitter_type == 3 or jitter_type == 4:
        #  3 和 4啥都不用做
        elif jitter_type == 5 or jitter_type == 6:
            children_node_Map["ji"]["shake"]["proportion"] = float(Section / 100)

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)


    # 设置报文乱序配置 周期乱序
    @checkParameter()
    def set_Reordering_Cycle(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                              Cycle_type: int, period: int, count: int, delayMin: float, delayMax: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param Cycle_type: 周期乱序的类型[1.Random,2.Burst]
        :param period: 周期
        :param count: 抽取的报文数
        :param delayMin: 时延最小值
        :param delayMax: 时延最大值
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "reo"
        children_node_Map = {"s": 3, "cy": {"type": Cycle_type, "period": period, "count": count, "dmi": delayMin, "dma": delayMax}}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)


    # 设置重复报文配置  默认类型
    @checkParameter()
    def set_Duplication_Normal(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                               probability: float):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param probability: 重复报文概率
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        if (len(str(probability).split(".")[1])) > int(self.duplicationNormalProbabilityDecimal):
            return r'{"errCode":"-500","errMsg":"ERROR","errReason":"Error parameter values(probability)"}'
        parent_node_ID = "du"
        children_node_Map = {"s": "1", "p": probability}
        properties_Map = {}
        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                                  parent_node_ID,
                                                  children_node_Map, properties_Map)


    # 设置重复报文配置  曲线控制报文重复概率变化
    @checkParameter()
    def set_Duplication_Jitter(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int,
                               MAX: float, MIN: float, jitter_type: int,
                               Phase: int = 0, Period: int = 60,
                               Rise: int = 30, Fall: int = 20, Section: int = 50):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1.仅损伤下行，2.仅损伤上行]
        :param MAX: 最大乱序概率
        :param MIN: 最小乱序概率
        :param jitter_type: 抖动图像类型
        :param Phase: 曲线初始位置
        :param Period: 抖动的周期(最大3600秒)
        :param Rise: 曲线类型2的参数，上升曲线占整条曲线的比例
        :param Fall: 曲线类型2的参数，下降曲线占整条曲线的比例
        :param Section: 曲线类型5、6的参数，前半段曲线占整条曲线的比例
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        parent_node_ID = "du"
        children_node_Map = {"s": 2, "shake": {"max": MAX, "min": MIN, "cycle": Period, "phase": Phase}}
        properties_Map = {"shake": {"type_id": jitter_type}}

        # 需要判断类型, 然后children_mode_Map再作出相应的调整
        if jitter_type == 1:
            children_node_Map["shake"]["phase"] = float(Phase / 100)
        elif jitter_type == 2:
            children_node_Map["shake"]["proportion_up"] = float(Rise / 100)
            children_node_Map["shake"]["proportion_down"] = float(Fall / 100)
        # elif jitter_type == 3 or jitter_type == 4:
        #  3 和 4啥都不用做
        elif jitter_type == 5 or jitter_type == 6:
            children_node_Map["shake"]["proportion"] = float(Section / 100)

        return self.set_tag_children(holowan_ip, holowan_port, engineID, pathID, pathDirection,
                                     parent_node_ID,
                                     children_node_Map, properties_Map)


    # 获取path链路xml信息
    @checkParameter()
    def get_path_config_information(self, holowan_ip:str, holowan_port:str, engineID:int, pathID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :return: 修改虚拟链路损伤参数XML
        '''
        try:
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/path_config_info_{2}_{3}.xml".format(holowan_ip, holowan_port, engineID, pathID)
            return self.session.get(requestURL, verify=False).text
        except Exception as e:
            if self.has_engine(holowan_ip, holowan_port, engineID) is False:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE has no found"}}'.format(self.PCengineIDError)
            if self.right_path(holowan_ip, holowan_port, engineID, pathID) is False:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathIDError)
            if self.has_path(holowan_ip, holowan_port, engineID, pathID) is False:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathNotFound)

    # 关闭相关损伤配置
    def close_damage_config(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection: int, tapPath: str):
        try:
            tapPath_list = []
            if pathDirection == 1:
                tapPath_list.append("pltr" + tapPath)
            elif pathDirection == 2:
                tapPath_list.append("prtl" + tapPath)
            elif pathDirection == 3:
                tapPath_list.append("pltr" + tapPath)
                tapPath_list.append("prtl" + tapPath)
            else:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"Error pathDirection"}}'.format(self.PCpathDirectionError)
            tagText = 1
            if tapPath == "/md/cs":
                tagText = 0
            response = None
            for tagPath in tapPath_list:
                response = self.set_one_tag_text(holowan_ip, holowan_port, engineID, pathID, tagPath, tagText)
                if response is RuntimeError:
                    break
                elif json.loads(response)['errCode'] != "0":
                    break
            return response
        except RuntimeError as e:
            return e


    ''' ===============================================Classifier===================================================='''
    # 添加IPV4分类器  无隧道
    @checkParameter()
    def add_IPV4_to_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "action": action}
        return self.classifier(self.IPV4, self.ADD, parameters)

    # 删除IPV4分类器  无隧道
    @checkParameter()
    def remove_IPV4_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID:int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "action": action}
        return self.classifier(self.IPV4, self.REMOVE, parameters)

    # 删除所有的IPV4分类器
    @checkParameter()
    def remove_all_IPV4_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.IPV4, self.REMOVEALL, parameters)

    # 创建IPV4节点
    def create_IPV4(self, holowan_ip: str, holowan_port: str, engineID:int, sourceIP: str, sourceMask: int,
                    destinationIP: str, destinationMask: int, TOS: str, action: int):
        ipv4_node = xt.create_node("ipv4", {}, {})
        children_node_Map = {"src": "", "smask": sourceMask, "dst": "", "dmask": destinationMask, "tos": "",
                             "path_id": action}
        properties = {}
        # ===============src================== #
        if sourceIP == "any":
            properties["src"] = {"any": 1}
        elif mt.isIP(sourceIP):
            children_node_Map["src"] = sourceIP
        else:
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"sourceIP必须输入any或ipv4地址"}')
        # ===============src mask================== #
        if sourceMask not in [32, 24, 16, 8]:
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"sourceIP掩码必须为32、24、16、8中的一个"}')
        # ===============dst================== #
        if destinationIP == "any":
            properties["dst"] = {"any": 1}
        elif mt.isIP(destinationIP):
            children_node_Map["dst"] = destinationIP
        else:
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"destinationIP必须输入any或ipv4地址"}')
        # ===============dst mask================== #
        if destinationMask not in [32, 24, 16, 8]:
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"destinationIP掩码必须为32、24、16、8中的一个"}')
        # ===============tos================== #
        if TOS == "any":
            properties["tos"] = {"any": 1}
        elif mt.isDoubleHexadecimal(TOS):
            children_node_Map["tos"] = TOS
        else:
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"TOS必须输入any或2位十六进制"}')
        # ===============action================== #
        paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        if action not in [-1, -2] and action not in paths_dic.keys():
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"action值输入错误"}')
        xt.add_children(ipv4_node, children_node_Map)
        xt.add_properties(ipv4_node, properties)
        return ipv4_node

    # 添加IPV6分类器  无隧道
    @checkParameter()
    def add_IPV6_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                               sourceIP: str, destinationIP: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "destinationIP": destinationIP, "action": action}
        return self.classifier(self.IPV6, self.ADD, parameters)

    # 删除IPV6分类器  无隧道
    @checkParameter()
    def remove_IPV6_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int,
                               sourceIP: str, destinationIP: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "destinationIP": destinationIP, "action": action}
        return self.classifier(self.IPV6, self.REMOVE, parameters)

    # 删除所有的IPV6分类器
    @checkParameter()
    def remove_all_IPV6_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.IPV6, self.REMOVEALL, parameters)

    # 创建IPV6节点
    def create_IPV6(self, holowan_ip:str, holowan_port:str, engineID:int, sourceIP: str, destinationIP: str, action: int):
        ipv6_node = xt.create_node("ipv6", {}, {})
        children_node_Map = {"src": "", "dst": "", "path_id": action}
        properties = {}
        # ===============src================== #
        if sourceIP == "any":
            properties["src"] = {"any": 1}
        elif mt.isIPV6(sourceIP):
            children_node_Map["src"] = sourceIP
        else:
            return RuntimeError(r'{"errCode":"-10","errMsg":"ERROR","errReason":"sourceIP必须输入any或ipv6地址"}')
        # ===============dst================== #
        if destinationIP == "any":
            properties["dst"] = {"any": 1}
        elif mt.isIPV6(destinationIP):
            children_node_Map["dst"] = destinationIP
        else:
            return RuntimeError(r'{"errCode":"-10","errMsg":"ERROR","errReason":"destinationIP必须输入any或ipv6地址"}')
        # ===============Action================== #
        paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        if action not in [-1, -2] and action not in paths_dic.keys():
            return RuntimeError(r'{"errCode":"-10","errMsg":"ERROR","errReason":"action值输入错误"}')
        xt.add_children(ipv6_node, children_node_Map)
        xt.add_properties(ipv6_node, properties)
        return ipv6_node

    # 添加MAC分类器  无隧道
    @checkParameter()
    def add_MAC_to_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int, sourceMAC: str,
                              destinationMAC: str, EtherType: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceMAC": sourceMAC, "destinationMAC": destinationMAC, "EtherType": EtherType, "action": action}
        return self.classifier(self.MAC, self.ADD, parameters)

    # 删除MAC分类器  无隧道
    @checkParameter()
    def remove_MAC_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int, sourceMAC: str,
                              destinationMAC: str, EtherType: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceMAC": sourceMAC, "destinationMAC": destinationMAC, "EtherType": EtherType, "action": action}
        return self.classifier(self.MAC, self.REMOVE, parameters)

    # 删除所有的MAC分类器
    @checkParameter()
    def remove_all_MAC_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.MAC, self.REMOVEALL, parameters)

    # 创建MAC节点
    def create_MAC(self, holowan_ip:str, holowan_port:str, engineID:int, sourceMAC: str, destinationMAC: str, EtherType: str, action: int):
        MAC_node = xt.create_node("mac", {}, {})
        children_node_Map = {"src": "", "dst": "", "type": "", "path_id": action}
        properties = {}
        # ===============src================== #
        if sourceMAC == "any":
            properties["src"] = {"any": 1}
        elif mt.isMac(sourceMAC):
            children_node_Map["src"] = sourceMAC
        else:
            return RuntimeError(r'{"errCode":"-6","errMsg":"ERROR","errReason":"sourceIP必须输入any或MAC地址"}')
        # ===============dst================== #
        if destinationMAC == "any":
            properties["dst"] = {"any": 1}
        elif mt.isMac(destinationMAC):
            children_node_Map["dst"] = destinationMAC
        else:
            return RuntimeError(r'{"errCode":"-6","errMsg":"ERROR","errReason":"destinationIP必须输入any或MAC地址"}')
        # ===============EtherType================== #
        if EtherType == "any":
            properties["type"] = {"any": 1}
        elif EtherType in self.etherType.split(", "):
            children_node_Map["type"] = EtherType
        else:
            return RuntimeError(r'{"errCode":"-6","errMsg":"ERROR","errReason":"MAC的EtherType输入值不正确"}')
        # ===============Action================== #
        paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        if action not in [-1, -2] and action not in paths_dic.keys():
            return RuntimeError(r'{"errCode":"-6","errMsg":"ERROR","errReason":"action值输入错误"}')
        xt.add_children(MAC_node, children_node_Map)
        xt.add_properties(MAC_node, properties)
        return MAC_node

    # 添加TCP分类器
    @checkParameter()
    def add_TCP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePort: str, destPort: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.TCP, self.ADD, parameters)

    # 删除TCP分类器
    @checkParameter()
    def remove_TCP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePort: str, destPort: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.TCP, self.REMOVE, parameters)

    # 删除所有TCP分类器
    @checkParameter()
    def remove_all_TCP_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.TCP, self.REMOVEALL, parameters)

    # 添加UDP分类器 无隧道
    @checkParameter()
    def add_UDP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePort: str, destPort: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.UDP, self.ADD, parameters)

    # 删除UDP分类器 无隧道
    @checkParameter()
    def remove_UDP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                   sourcePort: str, destPort: str,
                                   checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.UDP, self.REMOVE, parameters)

    # 删除所有UDP分类器
    @checkParameter()
    def remove_all_UDP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.UDP, self.REMOVEALL, parameters)

    # 添加TCP端口范围过滤
    @checkParameter()
    def add_TCP_range_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePortMin: str, sourcePortMax: str, destPortMin: str, destPortMax: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePortMin": sourcePortMin, "sourcePortMax": sourcePortMax,"destPortMin": destPortMin, "destPortMax": destPortMax,"checkVersion": checkVersion, "action": action}
        return self.classifier(self.TCPRANGE, self.ADD, parameters)

    # 删除TCP端口范围过滤
    @checkParameter()
    def remove_TCP_range_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePortMin: str, sourcePortMax: str, destPortMin: str, destPortMax: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePortMin": sourcePortMin, "sourcePortMax": sourcePortMax, "destPortMin": destPortMin,
                      "destPortMax": destPortMax, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.TCPRANGE, self.REMOVE, parameters)

    # 添加UDP端口范围过滤
    @checkParameter()
    def add_UDP_range_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                    sourcePortMin: str, sourcePortMax: str, destPortMin: str, destPortMax: str,
                                    checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID,
                      "portID": portID,
                      "sourcePortMin": sourcePortMin, "sourcePortMax": sourcePortMax, "destPortMin": destPortMin,
                      "destPortMax": destPortMax, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.UDPRANGE, self.ADD, parameters)

    # 删除TCP端口范围过滤
    @checkParameter()
    def remove_UDP_range_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                         sourcePortMin: str, sourcePortMax: str, destPortMin: str, destPortMax: str,
                                         checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID,
                      "portID": portID,
                      "sourcePortMin": sourcePortMin, "sourcePortMax": sourcePortMax, "destPortMin": destPortMin,
                      "destPortMax": destPortMax, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.UDPRANGE, self.REMOVE, parameters)


    # 添加SCTP分类器
    @checkParameter()
    def add_SCTP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourcePort: str, destPort: str,
                              checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.SCTP, self.ADD, parameters)

    # 删除SCTP分类器 无隧道
    @checkParameter()
    def remove_SCTP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                    sourcePort: str, destPort: str,
                                    checkVersion: int, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourcePort": sourcePort, "destPort": destPort, "checkVersion": checkVersion, "action": action}
        return self.classifier(self.SCTP, self.REMOVE, parameters)

    # 删除所有SCTP分类器
    @checkParameter()
    def remove_all_SCTP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.SCTP, self.REMOVEALL, parameters)

    # 创建TCP/UDP节点
    def create_TCP_or_UDP(self, holowan_ip: str, holowan_port: str, engineID: int, TCPUDP: int, sourcePort: str, destPort: str, checkVersion: int, action: int):
        '''
        :param TCPUDP: 1：TCP，2：UDP, 3: SCTP
        :param sourcePort: 源端口号
        :param destPort: 目标端口号
        :param checkVersion:
        :param action:
        :return:
        '''
        tcpudp_node = xt.create_node("tcp_udp", {}, {})
        children_node_map = {"type": TCPUDP, "src": "", "dst": "", "check": checkVersion, "path_id": action}
        properties = {}
        # ===============src================== #
        if sourcePort == "any":
            properties["src"] = {"any": 1}
        elif mt.isPort(sourcePort):
            properties["src"] = {"num": 1}
            children_node_map["src"] = {"port1": sourcePort}
        else:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"sourcePort必须为1-65535之间"}')
        # ===============dst================== #
        if destPort == "any":
            properties["dst"] = {"any": 1}
        elif mt.isPort(destPort):
            properties["dst"] = {"num": 1}
            children_node_map["dst"] = {"port1": destPort}
        else:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"dstPort必须为1-65535之间"}')
        # ===============checkVersion================== #
        if checkVersion not in [0, 4, 6]:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"checkVersion必须为0、4、6中的一个"}')
        # ===============Action================== #
        paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        if action not in [-1, -2] and action not in paths_dic.keys():
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"action值输入错误"}')
        xt.add_children(tcpudp_node, children_node_map)
        xt.add_properties(tcpudp_node, properties)
        return tcpudp_node

    # 创建TCP/UDP端口范围过滤
    def create_TCP_or_UDP_range(self,holowan_ip: str, holowan_port: str, engineID: int, TCPUDP: int, sourcePortMin: str, sourcePortMax: str,destPortMin: str, destPortMax:str, checkVersion: int, action: int):
        '''
        :param TCPUDP: 1：TCP，2：UDP, 3: SCTP
        :param sourcePortMin: 源端口号最小值
        :param sourcePortMax: 源端口号最大值
        :param destPortMin: 目的端口号最小值
        :param destPortMax: 目的端口号最大值
        :param checkVersion:
        :param action:
        :return:
        '''
        tcpudp_node = xt.create_node("tcp_udp", {}, {})
        children_node_map = {"type": TCPUDP, "src": "", "dst": "", "check": checkVersion, "path_id": action}
        properties = {}
        # ===============src================== #
        if sourcePortMin == "any" and sourcePortMax == "any":
            properties["src"] = {"any": 1}
        elif mt.isPort(sourcePortMin) and mt.isPort(sourcePortMax):
            properties["src"] = {"num": 1}
            children_node_map["src"] = {"port1": sourcePortMin+"-"+sourcePortMax}
        else:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"sourcePort必须为1-65535之间"}')
        # ===============dst================== #
        if destPortMin == "any":
            properties["dst"] = {"any": 1}
        elif mt.isPort(destPortMin) and mt.isPort(destPortMax):
            properties["dst"] = {"num": 1}
            children_node_map["dst"] = {"port1": destPortMin+"-"+destPortMax}
        else:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"dstPort必须为1-65535之间"}')
        # ===============checkVersion================== #
        if checkVersion not in [0, 4, 6]:
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"checkVersion必须为0、4、6中的一个"}')
        # ===============Action================== #
        paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        if action not in [-1, -2] and action not in paths_dic.keys():
            return RuntimeError(r'{"errCode":"-11","errMsg":"ERROR","errReason":"action值输入错误"}')
        xt.add_children(tcpudp_node, children_node_map)
        xt.add_properties(tcpudp_node, properties)
        return tcpudp_node


    def create_TCP_or_UDP_or_SCTP(self, holowan_ip: str, holowan_port: str, engineID: int, type: int, sourcePort: str, destPort: str, checkVersion: int, action: int):
        '''
                :param type: 1：TCP，2：UDP, 3: SCTP
                :param sourcePort: 源端口号
                :param destPort: 目标端口号
                :param checkVersion:
                :param action:
                :return:
        '''
        return self.create_TCP_or_UDP(holowan_ip, holowan_port, engineID, type, sourcePort, destPort, checkVersion, action)

    # 添加Combination(IPV4 and TCP)过滤器
    @checkParameter()
    def add_IPV4_TCP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                   destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDTCP, self.ADD, parameters)

    # 删除Combination(IPV4 and TCP)过滤器
    @checkParameter()
    def remove_IPV4_TCP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                        destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDTCP, self.REMOVE, parameters)

    # 删除所有Combination(IPV4 and TCP)过滤器
    @checkParameter()
    def remove_all_IPV4_TCP_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.IPV4ANDTCP, self.REMOVEALL, parameters)

    # 添加Combination(IPV4 and UDP)过滤器
    @checkParameter()
    def add_IPV4_UDP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                   destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDUDP, self.ADD, parameters)

    # 删除Combination(IPV4 and UDP)过滤器
    @checkParameter()
    def remove_IPV4_UDP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                        destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDUDP, self.REMOVE, parameters)


    # 删除所有Combination(IPV4 and UDP)过滤器
    @checkParameter()
    def remove_all_IPV4_UDP_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.IPV4ANDUDP, self.REMOVEALL, parameters)

    # 添加Combination(IPV4 and SCTP)过滤器
    @checkParameter()
    def add_IPV4_SCTP_to_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                   destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDSCTP, self.ADD, parameters)

    # 删除Combination(IPV4 and SCTP)过滤器
    @checkParameter()
    def remove_IPV4_SCTP_from_Classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int, sourceIP: str,
                               sourceMask: int, destinationIP: str, destinationMask: int, TOS: str, sourcePort: str,
                                        destPort: str, action: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "sourceIP": sourceIP, "sourceMask": sourceMask, "destinationIP": destinationIP,
                      "destinationMask": destinationMask, "TOS": TOS, "sourcePort": sourcePort,
                      "destPort": destPort, "checkVersion": 4, "action": action}
        return self.classifier(self.IPV4ANDUDP, self.REMOVE, parameters)


    # 删除所有Combination(IPV4 and UDP)过滤器
    @checkParameter()
    def remove_all_IPV4_SCTP_from_Classifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.IPV4ANDSCTP, self.REMOVEALL, parameters)

    # 创建Combination(IPV4 and TCP/UDP)节点
    @checkParameter()
    def create_IPV4_and_TCP_or_UDP(self, holowan_ip: str, holowan_port: str, engineID: int, TCPUDP:int, sourceIP: str,
                                   sourceMask: int, destinationIP: str, destinationMask: int, TOS: str,
                                   sourcePort: str, destPort: str, checkVersion: int, action: int):
        # 创建ipv4节点
        ipv4_node = self.create_IPV4(holowan_ip, holowan_port, engineID, sourceIP, sourceMask, destinationIP, destinationMask, TOS, action)
        if type(ipv4_node) == RuntimeError:
            return ipv4_node
        ipv4_node.remove(xt.get_node(ipv4_node, "./path_id"))
        # 创建TCP节点
        tcp_or_udp_node = self.create_TCP_or_UDP(holowan_ip, holowan_port, engineID, TCPUDP, sourcePort, destPort, checkVersion, action)
        if type(tcp_or_udp_node) == RuntimeError:
            return tcp_or_udp_node
        tcp_or_udp_node.remove(xt.get_node(tcp_or_udp_node, "./path_id"))
        # 创建comb混合节点
        comb_node = xt.create_node("comb", {}, {})
        children_node_map = {"path_id": action}
        xt.add_children(comb_node, children_node_map)
        comb_node.append(ipv4_node)
        comb_node.append(tcp_or_udp_node)
        return comb_node

    def print_nested_dict(self, dictionary: dict, indent=0):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print(f"{' ' * indent}{key}:")
                self.print_nested_dict(value, indent + 4)
            else:
                print(f"{' ' * indent}{key}: {value}")

    @checkParameter()
    def add_raw_1byte_to_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                    action: int, rule_list: list):

        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "rule_list": rule_list, "action": action}
        return self.classifier(self.RAW1BYTE, self.ADD, parameters)

    @checkParameter()
    def add_raw_4byte_to_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                    action: int, rule_list: list):

        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "rule_list": rule_list, "action": action}
        return self.classifier(self.RAW4BYTE, self.ADD, parameters)

    @checkParameter()
    def remove_raw_1byte_from_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                         action: int, rule_list: list):

        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "rule_list": rule_list, "action": action}
        return self.classifier(self.RAW1BYTE, self.REMOVE, parameters)

    @checkParameter()
    def remove_raw_4byte_from_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int,
                                         action: int, rule_list: list):

        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID,
                      "rule_list": rule_list, "action": action}
        return self.classifier(self.RAW4BYTE, self.REMOVE, parameters)

    @checkParameter()
    def remove_all_1byte_from_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.RAW1BYTE, self.REMOVEALL, parameters)

    @checkParameter()
    def remove_all_4byte_from_classifier(self, holowan_ip: str, holowan_port: str, engineID: int, portID: int):
        parameters = {"holowan_ip": holowan_ip, "holowan_port": holowan_port, "engineID": engineID, "portID": portID}
        return self.classifier(self.RAW4BYTE, self.REMOVEALL, parameters)

    def __create_raw(self, holowan_ip: str, holowan_port: str, engineID: int, rule_list: list, action: int, type: int):

        # 检查list是否正确
        check_list = [{"layer": 2, "offset": 0, "mask": "0xAA", "value": "0xAA"}]
        if not mt.validate_list(rule_list, check_list):
            return RuntimeError(r'{"errCode":"-9","errMsg":"ERROR","errReason":"rule_list格式错误"}')

        # 创建一个新节点
        raw_node = xt.create_node("raw_group", {}, {})
        action_node = xt.create_node("path_id", {}, "{}".format(action))

        for rule in rule_list:
            rule_node = xt.create_node("raw", {}, {})
            sub_node = xt.create_node("{}".format("type"), {}, "{}".format(type))
            rule_node.append(sub_node)

            for key, value in rule.items():
                sub_node = xt.create_node("{}".format(key), {}, "{}".format(value))
                rule_node.append(sub_node)

            raw_node.append(rule_node)

        raw_node.append(action_node)
        return raw_node

    # 获取报文分类器（Packet Classifier）最后一次配置参数
    @checkParameter()
    def get_Packet_Classifier_config_information(self, holowan_ip: str, holowan_port: str, engineID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :return: 修改报文分类器参数XML
        '''
        try:
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/classifier_config_info_{2}.xml".format(holowan_ip, holowan_port, engineID)
            return self.session.get(requestURL, verify=False).text
        except Exception:
            return RuntimeError(r'{"errCode":"-3","errMsg":"ERROR","errReason":"ENGINE has no found"}')

    # 添加/删除/删除所有过滤器
    # classifierType（IPV4: 0, IPV6: 1, MAC: 2, TCP: 3, UDP: 4, IPV4&TCP: 5, IPV4&UDP: 6）
    # operation（0: 添加一个, 1: 删除一个, 2: 删除所有）
    @checkParameter()
    def classifier(self, classifierType: int, operation: int, parameters: dict):
        holowan_ip = parameters.get("holowan_ip")
        holowan_port = parameters.get("holowan_port")
        engineID = parameters.get("engineID")
        portID = parameters.get("portID")
        classifierNode = None
        classifierNodeName = ""
        try:
            parent_node_path = "./port"
            classifierXMLStr = self.get_Packet_Classifier_config_information(holowan_ip, holowan_port, engineID)
            if type(classifierXMLStr) is RuntimeError:
                return str(classifierXMLStr)
        except Exception:
            return r'{{"errCode":"-3","errMsg":"ERROR","errReason":"ENGINE has no found"}}'
        root = xt.xmlString_to_Object(classifierXMLStr)
        nodes = xt.get_nodes(root, parent_node_path)
        port_node = None
        for node in nodes:
            if xt.get_node(node, "./port_id").text == str(portID):
                port_node = node
        if port_node is None:
            return r'{{"errCode":"-16",' \
                   r'"errMsg":"ConnectionError",' \
                   r'"errReason":"Engine{0} has no port{1}"}}'.format(engineID, portID)
        if operation == self.ADD or operation == self.REMOVE:
            # 创建classifier节点
            if classifierType == self.IPV4:
                sourceIP = parameters.get("sourceIP")
                sourceMask = parameters.get("sourceMask")
                destinationIP = parameters.get("destinationIP")
                destinationMask = parameters.get("destinationMask")
                TOS = parameters.get("TOS")
                action = parameters.get("action")
                classifierNodeName = "ipv4"
                # 创建ipv4节点
                classifierNode = self.create_IPV4(holowan_ip, holowan_port, engineID, sourceIP, sourceMask, destinationIP, destinationMask, TOS, action)
            elif classifierType == self.IPV6:
                sourceIP = parameters.get("sourceIP")
                destinationIP = parameters.get("destinationIP")
                action = parameters.get("action")
                classifierNodeName = "ipv6"
                # 创建ipv6节点
                classifierNode = self.create_IPV6(holowan_ip, holowan_port, engineID, sourceIP, destinationIP, action)
            elif classifierType == self.MAC:
                sourceMAC = parameters.get("sourceMAC")
                destinationMAC = parameters.get("destinationMAC")
                EtherType = parameters.get("EtherType")
                action = parameters.get("action")
                classifierNodeName = "mac"
                # 创建MAC节点
                classifierNode = self.create_MAC(holowan_ip, holowan_port, engineID, sourceMAC, destinationMAC, EtherType, action)
            elif classifierType == self.TCP:
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "tcp_udp"
                # 创建tcp节点
                classifierNode = self.create_TCP_or_UDP(holowan_ip, holowan_port, engineID, 1, sourcePort, destPort, checkVersion, action)
            elif classifierType == self.UDP:
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "tcp_udp"
                # 创建udp节点
                classifierNode = self.create_TCP_or_UDP(holowan_ip, holowan_port, engineID, 2, sourcePort, destPort, checkVersion, action)
            elif classifierType == self.IPV4ANDTCP:
                sourceIP = parameters.get("sourceIP")
                sourceMask = parameters.get("sourceMask")
                destinationIP = parameters.get("destinationIP")
                destinationMask = parameters.get("destinationMask")
                TOS = parameters.get("TOS")
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "comb"
                # 创建IPV4&TCP混合节点
                classifierNode = self.create_IPV4_and_TCP_or_UDP(holowan_ip, holowan_port, engineID, 1, sourceIP, sourceMask,
                                                            destinationIP, destinationMask, TOS, sourcePort,
                                                            destPort, checkVersion, action)
            elif classifierType == self.IPV4ANDUDP:
                sourceIP = parameters.get("sourceIP")
                sourceMask = parameters.get("sourceMask")
                destinationIP = parameters.get("destinationIP")
                destinationMask = parameters.get("destinationMask")
                TOS = parameters.get("TOS")
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "comb"
                # 创建IPV4&UDP混合节点
                classifierNode = self.create_IPV4_and_TCP_or_UDP(holowan_ip, holowan_port, engineID, 2, sourceIP, sourceMask,
                                                            destinationIP, destinationMask, TOS, sourcePort, destPort,
                                                            checkVersion, action)
            elif classifierType == self.RAW1BYTE:
                rule_list = parameters.get("rule_list")
                action = parameters.get("action")
                classifierNodeName = "raw_group"
                classifierNode = self.__create_raw(holowan_ip=holowan_ip, holowan_port=holowan_port, engineID=engineID,
                                                   rule_list=rule_list, action=action, type=1)
            elif classifierType == self.RAW4BYTE:
                rule_list = parameters.get("rule_list")
                action = parameters.get("action")
                classifierNodeName = "raw_group"
                classifierNode = self.__create_raw(holowan_ip=holowan_ip, holowan_port=holowan_port, engineID=engineID,
                                                   rule_list=rule_list, action=action, type=4)
            elif classifierType == self.SCTP:
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "tcp_udp"
                # 创建SCTP节点
                classifierNode = self.create_TCP_or_UDP_or_SCTP(holowan_ip, holowan_port, engineID, 3, sourcePort, destPort, checkVersion, action)
            elif classifierType == self.IPV4ANDSCTP:
                sourceIP = parameters.get("sourceIP")
                sourceMask = parameters.get("sourceMask")
                destinationIP = parameters.get("destinationIP")
                destinationMask = parameters.get("destinationMask")
                TOS = parameters.get("TOS")
                sourcePort = parameters.get("sourcePort")
                destPort = parameters.get("destPort")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "comb"
                # 创建IPV4&SCTP混合节点
                classifierNode = self.create_IPV4_and_TCP_or_UDP(holowan_ip, holowan_port, engineID, 3, sourceIP, sourceMask,
                                                                 destinationIP, destinationMask, TOS, sourcePort, destPort,
                                                                 checkVersion, action)
            elif classifierType == self.TCPRANGE:
                sourcePortMin = parameters.get("sourcePortMin")
                sourcePortMax = parameters.get("sourcePortMax")
                destPortMin = parameters.get("destPortMin")
                destPortMax = parameters.get("destPortMax")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "tcp_udp"
                # 创建TCP range节点
                classifierNode = self.create_TCP_or_UDP_range(holowan_ip, holowan_port, engineID, 1, sourcePortMin, sourcePortMax,destPortMin, destPortMax,
                                                        checkVersion, action)
            elif classifierType == self.UDPRANGE:
                sourcePortMin = parameters.get("sourcePortMin")
                sourcePortMax = parameters.get("sourcePortMax")
                destPortMin = parameters.get("destPortMin")
                destPortMax = parameters.get("destPortMax")
                checkVersion = parameters.get("checkVersion")
                action = parameters.get("action")
                classifierNodeName = "tcp_udp"
                # 创建TCP range节点
                classifierNode = self.create_TCP_or_UDP_range(holowan_ip, holowan_port, engineID, 2, sourcePortMin,
                                                        sourcePortMax, destPortMin, destPortMax,
                                                        checkVersion, action)

            # 若创建节点错误，则返回错误
            if type(classifierNode) == RuntimeError:
                return str(classifierNode)
            if operation == self.ADD:
                if classifierNode is not None:  # 把classifier节点添加到父节点中
                    port_node.append(classifierNode)
            elif operation == self.REMOVE:
                remove_flags = False
                classifierNodes = xt.get_nodes(port_node, classifierNodeName)
                for node in classifierNodes:
                    if xt.classifier_equal(classifierNode, node):
                        port_node.remove(node)
                        remove_flags = True
                        break
                if remove_flags is False:
                    return r'{{"errCode":"-15",' \
                           r'"errMsg":"ConnectionError",' \
                           r'"errReason":"Engine{0} port{1} has not such {2}"}}'.\
                        format(engineID, portID, classifierNodeName)
        elif operation == self.REMOVEALL:
            # 遍历所有对应的classifier节点
            if classifierType == self.IPV4:
                for node in xt.get_nodes(port_node, "./ipv4"):
                    port_node.remove(node)
            elif classifierType == self.IPV6:
                for node in xt.get_nodes(port_node, "./ipv6"):
                    port_node.remove(node)
            elif classifierType == self.MAC:
                for node in xt.get_nodes(port_node, "./mac"):
                    port_node.remove(node)
            elif classifierType == self.TCP:
                for node in xt.get_nodes(port_node, "./tcp_udp"):
                    if xt.get_node(node, "./type").text == "1":
                        port_node.remove(node)
            elif classifierType == self.UDP:
                for node in xt.get_nodes(port_node, "./tcp_udp"):
                    if xt.get_node(node, "./type").text == "2":
                        port_node.remove(node)
            elif classifierType == self.IPV4ANDTCP:
                for node in xt.get_nodes(port_node, "./comb"):
                    if xt.get_node(node, "./tcp_udp/type").text == "1":
                        port_node.remove(node)
            elif classifierType == self.IPV4ANDUDP:
                for node in xt.get_nodes(port_node, "./comb"):
                    if xt.get_node(node, "./tcp_udp/type").text == "2":
                        port_node.remove(node)
            elif classifierType == self.RAW1BYTE:
                for node in xt.get_nodes(port_node, "./raw_group"):
                    if xt.get_node(node, "./raw/type").text == "1":
                        port_node.remove(node)
            elif classifierType == self.RAW4BYTE:
                for node in xt.get_nodes(port_node, "./raw_group"):
                    if xt.get_node(node, "./raw/type").text == "4":
                        port_node.remove(node)
            elif classifierType == self.SCTP:
                for node in xt.get_nodes(port_node, "./tcp_udp"):
                    if xt.get_node(node, "./type").text == "3":
                        port_node.remove(node)
            elif classifierType == self.IPV4ANDSCTP:
                for node in xt.get_nodes(port_node, "./comb"):
                    if xt.get_node(node, "./tcp_udp/type").text == "3":
                        port_node.remove(node)

        pathXMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api), pathXMLStr, self.session)

    # 设置分类器白名单
    @checkParameter()
    def set_classifier_whitelist(self, holowan_ip:str, holowan_port:str, engineID:int, direction:int, type:int, ip_list:list):
        '''
            :param holowan_ip: str
            :param holowan_port: 端口号
            :param engindID: 引擎 ID
            :param direction: 方向
            :param type: 操作类型： 0->添加  1->删除 2->清空
            :return: json
        '''

        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/set_classifier_whitelist?engine_id={2}&direction={3}&type={4}".format(holowan_ip, holowan_port,engineID,direction,"add" if type==0 else "delete")
        if type == 2:
            msg = json.loads(self.get_classifier_whitelist(holowan_ip,holowan_port,engineID,direction))
            ip_list = msg["data"]["ip_list"]
        postJson = json.dumps({"ip_list":ip_list})
        headers = {"Content-Type": "text/json;charset=UTF-8"}
        response = self.session.post(requestURL,postJson.encode('utf-8'), headers=headers)
        response.encoding = "utf-8"
        return response.text

    # 获取分类器白名单
    @checkParameter()
    def get_classifier_whitelist(self,holowan_ip:str, holowan_port:str, engineID:int, direction:int):
        '''
            :param holowan_ip: str
            :param holowan_port: 端口号
            :param engindID: 引擎 ID
            :param direction: 方向
            :return: json
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/get_classifier_whitelist?engine_id={2}&direction={3}".format(holowan_ip,
                                                                                                           holowan_port,
                                                                                                           engineID,
                                                                                                           direction)
        response = self.session.get(requestURL, verify=False)
        response.encoding = "utf-8"
        return response.text
    '''============================================================================================================='''
    # 开启指定引擎
    @checkParameter()
    def start_engine(self, holowan_ip: str, holowan_port: str, engineID: int):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/start_running?engine={2}".format(holowan_ip, holowan_port, engineID)
        return self.session.get(requestURL, verify=False).text

    # 关闭指定引擎
    @checkParameter()
    def stop_engine(self, holowan_ip: str, holowan_port: str, engineID: int):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/stop_running?engine={2}".format(holowan_ip, holowan_port, engineID)
        return self.session.get(requestURL, verify=False).text

    # 保存当前HoloWAN设备xml信息，用于开机后自动运行保存配置
    @checkParameter()
    def save_HoloWAN_information(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: json 正确：errCode 等于 0，错误：errCode小于0（详见附录二（1.3修改虚拟链路配置错误编号））
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api)
        HoloWAN_information_xml = self.get_HoloWAN_information(holowan_ip, holowan_port)
        return mt.post_original_api(requestURL, HoloWAN_information_xml, self.session)

    # 判断链路是否存在
    def exist_path(self, holowan_ip:str, holowan_port:str, engineID:int, pathID:int, pathName):
        HoloWAN_information_str = self.get_HoloWAN_information(holowan_ip, holowan_port)
        HoloWAN_information_tree = xt.xmlString_to_Object(HoloWAN_information_str)
        engine_nodes = xt.get_nodes(HoloWAN_information_tree, "./e")
        for engine_node in engine_nodes:
            if str(engineID) == xt.get_node(engine_node, "./ei").text:
                path_nodes = xt.get_nodes(engine_node, "./ep/p")
                for path in path_nodes:
                    if str(pathID) == xt.get_node(path, "./pi").text:
                        return True
        return False

    # 设置path：仅修改一个tag的text值
    def set_one_tag_text(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, tagPath: str, tagText):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param tagPath: 标签tag的路径
        :param tagText: 标签tag内容改为tagtext值
        :return: Json [正确：erroCode等于0， 错误：erroCode小于0（详见附录二（1.2修改虚拟链路损伤错误编号））]
        '''
        try:
            pathXMLStr = self.get_path_config_information(holowan_ip, holowan_port, engineID, pathID)
            if pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE has no found"}}'.format(self.PCengineIDError) or \
                    pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathIDError) or \
                    pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathNotFound):
                return pathXMLStr
            root = xt.xmlString_to_Object(pathXMLStr)
            node = xt.get_node(root, tagPath)
            xt.set_node_text(node, tagText)
            pathXMLStr = xt.xmlObject_to_string(root)
            return mt.post_original_api(self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api), pathXMLStr, self.session)
        except RuntimeError:
            return RuntimeError

    # 设置path：修改父节点下所有的子节点
    def set_tag_children(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, pathDirection,
                         parent_node_ID,
                         children_node_Map, properties_map):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param pathDirection: 链路损伤方向 [1. 仅损伤下行，2. 仅损伤上行]
        :param parent_node_ID: 父节点tag
        :param children_node_Map: 子节点map
        :param properties_map: 子节点属性map
        :return:
        '''
        try:
            parent_node_path_list = []
            if pathDirection == 1:
                parent_node_path_list.append("pltr/" + parent_node_ID)
            elif pathDirection == 2:
                parent_node_path_list.append("prtl/" + parent_node_ID)
            elif pathDirection == 3:
                parent_node_path_list.append("pltr/" + parent_node_ID)
                parent_node_path_list.append("prtl/" + parent_node_ID)

            else:
                return r'{{"errCode":"{}","errMsg":"ERROR","errReason":"Error pathDirection"}}'.format(self.PCpathDirectionError)
            pathXMLStr = self.get_path_config_information(holowan_ip, holowan_port, engineID, pathID)
            if pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"ENGINE has no found"}}'.format(self.PCengineIDError) or \
                    pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathIDError) or \
                    pathXMLStr == r'{{"errCode":"{}","errMsg":"ERROR","errReason":"PATH has no found"}}'.format(self.PCpathNotFound):
                return pathXMLStr
            root = xt.xmlString_to_Object(pathXMLStr)
            for parent_node_path in parent_node_path_list:
                node = xt.get_node(root, parent_node_path)
                xt.remove_children(node)
                xt.add_children(node, children_node_Map)
                # if len(properties_map) != 0:  # 添加properties属性
                xt.add_properties(node, properties_map)
            pathXMLStr = xt.xmlObject_to_string(root)
            return mt.post_original_api(
                self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.emulator_config_api),
                pathXMLStr, self.session)
        except RuntimeError:
            return RuntimeError

    # 修改preferences xml下的节点数据
    def set_preferences_tag(self, holowan_ip: str, holowan_port: str, childern_node_Map):
        parent_node_path_list = []
        pathXMLStr = self.get_preferences(holowan_ip, holowan_port)
        root = xt.xmlString_to_Object(pathXMLStr)
        for node_path in childern_node_Map:
            node = xt.get_node(root, node_path)
            xt.set_node_text(node, childern_node_Map.get(node_path))

        pathXMLStr = xt.xmlObject_to_string(root)
        url = self._protocol_header(holowan_ip) + "{0}:{1}/set_preferences".format(holowan_ip, holowan_port)
        return mt.post_original_api(url, pathXMLStr, self.session)

        # node = xt.get_node(root, "zero_line")
        # print(root)
        # print(node)
        # for parent_node_path in parent_node_path_list:
        #     node = xt.get_node(root, parent_node_path)
        #     xt.remove_children(node)
        #     xt.add_children(node, children_node_Map)
        #     if len(properties_map) != 0:  # 添加properties属性
        #         xt.add_properties(node, properties_map)
        # pathXMLStr = xt.xmlObject_to_string(root)


    ''' ===============================================统计数据接口===================================================='''
    # 获取虚拟链路path每秒完成统计数据
    @checkParameter()
    def get_path_current_data(self, holowan_ip:str, holowan_port:str, engineID:int, pathID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :return: xml 详见附录一（1.5 PATH每秒统计数据 XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/current_resault_data?engine={2}&path={3}".format(holowan_ip, holowan_port,
                                                                                      engineID, pathID)
        return self.session.get(requestURL, verify=False).text

    # 获取指定引擎的虚拟链路每秒指定类型的统计数据
    @checkParameter()
    def get_path_graph_current(self, holowan_ip:str, holowan_port:str, engineID:int, pathID:int, id: int, type: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param id: 指定请求起始位置的ID编号，第一次该值设定为0
        :param type: String 详见附录三（1.1 返回数据类型列举）
        :return: xml 详见附录一（1.5 PATH每秒统计数据 XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/graph_current?engine={2}&path={3}&id={4}&type={5}".format(holowan_ip, holowan_port,
                                                                                      engineID, pathID, id, type)
        return self.session.get(requestURL, verify=False).text

    # # 获取所有引擎目前所有的历史数据点数量
    # def get_HoloWAN_history_data_count(self, holowan_ip, holowan_port):
    #     '''
    #     :param holowan_ip: holowan IP地址
    #     :param holowan_port: holowan 端口号
    #     :return: xml 详见附录一（1.7设备当前历史数据存在点XML格式）
    #     '''
    #     requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/{2}".format(holowan_ip, holowan_port, self.history_data_count)
    #     print(requestURL)
    #     return requests.get(requestURL, verify=False).text

    # 获取指定引擎的虚拟链路历史完全统计数据点组
    @checkParameter()
    def get_HoloWAN_history_entire_data(self, holowan_ip:str, holowan_port:str, engineID:int, pathID:int, last: int, count: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param last: 指定请求的起始位置，第一次默认从0开始
        :param count: 每次获取的数量，最多支持每次获取3600个数据点
        :return:
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/entire_resault_data?engine={2}&path={3}&last={4}&count={5}".format(holowan_ip, holowan_port, engineID, pathID, last, count)
        return self.session.get(requestURL, verify=False).text

    # 获取指定引擎的虚拟链路指定类型的历史统计数据点组
    @checkParameter()
    def get_path_graph_current_dataGroup(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, type: str, id: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :param pathID: 链路ID
        :param type: String 详见附录三（1.1 返回数据类型列举）
        :param id: 指定数据的位置
        :return:
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}//graph_current?engine={2}&path={3}&type={4}&id={5}".format(holowan_ip, holowan_port, engineID, pathID, type, id)
        return self.session.get(requestURL, verify=False).text

    # 清空指定引擎统计数据
    @checkParameter()
    def clean_engine_statistic_data(self, holowan_ip:str, holowan_port:str, engineID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :return: json  正确：errCode等于0， 错误：errCode小于0（详见附录二（1.5引擎数据清空错误编号））
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/clean_engine_resault_data?engine={2}".format(holowan_ip, holowan_port, engineID)
        return self.session.get(requestURL, verify=False).text

    # 将指定引擎（Engine）虚拟链路（PATH）存在的统计数据导出为CSV文件
    @checkParameter()
    def create_csv(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, filePath: str):
        # 必须先关闭引擎，才能下载
        self.stop_engine(holowan_ip, holowan_port, engineID)
        requestURL1 = self._protocol_header(holowan_ip) + "{0}:{1}/csv_data?type=1&engine={2}&path={3}".format(holowan_ip, holowan_port, engineID, pathID)
        response = self.session.get(requestURL1,verify=False)
        if eval(response.text)["errCode"] == "0":
            requestURL2 = self._protocol_header(holowan_ip) + "{0}:{1}/csv_data?type=2&engine={2}&path={3}".format(holowan_ip, holowan_port, engineID, pathID)
            response2 = self.session.get(requestURL2,verify=False)
            with open(filePath, "w") as f:
                f.write(response2.text)
        return response.text

    # ===============================================设备控制口相关接口====================================================#
    # 获取设备控制口（Control Prot）网络状态
    @checkParameter()
    def get_network_information(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: xml 详见附录一（1.8设备控制口状态XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/network_info".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    # 设置设备控制口（Control Prot）网络状态
    @checkParameter()
    def set_network(self, holowan_ip: str, holowan_port: str, hostName: str, ipAddress: str,
                    ipNetmask: str, gateway: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param hostName: Hostname名称
        :param ipAddress: ip地址
        :param ipNetmask: ip地址掩码
        :param gateway: 网关
        :return: json 正确：errCode等于0， 错误：errCode小于0（详见附录二（1.7配置控制口错误编号））
        '''
        network_xmlStr = self.get_network_information(holowan_ip, holowan_port)
        tree = xt.xmlString_to_Object(network_xmlStr)
        node = xt.get_node(tree, "./network_settings")
        xt.get_node(node, "./hostname").text = hostName
        if mt.isIP(ipAddress): xt.get_node(node, "./ipaddr").text = ipAddress
        if mt.isIP(ipNetmask): xt.get_node(node, "./netmask").text = ipNetmask
        if mt.isIP(gateway): xt.get_node(node, "./gateway").text = gateway
        nodes = xt.get_nodes(tree, "./network_settings/")
        network_node = xt.create_node("holowan_admin_network_config_info", {}, "")
        for node in nodes:
            network_node.append(node)
        xt.get_node(network_node, "./dhcp_switch").text = "1"       # dhcp_switch必须设置为1才能发送，若为off则报错
        xmlStr = xt.xmlObject_to_string(network_node)
        url = self._protocol_header(holowan_ip) + "{0}:{1}/setting_network".format(holowan_ip, holowan_port)
        return mt.post_original_api(url, xmlStr, self.session)

    ''' ===============================================设备业务口相关接口===================================================='''
    @checkParameter()
    def get_worker_port_information(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: xml 详见附录一（1.9设备业务口状态XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/worker_port_info".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    '''===============================================设备日志相关接口===================================================='''
    # 获取当前设备本次开机以来运行日志信息
    @checkParameter()
    def get_log(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: 设备状态描述文本
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/log".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text


    '''===============================================Playback相关接口===================================================='''
    # 上传HoloWAN Playback 回放txt文件
    @checkParameter()
    def upload_playback_file(self, holowan_ip: str, holowan_port: str, uploadFilePath: str):
        '''
        :param holowan_ip:
        :param holowan_port:
        :param uploadFilePath: 文件路径
        :return: xml
        '''
        fileType = mt.getFileType(uploadFilePath)
        if fileType == "txt":
            filename = ntpath.basename(uploadFilePath)
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/playback_file_upload?filename={2}".format(holowan_ip, holowan_port, filename)
            with open(uploadFilePath, "rb") as file:
                request_file = {str(len(open(uploadFilePath, "rb").read())): (filename, file, "text/plain")}
                return self.session.post(requestURL, files=request_file).text
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"上传的回放文件必须为.txt格式"}'

    # 查看回放文件列表
    @checkParameter()
    def playback_file_list(self, holowan_ip: str, holowan_port: str):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/playback_file_list".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    # 删除某个已上传的回放文件
    @checkParameter()
    def playback_file_delete(self, holowan_ip: str, holowan_port: str, filename: str):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/playback_file_delete?filename={2}".format(holowan_ip, holowan_port, filename)
        return self.session.get(requestURL, verify=False).text

    # 获取某个已上传的回放文件的数据内容
    @checkParameter()
    def get_playback_data(self, holowan_ip: str, holowan_port: str, filename: str, isBrief: bool):
        '''
        :param filename: Playback文件名
        :param is_brief: 是否仅获取简要内容
        '''
        brief = "true" if (isBrief == True) else "false"
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/get_playback_data?filename={2}&brief={3}".format(holowan_ip, holowan_port, filename, brief)
        return self.session.get(requestURL, verify=False).text

    # 应用回放文件到指定链路
    @checkParameter()
    def playback_apply(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, filename: str):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/playback_apply?filename={2}&eid={3}&pid={4}".format(holowan_ip, holowan_port, filename, engineID, pathID)
        return self.session.get(requestURL, verify=False).text

    # 释放回放文件
    @checkParameter()
    def playback_release(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/playback_release?eid={2}&pid={3}".format(holowan_ip, holowan_port, engineID, pathID)
        return self.session.get(requestURL, verify=False).text

    # 获取回放状态
    @checkParameter()
    def get_playback_status(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/get_playback_status?eid={2}&pid={3}".format(holowan_ip, holowan_port, engineID, pathID)
        return self.session.get(requestURL, verify=False).text

    # 修改回放状态（默认模式）
    @checkParameter()
    def set_playback_default_status(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, action: str):
        '''
        :param action: 要设置的状态，play表示播放，pause表示暂停
        '''
        if action == "play":
            action_num = 1
        elif action == "pause":
            action_num = 2
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"action的值必须为play(播放)或pause(暂停)"}'

        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/set_playback_status?eid={2}&pid={3}&action={4}".format(holowan_ip, holowan_port, engineID, pathID, action_num)
        return self.session.get(requestURL, verify=False).text

    # 修改回放状态（游标模式）
    @checkParameter()
    def set_playback_cursor_status(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, action: str, cursor: int):
        '''
        :param action: 要设置的状态，play表示播放，pause表示暂停
        :param cursor: 要设置的游标位置
        '''
        if action == "play":
            action_num = 1
        elif action == "pause":
            action_num = 2
        else:
            return '{"errCode":"-400","errMsg":"error","errReason":"action的值必须为play(播放)或pause(暂停)"}'

        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/set_playback_status?eid={2}&pid={3}&action={4}&cursor={5}".format(holowan_ip, holowan_port, engineID, pathID, action_num, cursor)
        return self.session.get(requestURL, verify=False).text

    # 修改回放的带宽/丢包/时延开关
    @checkParameter()
    def set_playback_switch(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int, switchList: list):
        '''
        :param switchList: 开关值列表，长度必须为6，各个开关值依次控制bandwidth1，bandwidth2，delay1，delay2，loss1，loss2
        '''
        if len(switchList) != 6:
            return '{"errCode":"-400","errMsg":"error","errReason":"switchList的长度必须为6，例如[1,0,1,0,1,0]，各个开关值依次控制bandwidth1,bandwidth2,delay1,delay2,loss1,loss2"}'
        for i in range(len(switchList)):
            if switchList[i] < 0 or switchList[i] > 1:
                return '{"errCode":"-400","errMsg":"error","errReason":"回放的开关值必须为0或1"}'

        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/set_playback_switch?eid={2}&pid={3}&bandwidth1={4}&bandwidth2={5}&delay1={6}&delay2={7}&loss1={8}&loss2={9}"\
            .format(holowan_ip, holowan_port, engineID, pathID, switchList[0], switchList[1], switchList[2], switchList[3], switchList[4], switchList[5])
        return self.session.get(requestURL, verify=False).text

    # ===============================================设备管理相关接口====================================================#
    # 设置时间同步接口
    @checkParameter()
    def set_sync_system_time(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        '''
        current_time = int(time.time())
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/sync_system_time?time={2}".format(holowan_ip, holowan_port, current_time)
        return self.session.get(requestURL, verify=False).text

    # 获取时间同步接口
    @checkParameter()
    def get_sync_system_time(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/get_system_time".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    # ===============================================设备管理相关接口====================================================#
    # 重启设备接口
    @checkParameter()
    def reboot(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/reboot".format(holowan_ip, holowan_port)
        self.session.get(requestURL, verify=False)

    # ===============================================偏好设置相关接口====================================================#
    # 获取偏好设置
    @checkParameter()
    def get_preferences(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: xml 详见附录一（1.10偏好设置XML格式）
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/get_preferences".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    # 设置偏好
    @checkParameter()
    def set_preferences(self, holowan_ip: str, holowan_port: str,
                        clean_buffer: bool, enable_jumbo_frame: bool,
                        zero_line: str, ignore_frame_overhead: int,
                        language: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param clean_buffer: 下发Path配置时，是否清空buffer，True：是；False：否
        :param enable_jumbo_frame: 巨型帧是否开启，0: 关闭，1: 开启
        :param zero_line: 报文时延起点，取值：receive_time或bandwitdh_time
        :param ignore_frame_overhead: 带宽计算逻辑，0: 物理模式，1: 软件模式
        :param language: 语言，0：English，1：中文
        :return: xml 详见附录二（1.8下发偏好设置错误编号）
        '''

        children_node_Map = {"clean_buffer": "true" if clean_buffer else "false",
                             "enable_jumbo_frame": 1 if enable_jumbo_frame else 0,
                             "zero_line": zero_line,
                             "ignore_frame_overhead": ignore_frame_overhead,
                             "language": language}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    # 设置是否清空buffer
    @checkParameter()
    def set_clean_buffer(self, holowan_ip: str, holowan_port: str, clean_buffer: bool):
        children_node_Map = {"clean_buffer": "true" if clean_buffer else "false"}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    # 设置是否开启巨型帧
    @checkParameter()
    def set_enable_jumbo_frame(self, holowan_ip: str, holowan_port: str, enable_jumbo_frame: bool):
        children_node_Map = {"enable_jumbo_frame": 1 if enable_jumbo_frame else 0}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    # 设置报文时延起点
    @checkParameter()
    def set_zero_line(self, holowan_ip: str, holowan_port: str, zero_line: str):
        children_node_Map = {"zero_line": zero_line}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    # 设置带宽计算逻辑
    @checkParameter()
    def set_ignore_frame_overhead(self, holowan_ip: str, holowan_port: str, ignore_frame_overhead: int):
        children_node_Map = {"ignore_frame_overhead": ignore_frame_overhead}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    # 设置语言
    @checkParameter()
    def set_language(self, holowan_ip: str, holowan_port: str, language: int):
        children_node_Map = {"language": language}
        return self.set_preferences_tag(holowan_ip, holowan_port, children_node_Map)

    '''===============================================新增功能===================================================='''
    # 获取PATH链路Name
    @checkParameter()
    def get_path_Name(self, holowan_ip:str, holowan_port:str, engineID:int, pathID: int) -> str:
        xmlData = self.get_path_config_information(holowan_ip, holowan_port, engineID, pathID)
        return xt.get_node(xt.xmlString_to_Object(xmlData), "./pn").text

    # 获取当前设备支持的引擎数
    def get_engine_count(self, holowan_ip: str, holowan_port: str) -> int:
        response = self.get_HoloWAN_information(holowan_ip=holowan_ip, holowan_port=holowan_port)
        return int(xt.get_node(xt.xmlString_to_Object(response), "./eeq").text)

    # 判断引擎是否存在
    def has_engine(self, holowan_ip: str, holowan_port: str, engineID: int):
        if engineID > 0 and engineID <= self.get_engine_count(holowan_ip, holowan_port):
            return True
        else:
            return False

    # 判断链路ID是否正确
    def right_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        if pathID > 0 and pathID <= 15:
            return True
        return False

    # 判断链路是否存在
    def has_path(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        if pathID not in self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID).keys():
            return False
        return True

    # 判断链路开启状态
    def path_is_open(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        responseInString = self.get_HoloWAN_information(holowan_ip, holowan_port)
        responseInXml = xt.xmlString_to_Object(responseInString)
        for engineNode in xt.get_nodes(responseInXml, "./e"):
            if xt.get_node(engineNode, "./ei").text == str(engineID):
                for pathNode in xt.get_nodes(engineNode, "./ep/p"):
                    if xt.get_node(pathNode, "./pi").text == str(pathID):
                        isOpen = xt.get_node(pathNode, "./ie").text
                        if isOpen == "1":
                            return False
                        elif isOpen == "2":
                            return True
        return False

        # print(xt.xmlObject_to_string(a))
        # b = xt.get_nodes(a, "./ep/p")
        # print(b)

        # for engineNode in xt.get_nodes(responseInXml, "./e"):
        #     if xt.get_node(engineNode, "./ei").text == str(engineID):
        #         for pathNode in xt.get_nodes(engineNode, "./ep/p"):
        #             if xt.get_node(pathNode, "./pi").text == pathID:
        #                 isOpen = xt.get_node(pathNode, "./ie").text
        #                 if isOpen == 1:
        #                     return False
        #                 elif isOpen == 2:
        #                     return True
        # return False

    # 获取指定引擎Engine上所有的链路ID和Name，返回字典{pathID: pathName}
    @checkParameter()
    def get_pathDict_from_engine(self, holowan_ip: str, holowan_port: str, engineID: int) -> dict:
        path_dic = {}
        holowanInformationXmlStr = self.get_HoloWAN_information(holowan_ip, holowan_port)
        engineNodes = xt.get_nodes(xt.xmlString_to_Object(holowanInformationXmlStr), "./e")
        for engineNode in engineNodes:
            each_engineID = xt.get_node(engineNode, "./ei").text
            if int(each_engineID) == engineID:
                pathNodes = xt.get_nodes(engineNode, "./ep/p")
                for pathNode in pathNodes:
                    pathID = int(xt.get_node(pathNode, "./pi").text)
                    pathName = xt.get_node(pathNode, "./pn").text
                    path_dic[pathID] = pathName
        return path_dic

    # 获取指定引擎Engine上所有的链路ID，返回列表list
    def get_pathIDList_from_engine(self, holowan_ip: str, holowan_port: str, engineID: int) -> list:
        pathDict = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        return list(pathDict.keys())

    # 重置链路PATH
    @checkParameter()
    def resetPath(self, holowan_ip: str, holowan_port: str, engineID: int, pathID: int):
        # pathName = self.get_path_Name(holowan_ip, holowan_port, engineID, pathID)
        # return self.init_path(holowan_ip, holowan_port, engineID, pathID, pathName)
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/reset_path?eid={2}&pid={3}".format(holowan_ip, holowan_port, engineID, pathID)
        return self.session.get(requestURL, verify=False).text

    # 重置过滤器
    @checkParameter()
    def resetClassifier(self, holowan_ip:str, holowan_port:str, engineID:int, portID: int):
        remove_all_IPV4_from_Classifier_return = self.remove_all_IPV4_from_Classifier(holowan_ip, holowan_port, engineID, portID)
        if json.loads(remove_all_IPV4_from_Classifier_return)["errCode"] != "0":
            return remove_all_IPV4_from_Classifier_return
        remove_all_IPV6_from_Classifier_return = self.remove_all_IPV6_from_Classifier(holowan_ip, holowan_port, engineID, portID)
        if json.loads(remove_all_IPV6_from_Classifier_return)["errCode"] != "0":
            return remove_all_IPV6_from_Classifier_return
        remove_all_MAC_from_Classifier_return = self.remove_all_MAC_from_Classifier(holowan_ip, holowan_port, engineID, portID)
        if json.loads(remove_all_MAC_from_Classifier_return)["errCode"] != "0":
            return remove_all_MAC_from_Classifier_return
        add_MAC_to_Classifier_return = self.add_MAC_to_Classifier(holowan_ip, holowan_port, engineID, portID, "any", "any", "any", 1)
        if json.loads(add_MAC_to_Classifier_return)["errCode"] != "0":
            return add_MAC_to_Classifier_return
        return r'{"errCode":"0","errMsg":"OK","errReason":"API Successful implementation"}'

    # 重置引擎 (port1、port2各自仅剩一个MAC Classifier，链路仅剩PATH 1，ID=1，开启状态)
    # @checkParameter()
    def resetEngine(self, holowan_ip: str, holowan_port: str, engineID: int):
        # paths_dic = self.get_pathDict_from_engine(holowan_ip, holowan_port, engineID)
        # # 重置链路ATH
        # for pathId in paths_dic.keys():
        #     pathName = paths_dic[pathId]
        #     remove_path_return = self.remove_path(holowan_ip, holowan_port, engineID, pathId, pathName, True)
        #     if json.loads(remove_path_return)["errCode"] != "0":
        #         return remove_path_return
        # add_path_return = self.add_path(holowan_ip, holowan_port, engineID, 1)
        # if json.loads(add_path_return)["errCode"] != "0":
        #     return add_path_return
        # open_path_return = self.open_path(holowan_ip, holowan_port, engineID, 1, "PATH 1")
        # if json.loads(add_path_return)["errCode"] != "0":
        #     return open_path_return
        # # 重置过滤器
        # resetClassifier_return1 = self.resetClassifier(holowan_ip=holowan_ip, holowan_port=holowan_port, engineID=engineID, portID=2*(engineID-1)+1)
        # if json.loads(resetClassifier_return1)["errCode"] != "0":
        #     return resetClassifier_return1
        # resetClassifier_return2 = self.resetClassifier(holowan_ip=holowan_ip, holowan_port=holowan_port, engineID=engineID, portID=2*(engineID-1)+2)
        # if json.loads(resetClassifier_return1)["errCode"] != "0":
        #     return resetClassifier_return2
        # return r'{"errCode":"0","errMsg":"OK","errReason":"API Successful implementation"}'
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/reset_engine?eid={2}".format(holowan_ip, holowan_port, engineID)
        return self.session.get(requestURL, verify=False).text

    # 获取线路选择信息
    @checkParameter()
    def getRouteSelectInfo(self, holowan_ip: str, holowan_port: str):
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/route_select_info".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text

    # 根据线路选择配置损伤
    @checkParameter()
    def configRouteSelect(self, holowan_ip: str, engineID: int, pathID: int, holowan_port: str, client: str, server: str, network_type: str, isp: str, uplink_direction: int):
        min = [None] * 2
        delay = [None] * 2
        loss = [None] * 2
        shake = [None] * 2
        resp = []
        postJson = '{{"client":"{0}","server":"{1}","NetType":"{2}","operator":"{3}"}}'.format(client, server, network_type, isp)
        headers = {"Content-Type": "text/json;charset=UTF-8"}
        impairInfo = self.session.post(self._protocol_header(holowan_ip) + "{0}:{1}/route_select_config".format(holowan_ip, holowan_port), postJson.encode('utf-8'), headers=headers)
        impairInfo.encoding = "utf-8"

        for key, value in (dict(json.loads(impairInfo.text))).items():
            if key == "D1":
                delay[0] = round(float(dict(value).get('delay')), 2)
                loss[0] = round(float(dict(value).get('loss')), 3)
                shake[0] = round(float(dict(value).get('shake')), 3)
                min[0] = round(delay[0] - shake[0], 2) if delay[0] > shake[0] else 0.0
            if key == "D2":
                delay[1] = round(float(dict(value).get('delay')), 2)
                loss[1] = round(float(dict(value).get('loss')), 3)
                shake[1] = round(float(dict(value).get('shake')), 3)
                min[1] = round(delay[1] - shake[1], 2) if delay[1] > shake[1] else 0.0
            if key == "uplink":
                bw_up = round(float(value), 2)
            if key == "downlink":
                bw_down = round(float(value), 2)

        if network_type == "2G":
            rate_unit = 2
        else:
            rate_unit = 3

        for d in range(1, 3):
            resp.append(self.set_Delay_Normal(holowan_ip, holowan_port, engineID, pathID, d, min[d - 1], delay[d - 1], shake[d - 1], 1))
            resp.append(self.set_Loss_Random(holowan_ip, holowan_port, engineID, pathID, d, loss[d - 1]))

            if d == uplink_direction:
                resp.append(self.set_path_Bandwidth_Fixed(holowan_ip, holowan_port, engineID, pathID, d, bw_up, rate_unit))
            else:
                resp.append(self.set_path_Bandwidth_Fixed(holowan_ip, holowan_port, engineID, pathID, d, bw_down, rate_unit))

        return resp

# ===============================================pixel设置相关接口====================================================#
    # 设置抓包的目标对象
    @checkParameter()
    def set_capture(self, holowan_ip: str, holowan_port: str, captureID: int, pathID: int, mode: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param pathID: 链路ID,支持1-15
        :param mode: 选择抓包的目标对象，分为 all_pkt,all_path,one_path
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_config?type=change_mode&cid={2}&pid={3}&mode={4}".format(holowan_ip,
                                                                                                      holowan_port,
                                                                                                      captureID, pathID,
                                                                                                      mode)
        return self.session.get(requestURL, verify=False).text

    # 清空抓包结果
    @checkParameter()
    def clean_capture(self, holowan_ip: str, holowan_port: str, captureID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_config?type=clean&cid={2}".format(holowan_ip, holowan_port, captureID)
        return self.session.get(requestURL, verify=False).text

    # 开始抓包
    @checkParameter()
    def run_capture(self, holowan_ip: str, holowan_port: str, captureID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_config?type=run&cid={2}".format(holowan_ip, holowan_port, captureID)
        return self.session.get(requestURL, verify=False).text

    # 停止抓包
    @checkParameter()
    def stop_capture(self, holowan_ip: str, holowan_port: str, captureID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_config?type=stop&cid={2}".format(holowan_ip, holowan_port, captureID)
        return self.session.get(requestURL, verify=False).text

    # 在Pixel Viewer页面导出PCAP
    @checkParameter()
    def Export_PCAP(self, holowan_ip: str, holowan_port: str, captureID: int, file_name: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param file_name: 保存的PCAP文件名
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_pcap?cid={2}&type=filtered".format(holowan_ip, holowan_port, captureID)
        return mt.create_and_write_file(file_name, self.session.get(requestURL, verify=False).content)


    # 获取Pixel Viewer中Filter Settings的配置参数
    @checkParameter()
    def get_capture_filter_rule_information(self, holowan_ip: str, holowan_port: str, engineID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param engineID: 引擎ID
        :return: 修改报文过滤参数XML
        '''
        try:
            requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter_rule_{2}.xml".format(holowan_ip, holowan_port, engineID)
            return self.session.get(requestURL, verify=False).text
        except Exception:
            xmlString = "<capture_filter><mac enable='0'><src any='1'>any</src><dst any='1'>any</dst><type any='1'>any</type></mac><ipv4 enable='0'><src any='1'>any</src><dst any='1'>any</dst><smask>32</smask><dmask>32</dmask><tos any='1'/></ipv4><ipv6 enable='0'><src any='1'>any</src><dst any='1'>any</dst></ipv6><tcp_udp enable='0'><type>1</type><src any='1'>any</src><dst any='1'>any</dst><check>0</check></tcp_udp><impair_flag enable='1'><after>1</after><before>0</before><cls_drop>0</cls_drop><all>1</all><no_impair>0</no_impair><mod>0</mod><frg_af>0</frg_af><dup>0</dup><bw_drop>0</bw_drop><los>0</los><ber>0</ber><reo>0</reo><bypass>0</bypass></impair_flag><direction enable='0'>0</direction><length enable='0'><min>0</min><max>1500</max></length><vlan enable='0'><fpcp any='1'>any</fpcp><fpid any='1'>any</fpid><enable_stag>0</enable_stag><spcp any='1'>any</spcp><spid any='1'>any</spid></vlan><pid enable='0'>1</pid></capture_filter>"
            return xmlString



    # 重置Filter Settings中所有选项
    @checkParameter()
    def Reset_Filter_Settings(self, holowan_ip: str, holowan_port: str, captureID: int,
                                engineID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param zenable: 选项开关， 0为关 1为开
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        root = xt.xmlString_to_Object(filterXMLStr)

        node1 = xt.get_node(root, ".")
        properties_Map = {"./pid": {"enable": 0}, "./direction": {"enable": 0}, "./impair_flag": {"enable": 1},
                          "./mac": {"enable": 0}, "./mac/src": {"any": 1}, "./vlan": {"enable": 0}, "./ipv4": {"enable": 0},
                          "./ipv6": {"enable": 0}, "./tcp_udp": {"enable": 0}, "./length": {"enable": 0}}
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中path的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_path(self, holowan_ip: str, holowan_port: str, captureID: int,
                                engineID: int, enable: str, pathID: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关， 0为关 1为开
        :param pathID: 链路ID
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        root = xt.xmlString_to_Object(filterXMLStr)
        node = xt.get_node(root, "./pid")
        xt.set_node_text(node, pathID)

        node1 = xt.get_node(root, ".")
        properties_Map = {"./pid": {"enable": enable}}
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)

    # 设置Filter Settings中Direction的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_direction(self, holowan_ip: str, holowan_port: str, captureID: int,
                                     engineID: int, enable: str, direction: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关， 0为关 1为开
        :param direction: 报文通过的方向，0表示Left to Rright   1表示Right to Lift
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        root = xt.xmlString_to_Object(filterXMLStr)
        node = xt.get_node(root, "./direction")
        xt.set_node_text(node, direction)

        node1 = xt.get_node(root, ".")
        properties_Map = {"./direction": {"enable": enable}}
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)

    # 设置Filter Settings中Type的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_type(self, holowan_ip: str, holowan_port: str, captureID: int,engineID: int,
                                enable: str,after: int, before: int, cls_drop: int, all: int, no_impair: int,
                                mod: int, frg_af: int, frg_be: int, dup: int, bw_drop: int, los: int, ber: int, reo: int, bypass: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable:
        :param after: after表示显示报文损伤后，0表示不选，1表示选择
        :param before: before表示显示报文损伤前,0表示不选，1表示选择
        :param cls_drop
        :param all:
        :param no_impair:
        :param mod:
        :param frg_af:
        :param dup:
        :param bw_drop:
        :param los:
        :param ber:
        :param reo:
        :param bypass:
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        if (after == 1 and before == 1) or (after == 0 and before == 0):
            return RuntimeError("{'errCode':'-526','errMsg':'ERROR','errReason': 'The value of after and before can not be same at the same time'}")

        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"after": after, "before": before, "cls_drop": cls_drop, "all": all, "no_impair": no_impair,
                             "mod": mod, "frg_af": frg_af, "frg_be": frg_be, "dup": dup, "bw_drop": bw_drop, "los": los, "ber": ber, "reo": reo,
                             "bypass": bypass}
        node = xt.get_node(root, "./impair_flag")

        node1 = xt.get_node(root, ".")
        properties_Map = {"./impair_flag": {"enable": enable}}
        xt.add_properties(node1, properties_Map)

        xt.remove_children(node)
        xt.add_children(node, children_node_Map)
        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中MAC的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_MAC(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,
                               enable: str, src: str, dst: str, EtherType: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param src: 源MAC地址
        :param dst: 目标MAC地址
        :param EitherType: 其他类型
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        # 获取根节点
        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"src": "", "dst": "", "type": ""}
        properties_Map = {}

        # 清空子节点的内容
        node = xt.get_node(root, "./mac")
        xt.remove_children(node)

        # 更改节点属性
        node1 = xt.get_node(root, ".")
        properties_Map["./mac"] = {"enable": enable}

        # ===============src================== #
        if src == "any":
            properties_Map["./mac/src"] = {"any": "1"}
            children_node_Map["src"] = "any"
        elif mt.isMac(src):
            children_node_Map["src"] = src
        else:
            return RuntimeError(r'{"errCode":"-527","errMsg":"ERROR","errReason":"src必须输入any或MAC地址"}')
        # ===============dst================== #
        if dst == "any":
            properties_Map["./mac/dst"] = {"any": "1"}
            children_node_Map["dst"] = "any"
        elif mt.isMac(dst):
            children_node_Map["dst"] = dst
        else:
            return RuntimeError(r'{"errCode":"-528","errMsg":"ERROR","errReason":"dst必须输入any或MAC地址"}')
        # ===============EitherType================== #
        if EtherType == "any":
            properties_Map["./mac/type"] = {"any": "1"}
            children_node_Map["type"] = "any"
        elif EtherType in self.etherType.split(", "):
            children_node_Map["type"] = EtherType
        else:
            return RuntimeError(r'{"errCode":"-529","errMsg":"ERROR","errReason":"MAC的EtherType输入值不正确"}')

        xt.add_children(node, children_node_Map)
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)

    # 设置Filter Settings中VLAN的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_VLAN(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,enable: str,
                                fpcp: str, fpid: str, enable_stag: int, spcp: str, spid: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param fpcp: 源MAC地址
        :param fpid: 目标MAC地址
        :param enable_stag:
        :param spcp:
        :param spid:
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        if fpid != "any":
            if int(fpid) < 0 or int(fpid) > 4095:
                return '{"errCode":"-530","errMsg":"ERROR","errReason": "The value range of "fpid" is 0 to 4095"}'

        if spid != "any":
            if int(spid) < 0 or int(spid) > 4095:
                return '{"errCode":"-531","errMsg":"ERROR","errReason": "The value range of "spid" is 0 to 4095"}'

        # 获取根节点
        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"fpcp": fpcp, "fpid": fpid, "enable_stag": enable_stag, "spcp": spcp, "spid": spid}
        properties_Map = {}

        # 清空子节点的内容
        node = xt.get_node(root, "./vlan")
        xt.remove_children(node)

        # 更改节点属性
        node1 = xt.get_node(root, ".")
        properties_Map["./vlan"] = {"enable": enable}


        if fpcp == "any":
            properties_Map["./vlan/fpcp"] = {"any": "1"}
        if fpid == "any":
            properties_Map["./vlan/fpid"] = {"any": "1"}
        if spcp == "any":
            properties_Map["./vlan/spcp"] = {"any": "1"}
        if spid == "any":
            properties_Map["./vlan/spid"] = {"any": "1"}

        xt.add_children(node, children_node_Map)
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中IPv4的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_IPv4(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,enable: str,
                                src: str, dst: str, smask: int, dmask: int, tos: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param src: 源MAC地址
        :param dst: 目标MAC地址
        :param smask:
        :param dmask:
        :param tos:
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        # 获取根节点
        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"src": "", "dst": "", "smask": smask, "dmask": dmask, "tos": ""}
        properties_Map = {}

        # 清空子节点的内容
        node = xt.get_node(root, "./ipv4")
        xt.remove_children(node)

        # 更改节点属性
        node1 = xt.get_node(root, ".")
        properties_Map["./ipv4"] = {"enable": enable}

        # ===============src================== #
        if src == "any":
            properties_Map["./ipv4/src"] = {"any": "1"}
            children_node_Map["src"] = "any"
        elif mt.isIPV4(src):
            children_node_Map["src"] = src
        else:
            return RuntimeError(r'{"errCode":"-532","errMsg":"ERROR","errReason":"src必须输入any或ipv4地址"}')
        # ===============dst================== #
        if dst == "any":
            properties_Map["./ipv4/dst"] = {"any": "1"}
            children_node_Map["dst"] = "any"
        elif mt.isIPV4(dst):
            children_node_Map["dst"] = dst
        else:
            return RuntimeError(r'{"errCode":"-533","errMsg":"ERROR","errReason":"dst必须输入any或ipv4地址"}')
        if tos == "any":
            properties_Map["./ipv4/tos"] = {"any": "1"}
        else:
            children_node_Map["tos"] = tos

        xt.add_children(node, children_node_Map)
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中IPv6的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_IPv6(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,enable: str,
                                src: str, dst: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param src: 源MAC地址
        :param dst: 目标MAC地址
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        # 获取根节点
        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"src": "", "dst": ""}
        properties_Map = {}

        # 清空子节点的内容
        node = xt.get_node(root, "./ipv6")
        xt.remove_children(node)

        # 更改节点属性
        node1 = xt.get_node(root, ".")
        properties_Map["./ipv6"] = {"enable": enable}

        # ===============src================== #
        if src == "any":
            properties_Map["./ipv6/src"] = {"any": "1"}
            children_node_Map["src"] = "any"
        elif mt.isIPV6(src):
            children_node_Map["src"] = src
        else:
            return RuntimeError(r'{"errCode":"-534","errMsg":"ERROR","errReason":"src必须输入any或ipv6地址"}')
        # ===============dst================== #
        if dst == "any":
            properties_Map["./ipv6/dst"] = {"any": "1"}
            children_node_Map["dst"] = "any"
        elif mt.isIPV6(dst):
            children_node_Map["dst"] = dst
        else:
            return RuntimeError(r'{"errCode":"-535","errMsg":"ERROR","errReason":"dst必须输入any或ipv6地址"}')

        xt.add_children(node, children_node_Map)
        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中TCP/UDP的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_TCP_UDP(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,enable: str,
                                type: int, src: str, dst: str, check: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param type:
        :param src: 源MAC地址
        :param dst: 目标MAC地址
        :param check:
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        if src != "any":
            if int(src) < 0 or int(src) > 65535:
                return '{"errCode":"-536","errMsg":"ERROR","errReason": "The value range of "src" is 0 to 65535"}'

        if dst != "any":
            if int(dst) < 0 or int(dst) > 65535:
                return '{"errCode":"-537","errMsg":"ERROR","errReason": "The value range of "dst" is 0 to 65535"}'

        # 获取根节点
        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"type": type, "src": src, "dst": dst, "check": check}

        properties_Map = {}

        # 清空子节点内容
        node = xt.get_node(root, "./tcp_udp")
        xt.remove_children(node)

        # 更改节点属性
        node1 = xt.get_node(root, ".")
        properties_Map["./tcp_udp"] = {"enable": enable}

        if src == "any":
            properties_Map["./tcp_udp/src"] = {"any": "1"}
        else:
            properties_Map["./tcp_udp/src"] = {"num": "1"}
            children_node_Map["src"] = {"port1":src}

        if dst == "any":
            properties_Map["./tcp_udp/dst"] = {"any": "1"}
        else:
            properties_Map["./tcp_udp/dst"] = {"num": "1"}
            children_node_Map["dst"] = {"port1":dst}

        xt.add_children(node, children_node_Map)

        xt.add_properties(node1, properties_Map)

        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 设置Filter Settings中Packet Length的选项
    @checkParameter()
    @checkPixelSettings()
    def set_capture_filter_Packet_Length(self, holowan_ip: str, holowan_port: str, captureID: int, engineID: int,enable: str,
                                min: int, max: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID 抓包ID
        :param engineID: 引擎ID
        :param enable: 选项开关
        :param min:
        :param max:
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        try:
            filterXMLStr = self.get_capture_filter_rule_information(holowan_ip, holowan_port, engineID)
        except Exception:
            return RuntimeError("Engine{} is not Found".format(engineID))

        root = xt.xmlString_to_Object(filterXMLStr)
        children_node_Map = {"min": min, "max": max}
        node = xt.get_node(root, "./length")

        node1 = xt.get_node(root, ".")
        properties_Map = {"./length": {"enable": enable}}
        xt.add_properties(node1, properties_Map)

        xt.remove_children(node)
        xt.add_children(node, children_node_Map)
        XMLStr = xt.xmlObject_to_string(root)
        return mt.post_original_api(
            self._protocol_header(holowan_ip) + "{0}:{1}/capture_filter?cid={2}".format(holowan_ip, holowan_port, captureID), XMLStr, self.session)


    # 在Pixel Viewer页面保存抓包内容
    @checkParameter()
    def Save_Pixel_File(self, holowan_ip: str, holowan_port: str, captureID: int,
                        file_name: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param file_name: 保存的名字
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_file?type=save&cid={2}&file_name={3}".format(holowan_ip, holowan_port, captureID, file_name)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面获取所有已保存的pixel文件信息
    @checkParameter()
    def get_all_Pixel_Files(self, holowan_ip: str, holowan_port: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_file?type=list".format(holowan_ip, holowan_port)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面加载pixel文件
    @checkParameter()
    def load_Pixel_File(self, holowan_ip: str, holowan_port: str,
                        file_name: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param file_name:文件名
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_file?type=load&file_name={2}".format(holowan_ip, holowan_port, file_name)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面释放pixel文件
    @checkParameter()
    def release_Pixel_File(self, holowan_ip: str, holowan_port: str,captureID: int,
                        file_name: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param file_name:文件名
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_file?type=release&cid={2}&file_name={3}".format(holowan_ip, holowan_port, captureID, file_name)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面删除pixel文件
    @checkParameter()
    def delete_Pixel_File(self, holowan_ip: str, holowan_port: str,
                        file_name: str):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param file_name:文件名
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_file?type=delete&file_name={2}".format(holowan_ip, holowan_port, file_name)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面获取转发速率统计
    @checkParameter()
    def get_tx_rate(self, holowan_ip: str, holowan_port: str,captureID: int,
                        time_unit: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param time_unit: 统计的时间间隔
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_statistic?cid={2}&type=tx_rate&time_unit={3}".format(holowan_ip, holowan_port, captureID, time_unit)
        return self.session.get(requestURL, verify=False).text


    # 在Pixel Viewer页面获取接收速率统计
    @checkParameter()
    def get_rx_rate(self, holowan_ip: str, holowan_port: str,captureID: int,
                        time_unit: int):
        '''
        :param holowan_ip: holowan IP地址
        :param holowan_port: holowan 端口号
        :param captureID: 抓包ID
        :param time_unit: 统计的时间间隔
        :return: json 正确：errCode 等于 0， 错误：errCode 小于0
        '''
        requestURL = self._protocol_header(holowan_ip) + "{0}:{1}/capture_statistic?cid={2}&type=rx_rate&time_unit={3}".format(holowan_ip, holowan_port, captureID, time_unit)
        return self.session.get(requestURL, verify=False).text

    def _protocol_header(self,holowan_ip:str)->str:
        if holowan_ip.find("https://")==-1:
            return "http://"
        else:
            return ""


if __name__ == '__main__':
    print("holowan")
    # holowan_ip = "192.168.1.223"
    # holowan_port = "8080"
    # engineID = 1
    # pathID = 1
    # pathDirection = 1
    # min = 1.0
    # mean = 50.0
    # stdDeviation = 10.0
    # enableReordering = 1
    # subholowan = SubHoloWAN(holowan_ip, holowan_port)
    # subResponse = subholowan.set_Delay_Normal()
    # print(subResponse)
    # print(os.path.abspath("123.txt"))
