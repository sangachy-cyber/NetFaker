import requests
from holowan.myUtils import XmlUtil as xt, MyUtil as mt
from functools import wraps
import inspect

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
                if type(arg) == int and param.annotation ==float:
                    arg = float(arg)
                if param.annotation != inspect._empty and not isinstance(arg, param.annotation):
                    raise TypeError("{} parameter type error".format(param.name))
                if param.name == "holowan_ip":
                    if mt.isIP(arg) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error IP address"}'
                if param.name == "holowan_port":
                    if mt.isPort(arg) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error Port"}'
                newArgs.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, int) and params[k].annotation == float:
                    v = float(v)
                if params[k].annotation != inspect._empty and not isinstance(v, params[k].annotation):
                    raise TypeError("{} parameter type error".format(params[k].name))
                if k == "holowan_ip":
                    if mt.isIP(v) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error IP address"}'
                if k == "holowan_port":
                    if mt.isPort(v) is False:
                        return '{"errCode":"-500","errMsg":"Parameter ERROR","errReason":"Error Port"}'
                newKwargs[k] = v
            # return function(*args, **kwargs)
            return function(*tuple(newArgs), **dict(newKwargs))
        return wrapper

class GetHoloWANParameter:
    _get_HoloWAN_information_api = "statistics_information"

    def __int__(self):
        pass

    # 获取当前HoloWAN设备xml信息
    @checkParameter()
    def get_HoloWAN_information(self, holowanIp: str, holowanPort: str) -> str:
        requestURL = "http://{0}:{1}/{2}".format(holowanIp, holowanPort, self._get_HoloWAN_information_api)
        return requests.get(requestURL).text

    # 获取当前设备支持的引擎数量
    def get_the_number_of_engines(self, holowanIp: str, holowanPort: str):
        holowanInformation: str = self.get_HoloWAN_information(holowanIp, holowanPort)
        tagPath = "eeq"
        condition = {}
        return xt.getStringTagText(holowanInformation, tagPath, condition)

    # 获取引擎下的链路数量
    def get_the_number_of_paths_in_engine(self, holowanIp: str, holowanPort: str):
        holowanInformation: str = self.get_HoloWAN_information(holowanIp, holowanPort)
        holowanObject = xt.xmlString_to_Object(holowanInformation)
        eNodeList = xt.get_nodes(holowanObject, "e")
        for eNode in eNodeList:
            pass


if __name__ == "__main__":
    holowanIp = "192.168.1.223"
    holowanPort = "8080"
    ghp = GetHoloWANParameter()
    response = ghp.get_the_number_of_engines(holowanIp, holowanPort)
    print(response)
