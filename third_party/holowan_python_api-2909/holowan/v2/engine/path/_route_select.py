"""
HoloWAN Network Emulator
Python API
"""
import json

import requests

from holowan.v2.engine.path._bandwidth import BandwidthFixed
from holowan.v2.engine.path._delay import DelayNormal
from holowan.v2.engine.path._loss import LossRandom
from holowan.v2.engine.path._path import Impairments, _DIRECTION_L2R, _DIRECTION_R2L
from holowan.v2._commons import _route_select_config_url


class RouteSelectClient:
    """线路选择客户端预定义

    """
    BeiJing: str = r"北京市"
    TianJin: str = r"天津市"
    HeBei: str = r"河北省"
    ShanXi: str = r"山西省"
    NeiMengGu: str = r"内蒙古自治区"
    LiaoNing: str = r"辽宁省"
    JiLin: str = r"吉林省"
    HeiLongJiang: str = r"黑龙江省"
    ShangHai: str = r"上海市"
    JiangSu: str = r"江苏省"
    ZheJiang: str = r"浙江省"
    AnHui: str = r"安徽省"
    FuJian: str = r"福建省"
    JiangXi: str = r"江西省"
    ShanDong: str = r"山东省"
    HeNan: str = r"河南省"
    HuBei: str = r"湖北省"
    HuNan: str = r"湖南省"
    GuangDong: str = r"广东省"
    GuangXi: str = r"广西壮族自治区"
    HaiNan: str = r"海南省"
    ChongQing: str = r"重庆市"
    SiChuan: str = r"四川省"
    GuiZhou: str = r"贵州省"
    YunNan: str = r"云南省"
    XiZang: str = r"西藏自治区"
    ShaanXi: str = r"陕西省"
    GanSu: str = r"甘肃省"
    QingHai: str = r"青海省"
    NingXia: str = r"宁夏回族自治区"
    XinJiang: str = r"新疆维吾尔自治区"


class RouteSelectServer:
    """线路选择服务器预定义

    """
    TianJin: str = r"天津"
    BeiJing: str = r"北京"
    ShangHai: str = r"上海"
    ChongQing: str = r"重庆"
    ChengDu: str = r"成都"
    HangZhou: str = r"杭州"
    NanJing: str = r"南京"
    FuZhou: str = r"福州"
    JiNan: str = r"济南"
    GuangZhou: str = r"广州"
    HeFei: str = r"合肥"
    ShenYang: str = r"沈阳"
    HongKong: str = r"中国香港"
    XiAn: str = r"西安"


class RouteSelectISP:
    """线路选择ISP预定义

    """
    CHINA_TELECOM: str = r"中国电信"
    CHINA_MOBILE: str = r"中国移动"
    CHINA_UNITED: str = r"中国联通"


class RouteSelectNetType:
    """线路选择网络类型预定义

    """
    TwoG: str = r"2G"
    ThreeG: str = r"3G"
    FourG: str = r"4G"
    WIFI: str = r"WiFi"

    def __init__(self):
        pass


class RouteSelect(object):
    """线路选择
    RouteSelect 是一组预定义的损伤，类似 Path，包含两个方向的损伤。
    每个方向的损伤都是一个 Impairments 对象。

    Notes:
        强制关键字传参

    Args:
        holowan_ip (IPAddress): HoloWAN 的 ip 地址。
        holowan_port(PortNumber): HoloWAN 的控制端口的端口号。
        client(str): 客户端
        server(str): 服务器
        network_type(str): 网络类型
        isp(str): 运营商
        uplink_direction(int): 方向

    """

    def __init__(self, *, holowan_ip, holowan_port, client: str, server: str, network_type: str, isp: str,
                 uplink_direction: int):
        self._holowan_ip = holowan_ip
        self._holowan_port = holowan_port
        self._client = client
        self._server = server
        self._network_type = network_type
        self._isp = isp
        self._uplink_direction = uplink_direction

        self._l2r = Impairments(1)
        self._r2l = Impairments(2)

        self._generate_impairments()

    def _generate_impairments(self):
        min = [None] * 2
        delay = [None] * 2
        loss = [None] * 2
        shake = [None] * 2
        resp = []
        postJson = '{{"client":"{0}","server":"{1}","NetType":"{2}","operator":"{3}"}}'.format(
            self._client, self._server,
            self._network_type, self._isp
        )
        headers = {"Content-Type": "text/json;charset=UTF-8"}
        impairInfo = requests.post(
            _route_select_config_url(
                self._holowan_ip, self._holowan_port
            ),
            postJson.encode('utf-8'),
            headers=headers
        )
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

        if self._network_type == "2G":
            rate_unit = 2
        else:
            rate_unit = 3

        for d in range(1, 3):
            delay_normal = DelayNormal(
                minimum=min[d - 1], mean=delay[d - 1],
                std_deviation=shake[d - 1], enable_reordering=1
            )
            if d == 1:
                self._l2r.delay = delay_normal
            else:
                self._r2l.delay = delay_normal

            loss_random = LossRandom(loss_rate=loss[d - 1])
            if d == 1:
                self._l2r.loss = loss_random
            else:
                self._r2l.loss = loss_random

            if d == self._uplink_direction:
                band_fixed = BandwidthFixed(rate=bw_up, unit=rate_unit)
                if d == 1:
                    self._l2r.bandwidth = band_fixed
                else:
                    self._r2l.bandwidth = band_fixed
            else:
                band_fixed = BandwidthFixed(rate=bw_down, unit=rate_unit)
                if d == 1:
                    self._l2r.bandwidth = band_fixed
                else:
                    self._r2l.bandwidth = band_fixed

    @property
    def l2r(self) -> Impairments:
        self._l2r.update()
        return self._l2r

    @property
    def r2l(self) -> Impairments:
        self._r2l.update()
        return self._r2l
