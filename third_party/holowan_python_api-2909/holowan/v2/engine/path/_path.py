"""
HoloWAN Network Emulator
Python API
"""

import xml.etree.cElementTree as ET

import holowan
from holowan.v2._holowan_types import HoloWANConfigXML, EngineID, PathID, check_parameter
from holowan.v2._xml_holder import XMLHolder
from holowan.v2.engine.path._bandwidth import bandwidth_factory, BandwidthFixed
from holowan.v2.engine.path._bg_utilization import bg_utilization_factory, BackgroundUtilizationDisable
from holowan.v2.engine.path._corruption import corruption_factory, BER
from holowan.v2.engine.path._delay import delay_factory, DelayConstant
from holowan.v2.engine.path._duplication import duplication_factory, DuplicationNormal
from holowan.v2.engine.path._frame_overhead import frame_overhead_factory, FrameOverhead24Ethernet
from holowan.v2.engine.path._impairment_base import (
    Bandwidth, _BANDWIDTH_TAG,
    BackgroundUtilization, _BG_UTILIZATION_TAG,
    QueueLimit, _QUEUE_LIMIT_TAG,
    Modify, _MODIFY_TAG,
    MTU, _MTU_TAG,
    FrameOverhead, _FRAME_OVERHEAD_TAG,
    Delay, _DELAY_TAG,
    Loss, _LOSS_TAG,
    Corruption, _CORRUPTION_TAG,
    Reordering, _REORDERING_TAG,
    Duplication, _DUPLICATION_TAG,
    ImpairmentTypes, _IMPAIRMENT_ENABLE_PROPERTY
)
from holowan.v2.engine.path._loss import loss_factory, LossRandom
from holowan.v2.engine.path._modify import modify_factory, ModifyDisable
from holowan.v2.engine.path._mtu import mtu_factory, MTUDisable
from holowan.v2.engine.path._queue_limit import queue_limit_factory, QueueLimitSimple
from holowan.v2.engine.path._reordering import reordering_factory, ReorderingNormal
from holowan.v2.utils import xml_util as xt

_PATH_CONFIG: str = r"pc"

_DIRECTION_L2R: str = r"pltr"
_DIRECTION_R2L: str = r"prtl"

_ENGINEID: str = r"eid"
_PATHID: str = r"pid"
_PATH_NAME: str = r"pn"
_PATH_DIRECTION: str = r"pd"

_PATH_REDIRECT: str = r"rd"

_PATH_CONFIG_VERSION: str = r"version"

_PATH_CONFIG_FORMAT: str = """<?xml version="1.0" encoding="UTF-8"?>
<pc>
    <version>0</version>
    <eid>1</eid>
    <pid>1</pid>
    <pn>PATH 1</pn>
    <pd>3</pd>
    <pltr></pltr>
    <prtl></prtl>
</pc>
"""

_impairments_dir_str = {
    1: r"pltr",
    2: r"prtl"
}

_impairments_dir_int = {
    r"pltr": 1,
    r"prtl": 2
}


class Impairments(XMLHolder):
    """损伤组
    一组损伤集合，设计为：引擎无关，路径无关。
    每组损伤，即一个 Impairments 对象包括：
        1. redirection(int)，默认为：0
        2. bandwidth(Bandwidth 子类)，默认为：BandwidthFixed(rate=1000,unit=3)
        3. background_utilization(BackgroundUtilization 子类)，默认为：BackgroundUtilizationDisable()
        4. queue_limit(QueueLimit 子类)，默认为：QueueLimitSimple(depth=250)
        5. modify(Modify 子类)，默认为：ModifyDisable()
        6. mtu(MTU 子类)，默认为：MTUDisable()
        7. frame_overhead(FrameOverhead 子类)，默认为：FrameOverhead24Ethernet()
        8. delay(Delay 子类)，默认为：DelayConstant(delay=0.0)
        9. loss(Loss 子类)，默认为：LossRandom(loss_rate=0.0)
        10. corruption(Corruption 子类)，默认为：BER(error_rate=0,error_rate_index=14,crc=1)
        11. reordering(Reordering 子类)，默认为：ReorderingNormal(probability=0,min_delay=0.1,max_delay=0.5)
        12. duplication(Duplication 子类)，默认为：DuplicationNormal(probability=0)

    Args:
            direction: 方向。分为 port1 -> port2 和 port2 -> port1。对应为 pltr 和 prtl

    Examples:
        >>> 创建一组 port1 -> port2,即 pltr 方向的损伤，所有损伤为默认状态
        >>> impair = Impairments(direction=1)

    """
    _bandwidth: Bandwidth
    _bakg_utilization: BackgroundUtilization
    _queue_limit: QueueLimit
    _modify: Modify
    _mtu: MTU
    _frame_overhead: FrameOverhead
    _delay: Delay
    _loss: Loss
    _corruption: Corruption
    _reorder: Reordering
    _dup: Duplication
    _redirect: int

    def __init__(self, direction: int):
        """构造一组损伤

        Args:
            direction: 方向。分为 port1 -> port2 和 port2 -> port1。对应为 pltr 和 prtl

        """
        super().__init__(_impairments_dir_str[direction])
        self._direction = _impairments_dir_str[direction]
        self._bandwidth = None
        self._bakg_utilization = None
        self._queue_limit = None
        self._modify = None
        self._mtu = None
        self._frame_overhead = None
        self._delay = None
        self._loss = None
        self._corruption = None
        self._reorder = None
        self._dup = None
        self._redirect = 0
        rd_node = ET.Element(_PATH_REDIRECT)
        rd_node.text = str(self._redirect)
        self.add_child(rd_node)
        self.reset()

    @property
    def direction(self) -> int:
        return _impairments_dir_int[self._direction]

    @direction.setter
    def direction(self, value: int):
        self._direction = _impairments_dir_str[value]
        self.update()

    def reset(self) -> None:
        """重置损伤为默认状态

        """
        self.reset_bandwidth()
        self.reset_background_utilization()
        self.reset_queue_limit()
        self.reset_modify()
        self.reset_mtu()
        self.reset_frame_overhead()
        self.reset_delay()
        self.reset_loss()
        self.reset_corruption()
        self.reset_reordering()
        self.reset_duplication()

        self.update()

    def reset_bandwidth(self) -> None:
        """重置带宽限制为默认状态

        """
        self._bandwidth = BandwidthFixed(rate=1000, unit=3)

    def reset_background_utilization(self) -> None:
        """重置背景流量为默认状态

        """
        self._bakg_utilization = BackgroundUtilizationDisable()

    def reset_queue_limit(self) -> None:
        """重置队列限制为默认状态

        """
        self._queue_limit = QueueLimitSimple(option=3)

    def reset_modify(self) -> None:
        """重置报文修改为默认状态

        """
        self._modify = ModifyDisable()

    def reset_mtu(self) -> None:
        """重置 MTU 限制为默认状态

        """
        self._mtu = MTUDisable()

    def reset_frame_overhead(self) -> None:
        """重置帧开销为默认状态

        """
        self._frame_overhead = FrameOverhead24Ethernet()
        self._frame_overhead.disable_impair()

    def reset_delay(self) -> None:
        """重置时延为默认状态

        """
        self._delay = DelayConstant(delay=0.0)
        self._delay.disable_impair()

    def reset_loss(self) -> None:
        """重置丢包为默认状态

        """
        self._loss = LossRandom(loss_rate=0.0)
        self._loss.disable_impair()

    def reset_corruption(self) -> None:
        """重置误码为默认状态

        """
        self._corruption = BER(error_rate=0, error_rate_index=14, crc=1)
        self.corruption.disable_impair()

    def reset_reordering(self) -> None:
        """重置报文乱序为默认状态

        """
        self._reorder = ReorderingNormal(probability=0, min_delay=0.1, max_delay=0.5)
        self._reorder.disable_impair()

    def reset_duplication(self) -> None:
        """重置报文重复为默认状态

        """
        self._dup = DuplicationNormal(probability=0)
        self._dup.disable_impair()

    # ==========================================================
    # def enable_impairment(self,type:str,value:bool)->None:
    #     if type == ImpairmentTypes.BANDWIDTH:
    #         # The bandwidth can not be disabled.
    #         pass
    #     elif type == ImpairmentTypes.BACKGROUND_UTILIZATION:
    #         if value:
    #             self.background_utilization.enable_impair()
    #         else:
    #             self.background_utilization.disable_impair()
    #     elif type == ImpairmentTypes.QUEUE_LIMIT:
    #         # The queue limit can not be disabled.
    #         pass
    #     elif type == ImpairmentTypes.MODIFY:
    #         if value:
    #             self.modify.enable_impair()
    #         else:
    #             self.modify.disable_impair()
    #     elif type == ImpairmentTypes.MTU:
    #         if value:
    #             self.mtu.enable_impair()
    #         else:
    #             self.mtu.disable_impair()
    #     elif type == ImpairmentTypes.DELAY:
    #         if value:
    #             self.delay.enable_impair()
    #         else:
    #             self.delay.disable_impair()
    #     elif type == ImpairmentTypes.LOSS:
    #         if value:
    #             self.loss.enable_impair()
    #         else:
    #             self.loss.disable_impair()
    #     elif type == ImpairmentTypes.CORRUPTION:
    #         if value:
    #             self.corruption.enable_impair()
    #         else:
    #             self.corruption.disable_impair()
    #     elif type == ImpairmentTypes.REORDERING:
    #         if value:
    #             self.reordering.enable_impair()
    #         else:
    #             self.reordering.disable_impair()
    #     elif type == ImpairmentTypes.DUPLICATION:
    #         if value:
    #             self.duplication.enable_impair()
    #         else:
    #             self.duplication.disable_impair()
    #     elif type == ImpairmentTypes.REDIRECT:
    #         if value:
    #             self.redirection = 1
    #         else:
    #             self.redirection = 0
    #     else:
    #         raise ValueError("Unkown impairment type: {0}.".format(type))

    # ==========================================================

    def update(self):
        """更新损伤以及自身的 xml 结构

        """
        self.clear_children()
        rd_node = ET.Element(_PATH_REDIRECT)
        rd_node.text = str(self._redirect)
        self.add_child(rd_node)

        if self.bandwidth != None:
            self.add_child(self._bandwidth.node)
        if self._bakg_utilization != None:
            self.add_child(self._bakg_utilization.node)
        if self._queue_limit != None:
            self.add_child(self._queue_limit.node)
        if self._modify != None:
            self.add_child(self._modify.node)
        if self._mtu != None:
            self.add_child(self._mtu.node)
        if self._frame_overhead != None:
            self.add_child(self._frame_overhead.node)
        if self._delay != None:
            self.add_child(self._delay.node)
        if self._loss != None:
            self.add_child(self._loss.node)
        if self._corruption != None:
            self.add_child(self._corruption.node)
        if self._reorder != None:
            self.add_child(self._reorder.node)
        if self._dup != None:
            self.add_child(self._dup.node)

    def clone_from(self, src: 'Impairments') -> 'Impairments':
        """复制一份损伤,不复制方向

        Args:
            impairments:

        """
        self._bandwidth = src.bandwidth
        self._bakg_utilization = src.background_utilization
        self._queue_limit = src.queue_limit
        self._modify = src.modify
        self._mtu = src.mtu
        self._frame_overhead = src.frame_overhead
        self._delay = src.delay
        self._loss = src.loss
        self._corruption = src.corruption
        self._reorder = src.reordering
        self._dup = src.duplication
        self._redirect = src.redirection
        self.update()
        return self

    @property
    def redirection(self) -> int:
        """重定向功能

        Notes:
            可读可写
        """
        return self._redirect

    @redirection.setter
    def redirection(self, value: int) -> None:
        self._redirect = value
        self.set_node_text(_PATH_REDIRECT, self._redirect)

    @property
    def bandwidth(self) -> Bandwidth:
        """带宽限制

        Notes:
            可读可写
        """
        return self._bandwidth

    @bandwidth.setter
    @check_parameter
    def bandwidth(self, value: Bandwidth) -> None:
        # TODO: 所有的 setter 都做类型检查
        self._bandwidth = value

    @property
    def background_utilization(self) -> BackgroundUtilization:
        """背景流量

        Notes:
            可读可写
        """
        return self._bakg_utilization

    @background_utilization.setter
    @check_parameter
    def background_utilization(self, value: BackgroundUtilization) -> None:
        self._bakg_utilization = value

    @property
    def queue_limit(self) -> QueueLimit:
        """队列限制

        Notes:
            可读可写
        """
        return self._queue_limit

    @queue_limit.setter
    @check_parameter
    def queue_limit(self, value: QueueLimit) -> None:
        self._queue_limit = value

    @property
    def modify(self) -> Modify:
        """报文修改

        Notes:
            可读可写
        """
        return self._modify

    @modify.setter
    @check_parameter
    def modify(self, value: Modify) -> None:
        self._modify = value

    @property
    def mtu(self) -> MTU:
        """MTU 限制

        Notes:
            可读可写
        """
        return self._mtu

    @mtu.setter
    @check_parameter
    def mtu(self, value: MTU) -> None:
        self._mtu = value

    @property
    def frame_overhead(self) -> FrameOverhead:
        """帧开销

        Notes:
            可读可写
        """
        return self._frame_overhead

    @frame_overhead.setter
    @check_parameter
    def frame_overhead(self, value: FrameOverhead) -> None:
        self._frame_overhead = value

    @property
    def delay(self) -> Delay:
        """时延

        Notes:
            可读可写
        """
        return self._delay

    @delay.setter
    @check_parameter
    def delay(self, value: Delay) -> None:
        self._delay = value

    @property
    def loss(self) -> Loss:
        """丢包

        Notes:
            可读可写
        """
        return self._loss

    @loss.setter
    @check_parameter
    def loss(self, value: Loss) -> None:
        self._loss = value

    @property
    def corruption(self) -> Corruption:
        """误码

        Notes:
            可读可写
        """
        return self._corruption

    @corruption.setter
    @check_parameter
    def corruption(self, value: Corruption) -> None:
        self._corruption = value

    @property
    def reordering(self) -> Reordering:
        """报文乱序

        Notes:
            可读可写
        """
        return self._reorder

    @reordering.setter
    @check_parameter
    def reordering(self, value: Reordering) -> None:
        self._reorder = value

    @property
    def duplication(self) -> Duplication:
        """报文重复

        Notes:
            可读可写
        """
        return self._dup

    @duplication.setter
    @check_parameter
    def duplication(self, value: Duplication) -> None:
        self._dup = value

    def __str__(self):
        res = r"Impairments({0}):".format(_impairments_dir_int[self._direction])
        res += "\n\t"
        res += self._bandwidth.__str__()
        res += "\n\t"
        res += self._bakg_utilization.__str__()
        res += "\n\t"
        res += self._queue_limit.__str__()
        res += "\n\t"
        res += self._modify.__str__()
        res += "\n\t"
        res += self._mtu.__str__()
        res += "\n\t"
        res += self._frame_overhead.__str__()
        res += "\n\t"
        res += self._delay.__str__()
        res += "\n\t"
        res += self._loss.__str__()
        res += "\n\t"
        res += self._corruption.__str__()
        res += "\n\t"
        res += self._reorder.__str__()
        res += "\n\t"
        res += self._dup.__str__()
        return res


class Path(XMLHolder):
    """HoloWAN Path类
    此类附属于 Engine 类，作为 Engine 的组件。
    每个 Path 对象，包含两个方向的损伤，分别为 l2r 方向和 r2l 方向。每组损伤为 Impairments 对象。
    每个 Impairments 对象包含的损伤有：
        1. redirection(int)
        2. bandwidth(Bandwidth 子类)
        3. background_utilization(BackgroundUtilization 子类)
        4. queue_limit(QueueLimit 子类)
        5. modify(Modify 子类)
        6. mtu(MTU 子类)
        7. frame_overhead(FrameOverhead 子类)
        8. delay(Delay 子类)
        9. loss(Loss 子类)
        10. corruption(Corruption 子类)
        11. reordering(Reordering 子类)
        12. duplication(Duplication 子类)

    每种损伤作为 Impairments 的 property，可读可写。

    Notes:
        Path 类不具备与 HoloWAN 的通讯能力，对 Path 所作的更改需要通过 Engine 应用到 HoloWAN。

    Examples:
        >>> engine = Engine(holowan_ip, holowan_port, engine_id)
        >>> path1 = engine.get_path_by_id(path_id)
        >>> path1.l2r.bandwidth = BandwidthFixed(rate=1000,unit=3)
        >>> engine.apply_path_configuration(path1.path_id)


    """
    _engine_id: holowan.v2.EngineID
    _path_id: holowan.v2.PathID
    _path_name: str
    _path_direction: int
    _l2r: Impairments
    _r2l: Impairments

    def __init__(self, path_cfg_xml: HoloWANConfigXML = None):
        if path_cfg_xml == None:
            path_cfg_xml = _PATH_CONFIG_FORMAT

        root = xt.xmlString_to_Object(path_cfg_xml)
        super().__init__(_PATH_CONFIG, list(root))
        self._engine_id = None
        self._l2r = Impairments(_impairments_dir_int[_DIRECTION_L2R])
        self._r2l = Impairments(_impairments_dir_int[_DIRECTION_R2L])
        self._path_id = None
        self._path_name = None
        self._path_direction = None
        self._init_from_xml()

    def _reserve_nodes(self, _version: str):
        version_node = ET.Element(_PATH_CONFIG_VERSION)
        version_node.text = _version
        self.add_child(version_node)

        engineID_node = ET.Element(_ENGINEID)
        engineID_node.text = str(self._engine_id)
        self.add_child(engineID_node)

        pathID_node = ET.Element(_PATHID)
        pathID_node.text = str(self._path_id)
        self.add_child(pathID_node)

        path_name_node = ET.Element(_PATH_NAME)
        path_name_node.text = str(self._path_name)
        self.add_child(path_name_node)

        direction_node = ET.Element(_PATH_DIRECTION)
        direction_node.text = str(self._path_direction)
        self.add_child(direction_node)

    def reset(self):
        """重置当前 path 的所有损伤

        """
        self._l2r.reset()
        self._r2l.reset()
        self.update()

    def update(self):
        """更新自身的 xml 结构

        """
        # save the version
        _version = self.findtext(_PATH_CONFIG_VERSION)
        self.clear_children()
        # restore version
        self._reserve_nodes(_version)
        self._l2r.update()
        self._r2l.update()
        self.add_child(self._l2r.node)
        self.add_child(self._r2l.node)

    def _construct_direction(self, direction: str) -> None:
        if direction == _DIRECTION_L2R:
            direction_node = self._l2r
        else:
            direction_node = self._r2l
        for imp in list(self.find(direction)):
            imp_name = imp.tag
            # get "enable" first
            # the "enable" can be None
            # GUI 1.0 do not have "enable"
            # GUI 2.0 has "enable"(usually but not always)
            enable = imp.get(_IMPAIRMENT_ENABLE_PROPERTY)
            if enable is None:
                enable = r"0"
            if imp_name == _BANDWIDTH_TAG:
                direction_node.bandwidth = bandwidth_factory(imp)
            elif imp_name == _BG_UTILIZATION_TAG:
                background_utilization = bg_utilization_factory(imp)
                background_utilization.set_property(background_utilization.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.background_utilization = background_utilization
            elif imp_name == _QUEUE_LIMIT_TAG:
                direction_node.queue_limit = queue_limit_factory(imp)
            elif imp_name == _MODIFY_TAG:
                modify = modify_factory(imp)
                modify.set_property(modify.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.modify = modify
            elif imp_name == _MTU_TAG:
                mtu = mtu_factory(imp)
                mtu.set_property(mtu.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.mtu = mtu
            elif imp_name == _FRAME_OVERHEAD_TAG:
                frame_overhead = frame_overhead_factory(imp)
                frame_overhead.set_property(frame_overhead.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.frame_overhead = frame_overhead
            elif imp_name == _DELAY_TAG:
                delay = delay_factory(imp)
                delay.set_property(delay.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.delay = delay
            elif imp_name == _LOSS_TAG:
                loss = loss_factory(imp)
                loss.set_property(loss.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.loss = loss
            elif imp_name == _CORRUPTION_TAG:
                corruption = corruption_factory(imp)
                corruption.set_property(corruption.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.corruption = corruption
            elif imp_name == _REORDERING_TAG:
                reordering = reordering_factory(imp)
                reordering.set_property(reordering.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.reordering = reordering
            elif imp_name == _DUPLICATION_TAG:
                duplication = duplication_factory(imp)
                duplication.set_property(duplication.tag, _IMPAIRMENT_ENABLE_PROPERTY, enable)
                direction_node.duplication = duplication
            elif imp_name == _PATH_REDIRECT:
                direction_node.redirection = imp.text
            else:
                raise ValueError("Unkown impairment type: {0}.".format(imp_name))

    def _init_from_xml(self):
        self._engine_id = int(self.findtext(_ENGINEID))
        self._path_id = int(self.findtext(_PATHID))
        self._path_name = self.findtext(_PATH_NAME)
        self._path_direction = int(self.findtext(_PATH_DIRECTION))

        self._construct_direction(_DIRECTION_L2R)
        self._construct_direction(_DIRECTION_R2L)
        self._l2r.update()
        self._r2l.update()

    @property
    def engine_id(self) -> EngineID:
        return self._engine_id

    @engine_id.setter
    def engine_id(self, value: EngineID) -> None:
        self._engine_id = value

    @property
    def path_id(self) -> PathID:
        return self._path_id

    @path_id.setter
    def path_id(self, value: PathID) -> None:
        self._path_id = value

    @property
    def path_name(self) -> str:
        return self._path_name

    @path_name.setter
    def path_name(self, value: str) -> None:
        self._path_name = value

    @property
    def l2r(self) -> Impairments:
        return self._l2r

    @property
    def r2l(self) -> Impairments:
        return self._r2l

    @l2r.setter
    def l2r(self, value: Impairments) -> None:
        self._l2r = value
        self.update()

    @r2l.setter
    def r2l(self, value: Impairments) -> None:
        self._r2l = value
        self.update()

    def __str__(self):
        res = "Path(engine_id:{0},path_id:{1},name:{2}):\n".format(self._engine_id, self._path_id, self._path_name)
        res += self._l2r.__str__()
        res += "\n"
        res += self._r2l.__str__()
        return res
