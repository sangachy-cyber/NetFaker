"""
HoloWAN Network Emulator
Python API

Impairment Base Class

==========================================================
Why not design a base class for different impairments?

Because I think creating a long inheritance chain (depth>3) is not good.
A longer inheritance chain may be more elegant, but it is also more difficult to understand.

So one day, if there will be some new common features(for example, the "enable") for all types of impairments,
just copy it for each type of impairments. It is not a big deal.
==========================================================

"""

import xml.etree.cElementTree as ET
from abc import abstractmethod

from holowan.v2._holowan_types import check_parameter
from holowan.v2._xml_holder import XMLHolder

_IMPAIRMENT_TYPE: str = r"s"
_IMPAIRMENT_ENABLE_PROPERTY: str = r"enable"


class ImpairmentTypes(object):
    BANDWIDTH: str = r"bandwidth"
    BACKGROUND_UTILIZATION: str = r"background_utilization"
    QUEUE_LIMIT: str = r"queue_limit"
    MODIFY: str = r"modify"
    MTU: str = r"mtu"
    FRAME_OVERHEAD: str = r"frame_overhead"
    REDIRECT: str = r"redirect"
    DELAY: str = r"delay"
    LOSS: str = r"loss"
    CORRUPTION: str = r"corruption"
    REORDERING: str = r"reordering"
    DUPLICATION: str = r"duplication"


################################################################
_BANDWIDTH_TAG: str = r"bd"


class Bandwidth(XMLHolder):
    r"""Base class for bandwidth limitation impairments.

    The bandwidth limitations include:
        1. BandwidthFixed
        2. BandwidthJitter
        3. BandwidthTokenBucket

    See in '_bandwidth.py'.
    The three classes subclass this class.

    The bandwidth base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>    <type></type>
        >>>    <cfg_params>
        >>>    ...
        >>>    </cfg_params>
        >>> </tag>

    In the base class, only construct the <tag> and <type>.
    The details of the bandwidth limitation: <cfg_params>, will be constructed in subclass.

    """

    def __init__(self, mode: int, detail_tag: str):
        super().__init__(_BANDWIDTH_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self._detail_tag = detail_tag
        self.add_child(type_node)

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)
        return root

    @property
    def mode(self) -> int:
        return self._mode

    def __str__(self) -> str:
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


################################################################
_BG_UTILIZATION_TAG: str = r"bg"


class BackgroundUtilization(XMLHolder):
    r"""Base class for background utilization impairments.

    The background utilization include:
        1. BackgroundUtilizationRandom

    See in '_bg_utilization.py'.
    The class subclass this class.

    The background utilization base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <cfg_params></cfg_params>
        >>>
        >>>     <cfg_params></cfg_params>
        >>> </tag>

    In the base class, only construct the <tag> and its children.

    """

    def __init__(self, mode: int):
        super().__init__(_BG_UTILIZATION_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self.add_child(type_node)
        self._update_param()
        self.enable_impair()

    def _update_param(self) -> None:
        self.clear_children()
        self.add_children(self.impairment_params)

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


################################################################
_QUEUE_LIMIT_TAG: str = r"ql"


class QueueLimit(XMLHolder):
    r"""Base class for queue limit impairments.

    The queue limit include:
        1. QueueLimitSimple
        2. QueueLimitDropTail
        3. QueueLimitRED

    See in '_queue_limit.py'.
    The three classes subclass this class.

    The queue limit base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params>
        >>>
        >>>     </cfg_params>
        >>> </tag>

    In the base class, only construct the <tag> and <type>.
    The details of the queue limit: <cfg_params>, will be constructed in subclass.

    """

    def __init__(self, mode: int, detail_tag: str):
        super().__init__(_QUEUE_LIMIT_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self._detail_tag = detail_tag
        self.add_child(type_node)

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)
        return root

    @property
    def mode(self) -> int:
        return self._mode

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


################################################################
_MODIFY_TAG: str = r"md"

_MODIFY_CRC: str = r"crc"
_MODIFY_CHECKSUM: str = r"checksum"

_MATCH_HEADER: str = r"pt"
_MATCH_OFFSET: str = r"po"
_MATCH_SIZE: str = r"ps"
_MATCH_VALUE: str = r"pv"

_MODIFY_HEADER: str = r"mt"
_MODIFY_OFFSET: str = r"mo"
_MODIFY_SIZE: str = r"ms"
_MODIFY_VALUE: str = r"mv"

_MODIFY_TYPE_TAG: str = r"cs"


class Modify(XMLHolder):
    r"""Base class for modification impairments.

    The queue limit include:
        1. ModifyDisable
        2. ModifyNormal
        3. ModifyCycle
        4. ModifyRandom

    See in '_modify.py'.
    The four classes subclass this class.

    The modification base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params 1></cfg_params 1>
        >>>     <cfg_params n></cfg_params n>
        >>>
        >>> </tag>

    In the base class, construct the children of <tag>.

    """

    def __init__(self, mode: int) -> None:
        super().__init__(_MODIFY_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        self._mode = mode
        self.enable_impair()

    def _update_params(self):
        self.clear_children()
        self.add_children(self.impairment_params)

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def match_header(self) -> int:
        return self.impairment_params[_MATCH_HEADER]

    @match_header.setter
    @check_parameter
    def match_header(self, value: int) -> None:
        self.impairment_params[_MATCH_HEADER] = value
        self._update_params()

    # ===========================================================
    @property
    def match_offset(self) -> int:
        return self.impairment_params[_MATCH_OFFSET]

    @match_offset.setter
    @check_parameter
    def match_offset(self, value: int) -> None:
        self.impairment_params[_MATCH_OFFSET] = value
        self._update_params()

    # ===========================================================
    @property
    def match_size(self) -> int:
        return self.impairment_params[_MATCH_SIZE]

    @match_size.setter
    @check_parameter
    def match_size(self, value: int) -> None:
        self.impairment_params[_MATCH_SIZE] = value
        self._update_params()

    # ===========================================================
    @property
    def match_value(self) -> str:
        return self.impairment_params[_MATCH_VALUE]

    @match_value.setter
    @check_parameter
    def match_value(self, value: str) -> None:
        self.impairment_params[_MATCH_VALUE] = value
        self._update_params()

    # ===========================================================
    @property
    def modify_header(self) -> int:
        return self.impairment_params[_MODIFY_HEADER]

    @modify_header.setter
    @check_parameter
    def modify_header(self, value: int) -> None:
        self.impairment_params[_MODIFY_HEADER] = value
        self._update_params()

    # ===========================================================
    @property
    def modify_offset(self) -> int:
        return self.impairment_params[_MODIFY_OFFSET]

    @modify_offset.setter
    @check_parameter
    def modify_offset(self, value: int) -> None:
        self.impairment_params[_MODIFY_OFFSET] = value
        self._update_params()

    # ===========================================================
    @property
    def modify_size(self) -> int:
        return self.impairment_params[_MODIFY_SIZE]

    @modify_size.setter
    @check_parameter
    def modify_size(self, value: int) -> None:
        self.impairment_params[_MODIFY_SIZE] = value
        self._update_params()

    # ===========================================================
    @property
    def modify_value(self) -> str:
        return self.impairment_params[_MODIFY_VALUE]

    @modify_value.setter
    @check_parameter
    def modify_value(self, value: str) -> None:
        self.impairment_params[_MODIFY_VALUE] = value
        self._update_params()

    # ===========================================================
    @property
    def crc(self) -> int:
        return self.impairment_params[_MODIFY_CRC]

    @crc.setter
    @check_parameter
    def crc(self, value: int) -> None:
        self.impairment_params[_MODIFY_CRC] = value
        self._update_params()

    # ===========================================================
    @property
    def checksum(self) -> int:
        return self.impairment_params[_MODIFY_CHECKSUM]

    @checksum.setter
    @check_parameter
    def checksum(self, value: int) -> None:
        self.impairment_params[_MODIFY_CHECKSUM] = value
        self._update_params()

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            if k == _MODIFY_TYPE_TAG:
                continue
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_MTU_TAG: str = r"m"
_MTU_TYPE: str = r"s"


class MTU(XMLHolder):
    r"""Base class for mtu impairments.

    The queue limit include:
        1. MTUDisable
        2. MTULimit

    See in '_mtu.py'.
    The two classes subclass this class.

    The mtu base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params 1></cfg_params 1>
        >>>
        >>>     <cfg_params n></cfg_params n>
        >>> </tag>

    In the base class, construct the children of <tag>.

    """

    def __init__(self, mode: int):
        super().__init__(_MTU_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        self._mode = mode
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    @property
    def mode(self) -> int:
        return self._mode

    def _update_param(self):
        self.clear_children()
        self.add_children(self.impairment_params)

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            if k == _MTU_TYPE:
                continue
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_FRAME_OVERHEAD_TAG: str = r"fo"

_FRAME_OVERHEAD_RATE: str = r"r"


class FrameOverhead(XMLHolder):
    r"""Base class for frame overhead impairments.

    The frame overhead include:
        1. FrameOverhead24Ethernet
        2. FrameOverhead4Ethernet
        3. FrameOverheadCustom

    See in '_frame_overhead.py'.
    The three classes subclass this class.

    The frame overhead base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params 1></cfg_params 1>
        >>>
        >>>     <cfg_params n></cfg_params n>
        >>> </tag>

    In the base class, construct the children of <tag>.

    """

    def __init__(self, mode: int, params: dict):
        super().__init__(_FRAME_OVERHEAD_TAG, params)
        self.impairment_params = params
        self._param_tag2str = {}
        self._mode = mode
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    @property
    def mode(self) -> int:
        return self._mode

    def _update_param(self):
        self.clear_children()
        self.add_children(self.impairment_params)

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        res += r"{0}:{1} ".format("rate", self.impairment_params[_FRAME_OVERHEAD_RATE])

        res += "}"
        return res


_DELAY_TAG: str = r"d"


class Delay(XMLHolder):
    r"""Base class for delay impairments.

    The delay include:
        1. DelayConstant
        2. DelayUniform
        3. DelayNormal
        4. DelayCustom
        5. DelayJitter
        6. DelayGamma
        7. DelayAccumulateBurst

    See in '_delay.py'.
    The seven classes subclass this class.

    The delay base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <delay_type>
        >>>         <param></param>
        >>>
        >>>     </delay_type>
        >>> </tag>

    In the base class, only construct the children of <tag>.
    The details of <delay_type> will be constructed in the method: _construct_details.
    For special details, the method: _construct_details in subclasses must be overwritten.

    """

    def __init__(self, mode: int, detail_tag: str):
        super().__init__(_DELAY_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self._detail_tag = detail_tag
        self.add_child(type_node)
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        return root

    @property
    def mode(self) -> int:
        return self._mode

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_LOSS_TAG: str = r"l"


class Loss(XMLHolder):
    r"""Base class for loss impairments.

    The loss include:
        1. LossRandom
        2. LossCycle
        3. LossBurst
        4. LossGilbertElliott
        5. LossJitter
        6. LossMarkov

    See in '_loss.py'.
    The six classes subclass this class.

    The loss base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <loss_type>
        >>>         <param></param>
        >>>
        >>>     </loss_type>
        >>>
        >>> </tag>

    In the base class, only construct the children of <tag>.
    The details of <loss_type> will be constructed in the method: _construct_details.
    For special details, the method: _construct_details in subclasses must be overwritten.

    """

    def __init__(self, mode: int, detail_tag: str):
        super().__init__(_LOSS_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self._detail_tag = detail_tag
        self.add_child(type_node)
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        return root

    @property
    def mode(self) -> int:
        return self._mode

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_CORRUPTION_TAG: str = r"cor"


class Corruption(XMLHolder):
    r"""Base class for corruption impairments.

    The corruption include:
        1. BER
        2. BERRange
        3. BERPacket

    See in '_corruption.py'.
    The three classes subclass this class.

    The corruption base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params 1></cfg_params 1>
        >>>     <cfg_params n></cfg_params n>
        >>> </tag>

    In the base class, construct the children of <tag>.

    """

    def __init__(self, mode: int):
        super().__init__(_CORRUPTION_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self.add_child(type_node)
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    @property
    def mode(self) -> int:
        return self._mode

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _update_param(self):
        self.clear_children()
        self.add_children(self.impairment_params)

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_REORDERING_TAG: str = r"reo"


class Reordering(XMLHolder):
    r"""Base class for reordering impairments.

    The reordering include:
        1. ReorderingNormal
        2. ReorderingJitter
        3. ReorderingCycle

    See in '_reordering.py'.
    The three classes subclass this class.

    The reordering base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <reo_type>
        >>>         <param></param>
        >>>
        >>>         <param></param>
        >>>    </reo_type>
        >>> </tag>

    In the base class, only construct the children of <tag>.
    The details of <loss_type> will be constructed in the method: _construct_details.
    For special details, the method: _construct_details in subclasses must be overwritten.

    """

    def __init__(self, mode: int, detail_tag: str):
        super().__init__(_REORDERING_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self._detail_tag = detail_tag
        self.add_child(type_node)
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    def _update_param(self) -> None:
        self.clear_children()
        self.add_child(self._construct_details())

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _construct_details(self) -> ET.Element:
        root = ET.Element(self._detail_tag)
        for k, v in self.impairment_params.items():
            node = ET.Element(k)
            node.text = str(v)
            root.append(node)

        return root

    @property
    def mode(self) -> int:
        return self._mode

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...


_DUPLICATION_TAG: str = r"du"


class Duplication(XMLHolder):
    r"""Base class for duplication impairments.

    The duplication include:
        1. DuplicationNormal
        2. DuplicationJitter

    See in '_duplication.py'.
    The two classes subclass this class.

    The duplication base class subclass XMLHolder.

    The structure likes:
        >>> <tag>
        >>>     <type></type>
        >>>     <cfg_params 1></cfg_params 1>
        >>>     ...
        >>>     <cfg_params n></cfg_params n>
        >>> </tag>

    In the base class, construct the children of <tag>.

    """

    def __init__(self, mode: int):
        super().__init__(_DUPLICATION_TAG)
        self.impairment_params = {}
        self._param_tag2str = {}
        type_node = ET.Element(_IMPAIRMENT_TYPE)
        type_node.text = str(mode)
        self._mode = mode
        self.add_child(type_node)
        self.enable_impair()

    @property
    def enable(self) -> bool:
        flag = self.get(_IMPAIRMENT_ENABLE_PROPERTY)
        if flag == "0":
            return False
        else:
            return True

    def enable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "1")

    def disable_impair(self):
        self.set_property(self.tag, _IMPAIRMENT_ENABLE_PROPERTY, "0")

    def mode(self) -> int:
        return self._mode

    def clear_children(self) -> None:
        for node in list(self):
            if node.tag != _IMPAIRMENT_TYPE:
                self.remove(node)

    def _update_param(self):
        self.clear_children()
        self.add_children(self.impairment_params)

    def __str__(self):
        res = self.__class__.__name__ + ":{ "
        for k, v in self.impairment_params.items():
            res += r"{0}:{1} ".format(self._param_tag2str[k], v)

        res += "}"
        return res

    @staticmethod
    @abstractmethod
    def construct_from_node(node: ET.Element) -> None:
        ...
