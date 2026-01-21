"""HoloWAN Recorder File 生成与解析工具。

提供面向对象的方式生成、解析和管理HoloWAN文件，支持从原始数据生成HoloWAN文件，
以及从现有HoloWAN文件读取数据进行分析。
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class HoloWANDataPoint:
    """HoloWAN文件中的单个数据点。

    存储单个时间点的网络参数数据，包括上行和下行的延迟、丢包率和带宽。

    Attributes:
        delay1: 上行延迟 (ms)
        loss1: 上行丢包率 (%)
        bw1: 上行带宽 (Mbps)
        delay2: 下行延迟 (ms)
        loss2: 下行丢包率 (%)
        bw2: 下行带宽 (Mbps)
    """

    delay1: float = 0.0  # 上行延迟 (ms)
    loss1: float = 0.0  # 上行丢包率 (%)
    bw1: float = 0.0  # 上行带宽 (Mbps)
    delay2: float = 0.0  # 下行延迟 (ms)
    loss2: float = 0.0  # 下行丢包率 (%)
    bw2: float = 0.0  # 下行带宽 (Mbps)

    def to_list(self) -> list:
        """转换为列表格式。

        Returns:
            list: [delay1, loss1, bw1, delay2, loss2, bw2]

        Examples:
            >>> data_point = HoloWANDataPoint(delay1=100.0, loss1=0.1, bw1=10.0, delay2=95.0, loss2=0.0, bw2=12.0)
            >>> data_point.to_list()
            [100.0, 0.1, 10.0, 95.0, 0.0, 12.0]
        """
        return [self.delay1, self.loss1, self.bw1, self.delay2, self.loss2, self.bw2]


@dataclass
class HoloWANWindow:
    """HoloWAN文件中的窗口数据。

    存储一段时间窗口内的多个数据点。

    Attributes:
        data_points: 窗口中的数据点列表
    """

    data_points: list[HoloWANDataPoint] = field(
        default_factory=list
    )  # 窗口中的数据点列表

    def add_data_point(self, data_point: HoloWANDataPoint) -> None:
        """添加数据点到窗口。

        Args:
            data_point: 要添加的数据点

        Examples:
            >>> window = HoloWANWindow()
            >>> data_point = HoloWANDataPoint(delay1=100.0, loss1=0.1)
            >>> window.add_data_point(data_point)
            >>> len(window.data_points)
            1
        """
        self.data_points.append(data_point)

    def get_all_data(self) -> list[list]:
        """获取窗口中所有数据点的列表格式。

        Returns:
            list[list]: 所有数据点的列表，每个数据点为一个包含6个元素的列表

        Examples:
            >>> window = HoloWANWindow()
            >>> window.add_data_point(HoloWANDataPoint(delay1=100.0, loss1=0.1))
            >>> window.add_data_point(HoloWANDataPoint(delay1=105.0, loss1=0.0))
            >>> window.get_all_data()
            [[100.0, 0.1, 0.0, 0.0, 0.0, 0.0], [105.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        """
        return [dp.to_list() for dp in self.data_points]


@dataclass
class HoloWANFile:
    """HoloWAN Recorder File 生成器类。

    提供面向对象的方式生成和管理HoloWAN文件，支持从原始数据生成HoloWAN文件，
    以及从现有HoloWAN文件读取数据进行分析。

    Attributes:
        operator: 运营商信息
        network_type: 网络类型
        signal_strength: 信号强度 (dbm)
        test_name: 测试名称
        destination: 测试目标地址
        interval_sec: 采样间隔 (秒)
        packet_size: 数据包大小 (字节)
        enable_reordering: 是否启用重排序
        switch: 开关配置字符串，格式为"1,1,0,1,1,0"
        switch_delay1: 上行延迟开关
        switch_loss1: 上行丢包率开关
        switch_bw1: 上行带宽开关
        switch_delay2: 下行延迟开关
        switch_loss2: 下行丢包率开关
        switch_bw2: 下行带宽开关
        start_time: 开始时间
        end_time: 结束时间
        loss_average: 平均丢包率
        data: 数据点列表
    """

    # 基本属性
    operator: str = "Generated"
    network_type: str = "CoreLab"
    signal_strength: int = -100
    test_name: str = field(
        default_factory=lambda: f"generated_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    )
    destination: str = "127.0.0.1:8080"
    interval_sec: float = 0.1
    packet_size: int = 500
    enable_reordering: bool = True

    # 开关配置字符串
    switch: str = "1,1,0,1,1,0"

    # 开关属性，单独拆分为6个变量
    switch_delay1: bool = True
    switch_loss1: bool = True
    switch_bw1: bool = False
    switch_delay2: bool = True
    switch_loss2: bool = True
    switch_bw2: bool = False

    # 动态生成的属性
    start_time: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    end_time: str = None
    loss_average: float = 0.0

    # 非dataclass字段，不会被自动初始化
    data: list[HoloWANDataPoint] = field(default_factory=list, init=False)

    def __post_init__(self):
        """初始化后处理，用于处理额外的初始化逻辑。

        处理测试名称、初始化动态属性、初始化数据列表，并解析开关字符串。
        """
        # 处理测试名称
        if self.test_name is None:
            self.test_name = f"generated_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        # 初始化动态属性
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.end_time = None
        self.loss_average = 0.0

        # 初始化数据列表
        self.data = []

        # 解析switch字符串并设置各个开关变量
        switch_values = list(map(int, self.switch.split(",")))
        self.switch_delay1 = bool(switch_values[0])
        self.switch_loss1 = bool(switch_values[1])
        self.switch_bw1 = bool(switch_values[2])
        self.switch_delay2 = bool(switch_values[3])
        self.switch_loss2 = bool(switch_values[4])
        self.switch_bw2 = bool(switch_values[5])

    def _get_switch_str(self) -> str:
        """获取开关配置的字符串表示。

        Returns:
            str: 开关配置字符串，格式为"1,1,0,1,1,0"

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> holowan_file.switch_delay1 = True
            >>> holowan_file.switch_loss1 = False
            >>> holowan_file._get_switch_str()
            '1,0,0,1,1,0'
        """
        return f"{int(self.switch_delay1)},{int(self.switch_loss1)},{int(self.switch_bw1)},{int(self.switch_delay2)},{int(self.switch_loss2)},{int(self.switch_bw2)}"

    def _set_header_field(self, field_name: str, value) -> None:
        """设置或修改文件头字段。

        Args:
            field_name: 字段名称
            value: 字段值

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> holowan_file._set_header_field("operator", "CustomOperator")
            >>> holowan_file.operator
            'CustomOperator'
        """
        if hasattr(self, field_name):
            setattr(self, field_name, value)

    def _add_data(self, ul_delay: float, ul_loss: float, ul_bw: float,
                  dl_delay: float, dl_loss: float, dl_bw: float) -> None:
        """添加单个数据点。

        Args:
            ul_delay: 上行延迟 (ms)
            ul_loss: 上行丢包率 (%)
            ul_bw: 上行带宽 (Mbps)
            dl_delay: 下行延迟 (ms)
            dl_loss: 下行丢包率 (%)
            dl_bw: 下行带宽 (Mbps)

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
            >>> len(holowan_file.data)
            1
            >>> holowan_file.data[0].delay1
            100.0
        """
        data_point = HoloWANDataPoint(
            delay1=ul_delay,
            loss1=ul_loss,
            bw1=ul_bw,
            delay2=dl_delay,
            loss2=dl_loss,
            bw2=dl_bw,
        )
        self.data.append(data_point)

    def _add_data_point(self, data_point: HoloWANDataPoint) -> None:
        """添加单个数据点对象。

        Args:
            data_point: HoloWANDataPoint对象

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> data_point = HoloWANDataPoint(delay1=100.0, loss1=0.1)
            >>> holowan_file._add_data_point(data_point)
            >>> len(holowan_file.data)
            1
        """
        self.data.append(data_point)

    def _add_window(self, window: HoloWANWindow) -> None:
        """添加窗口数据。

        Args:
            window: HoloWANWindow对象

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> window = HoloWANWindow()
            >>> window.add_data_point(HoloWANDataPoint(delay1=100.0))
            >>> window.add_data_point(HoloWANDataPoint(delay1=105.0))
            >>> holowan_file._add_window(window)
            >>> len(holowan_file.data)
            2
        """
        self.data.extend(window.data_points)

    def _add_window_data(self, windows: np.ndarray) -> None:
        """从窗口数据批量添加数据点。

        Args:
            windows: (N, T, 6) numpy数组，包含窗口数据 [delay1, loss1, bw1, delay2, loss2, bw2]

        Examples:
            >>> import numpy as np
            >>> holowan_file = HoloWANFile()
            >>> windows = np.array([[[100.0, 0.1, 10.0, 95.0, 0.0, 12.0]]])
            >>> holowan_file._add_window_data(windows)
            >>> len(holowan_file.data)
            1
        """
        for window in windows:
            for step in window:
                # 直接使用传入的数据，不进行额外处理
                self._add_data(*step)

    def _calculate_statistics(self):
        """计算文件统计信息。

        计算平均丢包率和结束时间。

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
            >>> holowan_file._add_data(105.0, 0.2, 9.5, 98.0, 0.1, 11.5)
            >>> holowan_file._calculate_statistics()
            >>> holowan_file.loss_average
            0.1
        """
        if not self.data:
            return

        # 计算平均丢包率
        avg_loss = np.mean(
            [(data_point.loss1 + data_point.loss2) / 2 for data_point in self.data]
        )
        self.loss_average = avg_loss

        # 计算总时长并更新结束时间
        total_time_sec = len(self.data) * self.interval_sec
        start_dt = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + pd.Timedelta(seconds=total_time_sec)
        self.end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def from_file(cls, filepath):
        """从HoloWAN文件读取数据，创建HoloWANFile实例。

        Args:
            filepath: HoloWAN文件路径

        Returns:
            HoloWANFile: HoloWANFile实例，包含读取的数据

        Examples:
            >>> # 假设有一个名为test.holowan的文件
            >>> # holowan_file = HoloWANFile.from_file("test.holowan")
            >>> # print(len(holowan_file.data))
        """
        header = {}
        data_points = []
        reading_data = False

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                # 检查是否开始读取数据
                if stripped.startswith("-"):
                    reading_data = True
                    continue

                # 解析文件行
                if not reading_data:
                    cls._parse_header_line(header, stripped)
                else:
                    cls._parse_data_line(data_points, stripped)

        # 创建并返回HoloWANFile实例
        return cls._create_holowan_file(header, data_points)

    @staticmethod
    def _parse_header_line(header, line):
        """解析文件头行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        # 使用字典映射来降低圈复杂度
        header_parsers = {
            "Operator:": HoloWANFile._parse_operator_line,
            "test_name:": HoloWANFile._parse_test_name_line,
            "Destination:": HoloWANFile._parse_destination_line,
            "Start Time:": HoloWANFile._parse_start_time_line,
            "End Time:": HoloWANFile._parse_end_time_line,
            "Interval(sec):": HoloWANFile._parse_interval_line,
            "Packet Size(byte):": HoloWANFile._parse_packet_size_line,
            "Loss Average:": HoloWANFile._parse_loss_average_line,
            "Enable Reordering:": HoloWANFile._parse_enable_reordering_line,
            "Switch:": HoloWANFile._parse_switch_line
        }

        # 查找并调用对应的解析器
        for prefix, parser in header_parsers.items():
            if line.startswith(prefix):
                parser(header, line)
                break

    @staticmethod
    def _parse_test_name_line(header, line):
        """解析测试名称行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["test_name"] = parts[1].strip().strip('"')

    @staticmethod
    def _parse_destination_line(header, line):
        """解析目标地址行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":", 1)
        if len(parts) > 1:
            header["destination"] = parts[1].strip().strip('"')

    @staticmethod
    def _parse_start_time_line(header, line):
        """解析开始时间行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":", 1)
        if len(parts) > 1:
            header["start_time"] = parts[1].strip()

    @staticmethod
    def _parse_end_time_line(header, line):
        """解析结束时间行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":", 1)
        if len(parts) > 1:
            header["end_time"] = parts[1].strip()

    @staticmethod
    def _parse_interval_line(header, line):
        """解析采样间隔行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["interval_sec"] = float(parts[1].strip())

    @staticmethod
    def _parse_packet_size_line(header, line):
        """解析数据包大小行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["packet_size"] = int(parts[1].strip())

    @staticmethod
    def _parse_loss_average_line(header, line):
        """解析平均丢包率行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["loss_average"] = float(parts[1].strip())

    @staticmethod
    def _parse_enable_reordering_line(header, line):
        """解析是否启用重排序行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["enable_reordering"] = (parts[1].strip().lower() == "true")

    @staticmethod
    def _parse_switch_line(header, line):
        """解析开关配置行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split(":")
        if len(parts) > 1:
            header["switch"] = parts[1].strip()

    @staticmethod
    def _parse_operator_line(header, line):
        """解析运营商行。

        Args:
            header: 存储解析结果的字典
            line: 要解析的行
        """
        parts = line.split()
        if len(parts) >= 5:
            header["operator"] = parts[1].strip('"')
            header["network_type"] = parts[3].strip('"')
            header["signal_strength"] = int(parts[5].strip("(dbm)"))

    @staticmethod
    def _parse_data_line(data_points, line):
        """解析数据行。

        Args:
            data_points: 存储数据点的列表
            line: 要解析的行
        """
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                # 提取数据
                delay1 = float(parts[0])
                loss1 = float(parts[1])
                bw1 = float(parts[2])
                delay2 = float(parts[3])
                loss2 = float(parts[4])
                bw2 = float(parts[5])

                # 创建数据点对象
                data_point = HoloWANDataPoint(
                    delay1=delay1,
                    loss1=loss1,
                    bw1=bw1,
                    delay2=delay2,
                    loss2=loss2,
                    bw2=bw2,
                )
                data_points.append(data_point)
            except (ValueError, IndexError):
                pass  # 跳过格式错误行

    @classmethod
    def _create_holowan_file(cls, header, data_points):
        """创建HoloWANFile实例并设置属性。

        Args:
            header: 包含文件头信息的字典
            data_points: 数据点列表

        Returns:
            HoloWANFile: 创建的HoloWANFile实例
        """
        # 创建HoloWANFile实例
        holowan_file = cls()

        # 设置文件头信息
        for key, value in header.items():
            if hasattr(holowan_file, key):
                setattr(holowan_file, key, value)

        # 手动解析switch字符串并设置各个开关变量
        if "switch" in header:
            switch_values = list(map(int, header["switch"].split(",")))
            holowan_file.switch_delay1 = bool(switch_values[0])
            holowan_file.switch_loss1 = bool(switch_values[1])
            holowan_file.switch_bw1 = bool(switch_values[2])
            holowan_file.switch_delay2 = bool(switch_values[3])
            holowan_file.switch_loss2 = bool(switch_values[4])
            holowan_file.switch_bw2 = bool(switch_values[5])

        # 添加数据点
        for data_point in data_points:
            holowan_file._add_data_point(data_point)

        return holowan_file

    def _get_data_points(self) -> list[HoloWANDataPoint]:
        """获取所有数据点，返回HoloWANDataPoint数组。

        Returns:
            list[HoloWANDataPoint]: HoloWANDataPoint数组

        Examples:
            >>> holowan_file = HoloWANFile()
            >>> holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
            >>> data_points = holowan_file._get_data_points()
            >>> len(data_points)
            1
            >>> isinstance(data_points[0], HoloWANDataPoint)
            True
        """
        return self.data.copy()

    def write_to_file(self, output_path):
        """生成HoloWAN文件。

        Args:
            output_path: 输出文件路径

        Returns:
            str: 生成的文件路径

        Examples:
            >>> import tempfile
            >>> holowan_file = HoloWANFile()
            >>> holowan_file._add_data(100.0, 0.1, 10.0, 95.0, 0.0, 12.0)
            >>> with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            ...     output_path = holowan_file.write_to_file(tmp.name)
            >>> import os
            >>> os.path.exists(output_path)
            True
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 计算统计信息
        self._calculate_statistics()

        # 获取开关列表用于数据过滤
        switch_values = [
            self.switch_delay1,
            self.switch_loss1,
            self.switch_bw1,
            self.switch_delay2,
            self.switch_loss2,
            self.switch_bw2,
        ]

        # 准备文件内容，避免最后一行空行
        file_content = []

        # 写入文件头
        file_content.append("HoloWAN Recorder File (www.msytest.com)")

        # 写入元信息
        file_content.append(
            f'Operator: "{self.operator}" NetworkType: "{self.network_type}" SignalStrength: {self.signal_strength}(dbm)'
        )
        file_content.append(f'test_name: "{self.test_name}"')
        file_content.append(f'Destination: "{self.destination}"')
        file_content.append(f"Start Time: {self.start_time}")
        file_content.append(f"End Time: {self.end_time}")
        file_content.append(f"Interval(sec): {self.interval_sec}")
        file_content.append(f"Packet Size(byte): {self.packet_size}")
        file_content.append(f"Loss Average: {self.loss_average:.2f}")
        file_content.append(f"Enable Reordering: {self.enable_reordering}")
        file_content.append(
            "Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)"
        )
        file_content.append(f"Switch: {self._get_switch_str()}")

        # 写入分隔符
        file_content.append("------------------------------------------------")

        # 写入数据，根据开关过滤
        for data_point in self.data:
            # 根据开关值过滤数据
            filtered_data = []
            # 获取数据点的所有属性值
            data_values = data_point.to_list()
            for _i, (value, switch) in enumerate(zip(data_values, switch_values, strict=True)):
                # 如果开关为False，将对应字段值设为0
                filtered_data.append(value if switch else 0.0)

            ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw = filtered_data
            file_content.append(
                f"{ul_delay:.2f},{ul_loss:.2f},{ul_bw:.6f},{dl_delay:.2f},{dl_loss:.2f},{dl_bw:.6f}"
            )

        # 写入文件，使用join避免最后一行空行
        with open(output_path, "w") as f:
            f.write("\n".join(file_content))

        print(f"✅ HoloWAN文件已生成: {output_path}")
        return output_path

