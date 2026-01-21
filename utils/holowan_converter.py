import numpy as np
import os
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class HoloWANDataPoint:
    """
    HoloWAN文件中的单个数据点
    """
    delay1: float = 0.0  # 上行延迟 (ms)
    loss1: float = 0.0    # 上行丢包率 (%)
    bw1: float = 0.0      # 上行带宽 (Mbps)
    delay2: float = 0.0   # 下行延迟 (ms)
    loss2: float = 0.0    # 下行丢包率 (%)
    bw2: float = 0.0      # 下行带宽 (Mbps)
    
    def to_list(self) -> list:
        """
        转换为列表格式
        
        返回:
            list: [delay1, loss1, bw1, delay2, loss2, bw2]
        """
        return [self.delay1, self.loss1, self.bw1, self.delay2, self.loss2, self.bw2]


@dataclass
class HoloWANWindow:
    """
    HoloWAN文件中的窗口数据
    """
    data_points: list[HoloWANDataPoint] = field(default_factory=list)  # 窗口中的数据点列表
    
    def add_data_point(self, data_point: HoloWANDataPoint) -> None:
        """
        添加数据点到窗口
        
        参数:
            data_point: 要添加的数据点
        """
        self.data_points.append(data_point)
    
    def get_all_data(self) -> list[list]:
        """
        获取窗口中所有数据点的列表格式
        
        返回:
            list[list]: 所有数据点的列表
        """
        return [dp.to_list() for dp in self.data_points]


@dataclass
class HoloWANFile:
    """
    HoloWAN Recorder File 生成器类
    提供面向对象的方式生成和管理HoloWAN文件
    """
    # 基本属性
    operator: str = "Generated"
    network_type: str = "Unknown"
    signal_strength: int = -100
    test_name: str = field(default_factory=lambda: f"generated_{datetime.now().strftime('%y%m%d_%H%M%S')}")
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
    start_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    end_time: str = None
    loss_average: float = 0.0
    
    # 非dataclass字段，不会被自动初始化
    data: list[HoloWANDataPoint] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """
        初始化后处理，用于处理额外的初始化逻辑
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
    
    def _get_switch_str(self):
        """
        获取开关配置的字符串表示
        
        返回:
            switch_str: 开关配置字符串，格式为"1,1,0,1,1,0"
        """
        return f"{int(self.switch_delay1)},{int(self.switch_loss1)},{int(self.switch_bw1)},{int(self.switch_delay2)},{int(self.switch_loss2)},{int(self.switch_bw2)}"
    
    def _set_header_field(self, field_name, value):
        """
        设置或修改文件头字段
        
        参数:
            field_name: 字段名称
            value: 字段值
        """
        if hasattr(self, field_name):
            setattr(self, field_name, value)
    
    def _add_data(self, ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw):
        """
        添加单个数据点
        
        参数:
            ul_delay: 上行延迟 (ms)
            ul_loss: 上行丢包率 (%)
            ul_bw: 上行带宽 (Mbps)
            dl_delay: 下行延迟 (ms)
            dl_loss: 下行丢包率 (%)
            dl_bw: 下行带宽 (Mbps)
        """
        data_point = HoloWANDataPoint(
            delay1=ul_delay,
            loss1=ul_loss,
            bw1=ul_bw,
            delay2=dl_delay,
            loss2=dl_loss,
            bw2=dl_bw
        )
        self.data.append(data_point)
    
    def _add_data_point(self, data_point: HoloWANDataPoint):
        """
        添加单个数据点对象
        
        参数:
            data_point: HoloWANDataPoint对象
        """
        self.data.append(data_point)
    
    def _add_window(self, window: HoloWANWindow):
        """
        添加窗口数据
        
        参数:
            window: HoloWANWindow对象
        """
        self.data.extend(window.data_points)
    
    def _add_window_data(self, windows):
        """
        从窗口数据批量添加数据点
        
        参数:
            windows: (N, T, 6) numpy数组，包含窗口数据 [delay1, loss1, bw1, delay2, loss2, bw2]
        """
        for window in windows:
            for step in window:
                # 直接使用传入的数据，不进行额外处理
                self._add_data(*step)
    
    def _calculate_statistics(self):
        """
        计算文件统计信息
        """
        if not self.data:
            return
        
        # 计算平均丢包率
        avg_loss = np.mean([(data_point.loss1 + data_point.loss2) / 2 for data_point in self.data])
        self.loss_average = avg_loss
        
        # 计算总时长并更新结束时间
        total_time_sec = len(self.data) * self.interval_sec
        start_dt = datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + pd.Timedelta(seconds=total_time_sec)
        self.end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @classmethod
    def from_file(cls, filepath):
        """
        从HoloWAN文件读取数据，创建HoloWANFile实例
        
        参数:
            filepath: HoloWAN文件路径
        
        返回:
            HoloWANFile: HoloWANFile实例，包含读取的数据
        """
        # 解析文件头和数据
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
                
                # 读取文件头信息
                if not reading_data:
                    # 解析运营商、网络类型和信号强度
                    if stripped.startswith("Operator:"):
                        parts = stripped.split()
                        if len(parts) >= 5:
                            header["operator"] = parts[1].strip('"')
                            header["network_type"] = parts[3].strip('"')
                            header["signal_strength"] = int(parts[5].strip('(dbm)'))
                    # 解析测试名称
                    elif stripped.startswith("test_name:"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["test_name"] = parts[1].strip().strip('"')
                    # 解析目标地址
                    elif stripped.startswith("Destination:"):
                        parts = stripped.split(":", 1)
                        if len(parts) > 1:
                            header["destination"] = parts[1].strip().strip('"')
                    # 解析开始时间
                    elif stripped.startswith("Start Time:"):
                        parts = stripped.split(":", 1)
                        if len(parts) > 1:
                            header["start_time"] = parts[1].strip()
                    # 解析结束时间
                    elif stripped.startswith("End Time:"):
                        parts = stripped.split(":", 1)
                        if len(parts) > 1:
                            header["end_time"] = parts[1].strip()
                    # 解析采样间隔
                    elif stripped.startswith("Interval(sec):"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["interval_sec"] = float(parts[1].strip())
                    # 解析数据包大小
                    elif stripped.startswith("Packet Size(byte):"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["packet_size"] = int(parts[1].strip())
                    # 解析平均丢包率
                    elif stripped.startswith("Loss Average:"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["loss_average"] = float(parts[1].strip())
                    # 解析是否启用重排序
                    elif stripped.startswith("Enable Reordering:"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["enable_reordering"] = parts[1].strip().lower() == "true"
                    # 解析开关配置
                    elif stripped.startswith("Switch:"):
                        parts = stripped.split(":")
                        if len(parts) > 1:
                            header["switch"] = parts[1].strip()
                
                # 读取数据
                else:
                    parts = stripped.split(",")
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
                                bw2=bw2
                            )
                            data_points.append(data_point)
                        except (ValueError, IndexError):
                            continue  # 跳过格式错误行
        
        # 创建HoloWANFile实例
        holowan_file = cls()
        
        # 设置文件头信息
        for key, value in header.items():
            if hasattr(holowan_file, key):
                setattr(holowan_file, key, value)
        
        # 手动解析switch字符串并设置各个开关变量
        if 'switch' in header:
            switch_values = list(map(int, header['switch'].split(',')))
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
        """
        获取所有数据点，返回HoloWANDataPoint数组
        
        返回:
            list[HoloWANDataPoint]: HoloWANDataPoint数组
        """
        return self.data.copy()
    
    def write_to_file(self, output_path):
        """
        生成HoloWAN文件
        
        参数:
            output_path: 输出文件路径
        
        返回:
            output_path: 生成的文件路径
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
            self.switch_bw2
        ]
        
        # 准备文件内容，避免最后一行空行
        file_content = []
        
        # 写入文件头
        file_content.append("HoloWAN Recorder File (www.msytest.com)")
        
        # 写入元信息
        file_content.append(f"Operator: \"{self.operator}\" NetworkType: \"{self.network_type}\" SignalStrength: {self.signal_strength}(dbm)")
        file_content.append(f"test_name: \"{self.test_name}\"")
        file_content.append(f"Destination: \"{self.destination}\"")
        file_content.append(f"Start Time: {self.start_time}")
        file_content.append(f"End Time: {self.end_time}")
        file_content.append(f"Interval(sec): {self.interval_sec}")
        file_content.append(f"Packet Size(byte): {self.packet_size}")
        file_content.append(f"Loss Average: {self.loss_average:.2f}")
        file_content.append(f"Enable Reordering: {self.enable_reordering}")
        file_content.append("Contents: Delay1(ms),Loss1(%),Bandwidth1(Mbps),Delay2(ms),Loss2(%),Bandwidth2(Mbps)")
        file_content.append(f"Switch: {self._get_switch_str()}")
        
        # 写入分隔符
        file_content.append("------------------------------------------------")
        
        # 写入数据，根据开关过滤
        for data_point in self.data:
            # 根据开关值过滤数据
            filtered_data = []
            # 获取数据点的所有属性值
            data_values = data_point.to_list()
            for i, (value, switch) in enumerate(zip(data_values, switch_values)):
                # 如果开关为False，将对应字段值设为0
                filtered_data.append(value if switch else 0.0)
            
            ul_delay, ul_loss, ul_bw, dl_delay, dl_loss, dl_bw = filtered_data
            file_content.append(f"{ul_delay:.2f},{ul_loss:.2f},{ul_bw:.6f},{dl_delay:.2f},{dl_loss:.2f},{dl_bw:.6f}")
        
        # 写入文件，使用join避免最后一行空行
        with open(output_path, 'w') as f:
            f.write('\n'.join(file_content))
        
        print(f"✅ HoloWAN文件已生成: {output_path}")
        return output_path


def windows_to_holowan_file(
    windows, 
    output_path, 
    operator="Generated",
    network_type="Unknown",
    signal_strength=-100,
    test_name=None,
    destination="127.0.0.1:8080",
    interval_sec=0.1,
    packet_size=500,
    enable_reordering=True,
    **kwargs
):
    """
    将窗口数据转换为HoloWAN Recorder File格式
    
    参数:
        windows: (N, T, 4) numpy数组，包含窗口数据 [UL_delay_log, UL_loss, DL_delay_log, DL_loss]
        output_path: 输出文件路径
        operator: 运营商信息
        network_type: 网络类型
        signal_strength: 信号强度 (dbm)
        test_name: 测试名称，默认使用当前时间
        destination: 测试目标地址
        interval_sec: 采样间隔 (秒)
        packet_size: 数据包大小 (字节)
        enable_reordering: 是否启用重排序
        **kwargs: 其他文件头参数
    """
    # 创建HoloWANFile实例
    holowan_file = HoloWANFile(
        operator=operator,
        network_type=network_type,
        signal_strength=signal_strength,
        test_name=test_name,
        destination=destination,
        interval_sec=interval_sec,
        packet_size=packet_size,
        enable_reordering=enable_reordering
    )
    
    # 处理窗口数据，转换为[delay1, loss1, bw1, delay2, loss2, bw2]格式
    processed_windows = []
    for window in windows:
        processed_window = []
        for step in window:
            ul_delay_log, ul_loss, dl_delay_log, dl_loss = step
            
            # 将对数延迟转换为实际延迟 (ms)
            ul_delay = np.exp(ul_delay_log) - 1.0
            dl_delay = np.exp(dl_delay_log) - 1.0
            
            # 将丢包率从比例转换为百分比
            ul_loss_pct = ul_loss * 100.0
            dl_loss_pct = dl_loss * 100.0
            
            # 带宽字段设为0.0（可根据实际需求调整）
            ul_bw = 0.0
            dl_bw = 0.0
            
            processed_window.append([ul_delay, ul_loss_pct, ul_bw, dl_delay, dl_loss_pct, dl_bw])
        processed_windows.append(processed_window)
    
    # 添加处理后的窗口数据
    holowan_file._add_window_data(processed_windows)
    
    # 写入文件
    return holowan_file.write_to_file(output_path)


def save_clustered_windows_to_holowan(
    windows, 
    labels, 
    output_dir="holowan_files",
    **kwargs
):
    """
    将不同聚类标签的窗口保存为单独的HoloWAN文件
    
    参数:
        windows: (N, T, 4) numpy数组，包含窗口数据
        labels: 长度为N的列表，包含每个窗口的聚类标签
        output_dir: 输出目录
        **kwargs: 传递给windows_to_holowan_file的其他参数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 按标签分组窗口
    label_groups = {}
    for i, label in enumerate(labels):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(windows[i])
    
    # 为每个标签生成HoloWAN文件
    output_files = []
    for label, group_windows in label_groups.items():
        # 将列表转换为numpy数组
        group_windows_np = np.array(group_windows)
        
        # 生成输出路径
        output_path = os.path.join(output_dir, f"holowan_{label}.txt")
        
        # 生成HoloWAN文件
        file_path = windows_to_holowan_file(
            group_windows_np, 
            output_path, 
            test_name=f"{label}_{datetime.now().strftime('%y%m%d_%H%M%S')}",
            **kwargs
        )
        output_files.append(file_path)
    
    return output_files
