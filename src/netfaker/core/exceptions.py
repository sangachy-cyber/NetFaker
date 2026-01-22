"""异常管理模块。

定义项目中使用的自定义异常类。
"""


class NetFakerError(Exception):
    """NetFaker 基础异常类。

    所有自定义异常都应继承自此类。
    """

    def __init__(self, message: str, code: int = 500):
        """初始化异常。

        Args:
            message: 异常消息
            code: 错误代码
        """
        self.message = message
        self.code = code
        super().__init__(self.message)


class ValidationError(NetFakerError):
    """参数验证异常。

    当输入参数不符合要求时抛出。
    """

    def __init__(self, message: str):
        """初始化验证异常。

        Args:
            message: 异常消息
        """
        super().__init__(message, code=400)


class StrategyError(NetFakerError):
    """策略执行异常。

    当仿真策略执行失败时抛出。
    """

    def __init__(self, message: str):
        """初始化策略异常。

        Args:
            message: 异常消息
        """
        super().__init__(message, code=500)


class FileError(NetFakerError):
    """文件操作异常。

    当文件读写操作失败时抛出。
    """

    def __init__(self, message: str):
        """初始化文件异常。

        Args:
            message: 异常消息
        """
        super().__init__(message, code=500)


class SimulationError(NetFakerError):
    """仿真流程异常。

    当仿真流程执行失败时抛出。
    """

    def __init__(self, message: str):
        """初始化仿真异常。

        Args:
            message: 异常消息
        """
        super().__init__(message, code=500)
