"""
异常处理测试模块。
"""

import pytest

from netfaker.core.exceptions import (
    FileError,
    NetFakerError,
    StrategyError,
    ValidationError,
)


class TestExceptions:
    """异常处理测试类。"""

    def test_netfaker_error_basic(self):
        """测试基础异常类的基本功能。"""
        # 测试默认参数
        error = NetFakerError("Test error")
        assert error.message == "Test error"
        assert error.code == 500
        assert str(error) == "Test error"

        # 测试自定义错误代码
        error = NetFakerError("Test error with code", code=404)
        assert error.message == "Test error with code"
        assert error.code == 404
        assert str(error) == "Test error with code"

        # 测试异常继承
        assert isinstance(error, Exception)
        assert isinstance(error, NetFakerError)

    def test_validation_error(self):
        """测试参数验证异常。"""
        error = ValidationError("Invalid parameter")
        assert error.message == "Invalid parameter"
        assert error.code == 400
        assert str(error) == "Invalid parameter"

        # 测试异常继承
        assert isinstance(error, NetFakerError)
        assert isinstance(error, Exception)

    def test_strategy_error(self):
        """测试策略执行异常。"""
        error = StrategyError("Strategy execution failed")
        assert error.message == "Strategy execution failed"
        assert error.code == 500
        assert str(error) == "Strategy execution failed"

        # 测试异常继承
        assert isinstance(error, NetFakerError)
        assert isinstance(error, Exception)

    def test_file_error(self):
        """测试文件操作异常。"""
        error = FileError("File not found")
        assert error.message == "File not found"
        assert error.code == 500
        assert str(error) == "File not found"

        # 测试异常继承
        assert isinstance(error, NetFakerError)
        assert isinstance(error, Exception)

    def test_exception_raising(self):
        """测试异常的抛出和捕获。"""
        # 测试基础异常
        with pytest.raises(NetFakerError, match="Test error"):
            raise NetFakerError("Test error")

        # 测试验证异常
        with pytest.raises(ValidationError, match="Invalid input"):
            raise ValidationError("Invalid input")

        # 测试策略异常
        with pytest.raises(StrategyError, match="Strategy failed"):
            raise StrategyError("Strategy failed")

        # 测试文件异常
        with pytest.raises(FileError, match="File error"):
            raise FileError("File error")

    def test_exception_chaining(self):
        """测试异常链。"""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            # 测试异常链
            with pytest.raises(NetFakerError) as excinfo:
                raise NetFakerError("Wrapped error") from e
            assert "Wrapped error" in str(excinfo.value)

    def test_exception_with_empty_message(self):
        """测试空消息异常。"""
        # 测试空消息
        error = NetFakerError("")
        assert error.message == ""
        assert error.code == 500
        assert str(error) == ""

        # 测试空消息的验证异常
        error = ValidationError("")
        assert error.message == ""
        assert error.code == 400

    def test_exception_error_code_types(self):
        """测试错误代码类型。"""
        # 测试不同类型的错误代码
        error = NetFakerError("Test", code=200)
        assert error.code == 200

        error = NetFakerError("Test", code=401)
        assert error.code == 401

        error = NetFakerError("Test", code=503)
        assert error.code == 503

    def test_exception_message_types(self):
        """测试消息类型。"""
        # 测试不同类型的消息
        error = NetFakerError("String message")
        assert error.message == "String message"

        # 测试包含特殊字符的消息
        error = NetFakerError("Message with special chars: !@#$%^&*()")
        assert "!@#$%^&*()" in error.message

        # 测试包含空格的消息
        error = NetFakerError("Message with spaces")
        assert error.message == "Message with spaces"
