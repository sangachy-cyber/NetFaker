import unittest

from settings import *


class ExampleDelay(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_delay_constant(self):
        """测试固定延迟参数"""
        from holowan.v2.engine.path import DelayConstant

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.delay = DelayConstant(delay=10)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_delay_jitter(self):
        """测试延迟抖动参数"""
        from holowan.v2.engine.path import DelayJitter

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.delay = DelayJitter(delay=99.9, jitter=51.1, type=1, enable_reordering=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_delay_normal(self):
        """测试正态分布延迟参数"""
        from holowan.v2.engine.path import DelayNormal

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        normal = DelayNormal(minimum=1.11, mean=55.55, std_deviation=11.11, enable_reordering=1)
        normal.enable_advanced_setup(period=50, min=900.1, max=1000, duration=2)
        self.path.l2r.delay = normal
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_delay_gamma(self):
        """测试伽马分布延迟参数"""
        from holowan.v2.engine.path import DelayGamma

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.delay = DelayGamma(shape=1.1, scale=2.1, enable_reordering=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_delay_customized_file1(self):
        """测试自定义文件延迟参数"""
        from holowan.v2.engine.path import DelayCustomizedFile

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.delay = DelayCustomizedFile(filename="1.txt", type=1, interval=100)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_delay_customized_file2(self):
        """测试自定义文件延迟参数"""
        from holowan.v2.engine.path import DelayCustomizedFile

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.delay = DelayCustomizedFile(filename="1.txt", type=0, interval=100)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
