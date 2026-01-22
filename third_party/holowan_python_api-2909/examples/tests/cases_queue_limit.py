import unittest

from settings import *


class ExampleQueueLimit(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_ql_simple(self):
        """测试简单队列限制参数"""
        from holowan.v2.engine.path import QueueLimitSimple

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.queue_limit = QueueLimitSimple(option=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_ql_red(self):
        """测试RED队列限制参数"""
        from holowan.v2.engine.path import QueueLimitRED

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.queue_limit = QueueLimitRED(weight=0.001, min_threshold=8, max_threshold=21, probability=0.3)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_ql_dt(self):
        """测试DropTail队列限制参数"""
        from holowan.v2.engine.path import QueueLimitDropTail

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.queue_limit = QueueLimitDropTail(depth=256, unit=2)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
