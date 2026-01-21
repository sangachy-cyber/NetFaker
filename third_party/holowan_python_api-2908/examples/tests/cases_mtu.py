import unittest

from settings import *


class ExampleMTU(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_mtu_disable(self):
        """测试禁用MTU功能"""
        from holowan.v2.engine.path import MTUDisable

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.mtu = MTUDisable()
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_mtu(self):
        """测试MTU限制参数"""
        from holowan.v2.engine.path import MTULimit

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.mtu = MTULimit(limit=1499, drop_df_packets=1, drop_oversize_packets=0)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
