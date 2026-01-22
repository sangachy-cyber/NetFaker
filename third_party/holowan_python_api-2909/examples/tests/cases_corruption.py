import unittest

from settings import *


class ExampleCorruption(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_corruption_BER(self):
        """测试比特错误率参数"""
        from holowan.v2.engine.path import BER

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.corruption = BER(error_rate=1, error_rate_index=14, crc=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_corruption_BERRange(self):
        """测试比特错误率范围参数"""
        from holowan.v2.engine.path import BERRange

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.corruption = BERRange(bit_error_rate=1, bit_error_rate_index=14, crc=1,
                                        range_list=["1-10", "11-20", "21-30"])
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_corruption_BERPacket(self):
        """测试数据包比特错误率参数"""
        from holowan.v2.engine.path import BERPacket

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.corruption = BERPacket(ber_per_packet=10, probability=5, crc=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_corruption_BERCount(self):
        """测试计数误码参数"""
        from holowan.v2.engine.path import BERCount

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.corruption = BERCount(ber_per_packet=10, start_offset=100, ber_count=6,crc=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
