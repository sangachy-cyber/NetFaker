import unittest

from settings import *


class ExampleModify(unittest.TestCase):

    def setUp(self):
        from holowan.v2.engine import Engine
        self.holowan_ip = HOLOWAN_IP
        self.holowan_port = HOLOWAN_PORT
        self.engine_id = ENGINE_ID
        self.engine = Engine(self.holowan_ip, self.holowan_port, self.engine_id)
        self.path_id = PATH_ID
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.reset()

    def test_modify_disable(self):
        """测试禁用修改功能"""
        from holowan.v2.engine.path import ModifyDisable

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.modify = ModifyDisable()
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_random1(self):
        """测试随机修改参数方式一"""
        from holowan.v2.engine.path import ModifyRandom

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.modify = ModifyRandom(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                        modify_header=1,
                                        modify_offset=0, modify_size=1, modify_value="0xF",
                                        random_rate=0.01,
                                        crc=1, checksum=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_random2(self):
        """测试随机修改参数方式二"""
        from holowan.v2.engine.path import ModifyRandom

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        modify_params = {
            "modify_header": 1, "modify_offset": 0, "modify_size": 1, "modify_value": "0xF", "random_rate": 0.1,
        }
        random = ModifyRandom(match_params=match_params, modify_params=modify_params, crc=1, checksum=1)

        self.path.l2r.modify = random
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_cycle1(self):
        """测试周期修改参数方式一"""
        from holowan.v2.engine.path import ModifyCycle

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        self.path.l2r.modify = ModifyCycle(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                       modify_header=1, modify_offset=0, modify_size=1, modify_value="0xF",
                                       cycle_period=1000, cycle_burst=100,crc=1, checksum=1)
        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_ranges1(self):
        """测试范围修改参数方式一"""
        from holowan.v2.engine.path import ModifyRanges

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        modify_ranges = [
            {"offset": 0, "value": "ff"},
            {"offset": 1, "value": "ee"}
        ]
        modify_rgs = ModifyRanges(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                            modify_ranges=modify_ranges, crc=1, checksum=1)
        print(modify_rgs)
        self.path.l2r.modify = modify_rgs

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_ranges2(self):
        """测试范围修改参数方式二"""
        from holowan.v2.engine.path import ModifyRanges

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        modify_ranges = [
            {"offset": 0, "value": "ff"},
            {"offset": 1, "value": "ee"}
        ]
        modify_rgs = ModifyRanges(match_params=match_params,modify_ranges=modify_ranges, crc=1, checksum=1)
        print(modify_rgs)
        self.path.l2r.modify = modify_rgs

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_insert1(self):
        """测试插入修改参数方式一"""
        from holowan.v2.engine.path import ModifyInsert

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        modify_ins = ModifyInsert(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                  insert_offset=1, insert_value="ff", crc=1, checksum=1)
        print(modify_ins)
        self.path.l2r.modify = modify_ins

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_insert2(self):
        """测试插入修改参数方式二"""
        from holowan.v2.engine.path import ModifyInsert

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        insert_params = {
            "insert_offset": 10, "insert_value": "ff"
        }
        modify_ins = ModifyInsert(match_params=match_params, insert_params=insert_params, crc=1, checksum=1)
        print(modify_ins)
        self.path.l2r.modify = modify_ins

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_delete1(self):
        """测试删除修改参数方式一"""
        from holowan.v2.engine.path import ModifyDelete

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        modify_del = ModifyDelete(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                  delete_offset=10, delete_size=5, crc=1, checksum=1)
        print(modify_del)
        self.path.l2r.modify = modify_del

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_delete2(self):
        """测试删除修改参数方式二"""
        from holowan.v2.engine.path import ModifyDelete

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        delete_params = {
            "delete_offset": 20, "delete_size": 8
        }
        modify_del = ModifyDelete(match_params=match_params, delete_params=delete_params, crc=1, checksum=1)
        print(modify_del)
        self.path.l2r.modify = modify_del

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_exchange1(self):
        """测试交换修改参数方式一"""
        from holowan.v2.engine.path import ModifyExchange

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        modify_ex = ModifyExchange(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                   previous_offset=12, previous_size=21, next_offset=34, next_size=43, crc=1, checksum=1)
        print(modify_ex)
        self.path.l2r.modify = modify_ex

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_exchange2(self):
        """测试交换修改参数方式二"""
        from holowan.v2.engine.path import ModifyExchange

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        exchange_params = {
            "previous_offset": 10, "previous_size": 10, "next_offset": 30, "next_size": 53
        }
        modify_ex = ModifyExchange(match_params=match_params, exchange_params=exchange_params, crc=1, checksum=1)
        print(modify_ex)
        self.path.l2r.modify = modify_ex

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_count1(self):
        """测试计数修改参数方式一"""
        from holowan.v2.engine.path import ModifyCount

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        modify_cnt = ModifyCount(match_header=1, match_offset=0, match_size=1, match_value="0x0",
                                 offset=12, value="ff", modify_offset=34, modify_count=43, crc=1, checksum=1)
        print(modify_cnt)
        self.path.l2r.modify = modify_cnt

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)

    def test_modify_count2(self):
        """测试计数修改参数方式二"""
        from holowan.v2.engine.path import ModifyCount

        # 测试正常参数范围
        self.path = self.engine.get_path_by_id(self.path_id)
        match_params = {
            "match_header": 1, "match_offset": 0, "match_size": 1, "match_value": "0x0",
        }
        count_params = {
            "offset": 22, "value": "ee", "modify_offset": 44, "modify_count": 53
        }
        modify_cnt = ModifyCount(match_params=match_params, count_params=count_params, crc=1, checksum=1)
        print(modify_cnt)
        self.path.l2r.modify = modify_cnt

        result = self.engine.apply_path_configuration(self.path)
        print(result)  # 打印result值
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
