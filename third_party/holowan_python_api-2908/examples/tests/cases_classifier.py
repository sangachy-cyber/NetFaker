import unittest

from settings import ENGINE_ID, HOLOWAN_IP, HOLOWAN_PORT


class ExampleClassifier(unittest.TestCase):

    def test_udp(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import UDPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置 UDP 端口号分类规则参数
        udp = UDPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)

        # 将 UDP 端口号分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(udp, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_tcp(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置 UDP 端口号分类规则参数
        tcp = TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)

        # 将 UDP 端口号分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(tcp, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_mac(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import MACRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置 UDP 端口号分类规则参数
        mac = MACRule(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any", action=1)

        # 将 UDP 端口号分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(mac, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_ipv4(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import IPv4Rule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 测试1: 设置 IPv4 地址分类规则参数 - 使用"any"
        ipv4_any = IPv4Rule(src="any", smask=32, dst="any", dmask=32, tos="any", action=1)
        result = engine1.apply_rule_to_classifier(ipv4_any, port=1)
        print(f"IPv4 any test result: {result}")
        assert result

        # 测试2: 使用具体的源IP地址
        ipv4_src = IPv4Rule(src="192.168.1.100", smask=24, dst="any", dmask=32, tos="any", action=2)
        result = engine1.apply_rule_to_classifier(ipv4_src, port=2)
        print(f"IPv4 specific src test result: {result}")
        assert result

        # 测试3: 使用具体的目标IP地址
        ipv4_dst = IPv4Rule(src="any", smask=32, dst="10.0.0.50", dmask=16, tos="any", action=3)
        result = engine1.apply_rule_to_classifier(ipv4_dst, port=3)
        print(f"IPv4 specific dst test result: {result}")
        assert result

        # 测试4: 使用具体的源和目标IP地址
        ipv4_both = IPv4Rule(src="172.16.0.10", smask=16, dst="8.8.8.8", dmask=32, tos="0x10", action=4)
        result = engine1.apply_rule_to_classifier(ipv4_both, port=4)
        print(f"IPv4 specific src and dst test result: {result}")
        assert result

    def test_ipv6(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import IPv6Rule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 测试1: 设置 IPv6 地址分类规则参数 - 使用"any"
        ipv6_any = IPv6Rule(src="any", dst="any", action=1)
        result = engine1.apply_rule_to_classifier(ipv6_any, port=1)
        print(f"IPv6 any test result: {result}")
        assert result

        # 测试2: 使用具体的源IPv6地址
        ipv6_src = IPv6Rule(src="2001:db8::100", dst="any", action=2)
        result = engine1.apply_rule_to_classifier(ipv6_src, port=2)
        print(f"IPv6 specific src test result: {result}")
        assert result

        # 测试3: 使用具体的目标IPv6地址
        ipv6_dst = IPv6Rule(src="any", dst="2001:db8::200", action=3)
        result = engine1.apply_rule_to_classifier(ipv6_dst, port=3)
        print(f"IPv6 specific dst test result: {result}")
        assert result

        # 测试4: 使用具体的源和目标IPv6地址
        ipv6_both = IPv6Rule(src="2001:db8::1", dst="2001:db8::2", action=4)
        result = engine1.apply_rule_to_classifier(ipv6_both, port=4)
        print(f"IPv6 specific src and dst test result: {result}")
        assert result

        # 测试5: 使用本地链路地址
        ipv6_link_local = IPv6Rule(src="fe80::1", dst="fe80::2", action=5)
        result = engine1.apply_rule_to_classifier(ipv6_link_local, port=5)
        print(f"IPv6 link-local test result: {result}")
        assert result

        # 测试6: 使用全局单播地址
        ipv6_global = IPv6Rule(src="2001:db8:1::100", dst="2001:db8:2::200", action=6)
        result = engine1.apply_rule_to_classifier(ipv6_global, port=6)
        print(f"IPv6 global unicast test result: {result}")
        assert result

    def test_raw_1_byte(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import RawByteRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置 Raw 1-Byte 字节偏移分类规则参数
        raw1 = RawByteRule(type=1, action=1)
        raw1.add_raw_byte(layer=2, offset=0, mask="0x0", value="0x0")
        raw1.add_raw_byte(layer=2, offset=0, mask="0x0", value="0x0")

        # 将 Raw 1-Byte 字节偏移分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(raw1, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_raw_4_byte(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import RawByteRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置 Raw 4-Byte 字节偏移分类规则参数
        raw2 = RawByteRule(type=4, action=1)
        # 设置 Raw 4-Byte 字节偏移分类规则具体参数
        raw2.add_raw_byte(layer=2, offset=0, mask="0xFF00FF00", value="0x11223344")

        # 将 Raw 4-Byte 字节偏移分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(raw2, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_vlan(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import VLANRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        vlan = VLANRule(fpcp="any", fpid="any", action=1)
        vlan.enable_second_tag(spcp="any", spid="any")
        # 将 VLAN 分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(vlan, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_mpls(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import MPLSRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        mpls = MPLSRule(label="any", action=1)
        # 将 VLAN 分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(mpls, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_sctp(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import SCTPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        sctp = SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1)
        # 将 VLAN 分类规则应用到 port1 上
        result = engine1.apply_rule_to_classifier(sctp, port=1)

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_pppoe(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import PPPoERule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        pppoe = PPPoERule(sid="any", code="any", action=1)
        result = engine1.apply_rule_to_classifier(pppoe, port=1)

        print(result)
        assert result

    def test_comb(self):
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import (
            CombinationRule,
            MACRule,
            PPPoERule,
            TCPRule,
        )

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        comb = CombinationRule(action=1)
        comb.add_rule(
            TCPRule(src=123, dst=123, check_version=0, action=1),
            MACRule(src="any", dst="any", type="any", action=1),
            PPPoERule(sid="any", code="any", action=1)
        )
        result = engine1.apply_rule_to_classifier(comb, port=1)

        print(result)
        assert result

    def test_container(self):
        from holowan.v2.engine import Engine, Sequential
        from holowan.v2.engine.classifier import MACRule, PacketClassifier, TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 设置报文分类器
        packet_classifier = PacketClassifier()
        packet_classifier.port1 = Sequential(
            TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            MACRule(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any", action=1)
        )
        packet_classifier.port2 = Sequential(
            TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            MACRule(src="39-A9-FA-D7-F5-E8", dst=r"any", type="any", action=1)
        )

        result = engine1.apply_classifier_changes()

        # 打印 result 判断是否应用成功
        print(result)
        assert result

    def test_insert_rearrange(self):
        from holowan.v2.engine import Engine, Sequential
        from holowan.v2.engine.classifier import (
            CombinationRule,
            MACRule,
            MPLSRule,
            PPPoERule,
            RawByteRule,
            SCTPRule,
            TCPRule,
            VLANRule,
        )

        # HoloWAN information
        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID

        # Get the engine
        engine1 = Engine(holowan_ip, holowan_port, engine_id)

        # Reset packet classifier
        # Due to the fact that the packet classifier in the initial state (reset engine state) of the Holowan engine has a default MACRule(src="any", dst="any", type="any", action=1) rule, reset the packet classifier(PacketClassifier.reset()) first when applying rules if necessary.
        # To avoid the influence of the default rules, reset before using the packet classifier.
        engine1.packet_classifier.reset()
        result = engine1.apply_classifier_changes()
        assert result

        # Create rules
        raw1 = RawByteRule(type=1, action=1)
        raw1.add_raw_byte(layer=2, offset=0, mask="0x0", value="0x0")
        raw1.add_raw_byte(layer=2, offset=0, mask="0x0", value="0x0")

        comb = CombinationRule(action=1)
        # Sub-rules will be added in the order of parameter input
        comb.add_rule(
            TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            MACRule(src="any", dst="any", type="any", action=1),
            PPPoERule(sid="any", code="any", action=1),
            MPLSRule(label="any", action=1)
        )

        # Set multiple classification rules at the same time
        # Use the Seqential container, rules will be placed in the container in the order of parameter input
        engine1.packet_classifier.port1 = Sequential(
            TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            SCTPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1),
            MACRule(src="any", dst="any", type="any", action=1),
            PPPoERule(sid="any", code="any", action=1)
        )

        # Use the PacketClassifier.add_rule to add a rule to the end.
        vlan = VLANRule(fpcp="any", fpid="any", action=1)
        engine1.packet_classifier.add_rule(1, vlan)

        # Apply changes to the packet classifier
        result = engine1.apply_classifier_changes()
        assert result

        # Set a classification rule (this rule will be set in an appended manner, that is, added to the end)
        # Add raw byte 1 rule to port 2
        result = engine1.apply_rule_to_classifier(raw1, port=2)
        print(result)

        # To modify an existed rule at index [index]
        engine1.packet_classifier.modify_rule_by_idx(1, 0, VLANRule(fpcp="any", fpid="any", action=1))
        result = engine1.apply_classifier_changes()
        print(result)

        # To rearrange the rules, using PacketClassifier.rearrange_rules
        engine1.packet_classifier.rearrange_rules(port=1, order=[0, 3, 1, 2])
        result = engine1.apply_classifier_changes()
        print(result)

        # To insert a rule
        engine1.packet_classifier.insert_rule(port=1, index=0, rule=TCPRule(src=[123, "456", "578-999"],
                                                                            dst=["12", 56, "589-999", "any"],
                                                                            check_version=0, action=1))
        print(engine1.apply_classifier_changes())


    def test_add_rule(self):
        """测试添加规则功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import MACRule, TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 测试添加规则到port1
        tcp_rule = TCPRule(src=[123, "456"], dst=["12", 56], check_version=0, action=1)
        engine1.packet_classifier.add_rule(1, tcp_rule)
        self.assertEqual(len(engine1.packet_classifier.port1), 1)
        result = engine1.apply_classifier_changes()
        print(result)

        # 测试添加规则到port2
        mac_rule = MACRule(src="any", dst="any", type="any", action=1)
        engine1.packet_classifier.add_rule(2, mac_rule)
        self.assertEqual(len(engine1.packet_classifier.port2), 1)
        result = engine1.apply_classifier_changes()
        print(result)

    def test_insert_rule(self):
        """测试插入规则功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import IPv4Rule, TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 先添加一个规则
        tcp_rule = TCPRule(src=[123], dst=[456], check_version=0, action=1)
        engine1.packet_classifier.add_rule(1, tcp_rule)

        # 在索引0处插入新规则
        ipv4_rule = IPv4Rule(src="any", smask=32, dst="any", dmask=32, tos="any", action=2)
        engine1.packet_classifier.insert_rule(1, 0, ipv4_rule)

        # 验证插入结果
        self.assertEqual(len(engine1.packet_classifier.port1), 2)
        self.assertIsInstance(engine1.packet_classifier.port1[0], IPv4Rule)
        result = engine1.apply_classifier_changes()
        print(result)

    def test_modify_rule(self):
        """测试修改规则功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 添加初始规则
        old_rule = TCPRule(src=[123], dst=[456], check_version=0, action=1)
        engine1.packet_classifier.add_rule(1, old_rule)

        # 修改规则
        new_rule = TCPRule(src=[789], dst=["any"], check_version=0, action=2)
        engine1.packet_classifier.modify_rule_by_idx(1, 0, new_rule)

        # 验证修改结果
        modified_rule = engine1.packet_classifier.port1[0]
        self.assertEqual(modified_rule.action, 2)
        result = engine1.apply_classifier_changes()
        print(result)

    def test_remove_rule(self):
        """测试删除规则功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 添加规则
        rule = TCPRule(src=[123], dst=[456], check_version=0, action=1)
        engine1.packet_classifier.add_rule(1, rule)

        # 删除规则
        engine1.packet_classifier.remove_rule(1, 0)

        # 验证删除结果
        self.assertEqual(len(engine1.packet_classifier.port1), 0)
        result = engine1.apply_classifier_changes()
        print(result)

    def test_rearrange_rules(self):
        """测试重排序规则功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import (
            IPv4Rule,
            MACRule,
            TCPRule,
        )

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)
        engine1.packet_classifier.reset()
        engine1.apply_classifier_changes()

        # 添加多个规则
        rules = [
            TCPRule(src=[123], dst=[456], check_version=0, action=1),
            IPv4Rule(src="any", smask=32, dst="any", dmask=32, tos="any", action=2),
            MACRule(src="any", dst="any", type="any", action=3)
        ]
        for rule in rules:
            engine1.packet_classifier.add_rule(1, rule)

        # 重排序规则
        new_order = [2, 0, 1]
        engine1.packet_classifier.rearrange_rules(1, new_order)

        # 验证重排序结果
        self.assertIsInstance(engine1.packet_classifier.port1[0], MACRule)
        self.assertIsInstance(engine1.packet_classifier.port1[1], TCPRule)
        self.assertIsInstance(engine1.packet_classifier.port1[2], IPv4Rule)
        result = engine1.apply_classifier_changes()
        print(result)

    def test_reset_classifier(self):
        """测试重置功能"""
        from holowan.v2.engine import Engine
        from holowan.v2.engine.classifier import PacketClassifier, TCPRule

        holowan_ip = HOLOWAN_IP
        holowan_port = HOLOWAN_PORT
        engine_id = ENGINE_ID
        engine1 = Engine(holowan_ip, holowan_port, engine_id)

        classifier = PacketClassifier()

        # 添加规则
        rule = TCPRule(src=[123], dst=[456], check_version=0, action=1)
        classifier.add_rule(1, rule)
        classifier.add_rule(2, rule)

        # 重置分类器
        classifier.reset()

        # 验证重置结果
        self.assertEqual(len(classifier.port1), 0)
        self.assertEqual(len(classifier.port2), 0)

        result = engine1.apply_classifier_changes()
        print(result)


if __name__ == '__main__':
    unittest.main()
