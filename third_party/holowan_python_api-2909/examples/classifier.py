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
holowan_ip = "192.168.1.111"
holowan_port = "8080"
engine_id = 3

# Get the engine
engine1 = Engine(holowan_ip, holowan_port, engine_id)

# Reset packet classifier
# Due to the fact that the packet classifier in the initial state (reset engine state) of the Holowan engine has a default MACRule(src="any", dst="any", type="any", action=1) rule, reset the packet classifier(PacketClassifier.reset()) first when applying rules if necessary.
# To avoid the influence of the default rules, reset before using the packet classifier.
engine1.packet_classifier.reset()
result = engine1.apply_classifier_changes()
print(result)

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
print(engine1.apply_classifier_changes())

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
engine1.packet_classifier.insert_rule(port=1,index=0,rule=TCPRule(src=[123, "456", "578-999"], dst=["12", 56, "589-999", "any"], check_version=0, action=1))
print(engine1.apply_classifier_changes())
