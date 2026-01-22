# Import related modules
from holowan.v2.engine import Engine
from holowan.v2.engine.path import *

# HoloWAN related information
holowan_ip = "192.168.1.111"
holowan_port = "8080"

engine_id = 3
path_id = 1

# When creating an Engine instance, the information and configuration of the corresponding engine will be obtained from the backend (State 1)
engine1 = Engine(holowan_ip, holowan_port, engine_id)
# If you need to get the latest status of the engine, please use the Engine.update() method
engine1.update()

# Reset the engine
result = engine1.reset_engine()
if not result:
    print(result.err_code, result.err_msg, result.err_reason)

# Save the engine
result = engine1.save_engine()
print(result)

# set engine name
engine1.set_engine_name(engine_id=engine_id, engine_name="new Engine 3")

# set path name
engine1.set_path_name(path_id=path_id, path_name="new PATH 3")

# apply a single impairment
# we recommend using the Methods: Engine.apply_impairment
bd_fix = BandwidthFixed(rate=100, unit=1)
result = engine1.apply_impairment(bd_fix, path_id=1, direction=1)
print(result)

# apply a group of impairments (apply multiple impairments at the same time), using Impairments.
impair1 = Impairments(1)
# Bandwidth
impair1.bandwidth = BandwidthTokenBucket(type=3,cir=1.1,cir_unit=1,cbs=2.2,cbs_unit=2,pir=3.3,pir_unit=3,pbs=4.4,pbs_unit=1)
# Background traffic
impair1.background_utilization = BackgroundUtilizationRandom(rate=5, burst=66)
# Queue depth
impair1.queue_limit = QueueLimitRED(weight=0.003, min_threshold=8, max_threshold=22, probability=0.03)
# Packet modification
impair1.modify = ModifyRandom(match_header=1, match_offset=1, match_size=1, match_value="0x0",
                              modify_header=1, modify_offset=0, modify_size=1, modify_value="0x1",
                              random_rate=0.1, crc=1, checksum=1)

# Delay
dnormal = DelayNormal(minimum=1.1, mean=55.5, std_deviation=9.9, enable_reordering=1)
dnormal.enable_advanced_setup(period=61, duration=2, min=899, max=999)
impair1.delay = dnormal

# Packet loss
impair1.loss = LossBurst(probability=5, min=10, max=19)
# Bit error
impair1.corruption = BERRange(bit_error_rate=1, bit_error_rate_index=13, crc=1, range_list=["1-10", "11-20"])
# Duplicate frame
impair1.duplication = DuplicationJitter(change_mode=ChangeMode(max=10, min=1, rise=0.29,
                                                               fall=0.19, period=59, mode=2))

result = engine1.apply_impairment(impair1, path_id=1, direction=3)
print(result)

# modify multiple impairments of a specific path at the same time
# get impairs from a specific path
path = engine1.get_path_by_id(path_id=1)
path.l2r.delay = dnormal
path.r2l.bandwidth = bd_fix
result = engine1.apply_path_configuration(path)
print(result)

# Modify the parameters of a certain impairment
# Method one (not recommended)
path1 = engine1.get_path_by_id(path_id=1)
# This method can only be used when you are sure of the specific type of impairment
# Otherwise, this method is likely to cause incorrect results
path1.l2r.delay.minimum = 1.12
result = engine1.apply_path_configuration(path1)
print(result)

# Method two (through the impairment instance object and the method of setting a single impairment)
dnormal.minimum = 1.12
result = engine1.apply_impairment(dnormal, path_id=1, direction=1)
print(result)
