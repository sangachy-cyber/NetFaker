#!/usr/bin/env python3

import json
import numpy as np

# 设置要分析的行为ID
behavior_id = 0

# 初始化统计变量
max_delay_up = 0.0
max_delay_down = 0.0
count = 0
all_delays_up = []
all_delays_down = []

# 读取训练数据
with open('output/datasets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('keep', True):
            # 检查行为ID
            cond = data['cond']
            actual_behavior_id = int(cond[11])
            
            if actual_behavior_id == behavior_id:
                count += 1
                window = data['window']
                
                # 提取原始延迟值
                delays_up = [row['delay_up_origin'] for row in window]
                delays_down = [row['delay_down_origin'] for row in window]
                
                all_delays_up.extend(delays_up)
                all_delays_down.extend(delays_down)
                
                # 更新最大值
                current_max_up = max(delays_up)
                current_max_down = max(delays_down)
                
                if current_max_up > max_delay_up:
                    max_delay_up = current_max_up
                if current_max_down > max_delay_down:
                    max_delay_down = current_max_down

# 计算统计信息
if all_delays_up:
    print(f'行为ID {behavior_id} (稳定状态) 分析结果:')
    print(f'训练样本数: {count}')
    print(f'上行延迟统计:')
    print(f'  最大值: {max_delay_up:.2f} ms')
    print(f'  最小值: {min(all_delays_up):.2f} ms')
    print(f'  平均值: {np.mean(all_delays_up):.2f} ms')
    print(f'  标准差: {np.std(all_delays_up):.2f} ms')
    print(f'  P95值: {np.percentile(all_delays_up, 95):.2f} ms')
    print(f'  P99值: {np.percentile(all_delays_up, 99):.2f} ms')
    print(f'下行延迟统计:')
    print(f'  最大值: {max_delay_down:.2f} ms')
    print(f'  最小值: {min(all_delays_down):.2f} ms')
    print(f'  平均值: {np.mean(all_delays_down):.2f} ms')
    print(f'  标准差: {np.std(all_delays_down):.2f} ms')
    print(f'  P95值: {np.percentile(all_delays_down, 95):.2f} ms')
    print(f'  P99值: {np.percentile(all_delays_down, 99):.2f} ms')
else:
    print(f'未找到行为ID {behavior_id} 的训练样本')
