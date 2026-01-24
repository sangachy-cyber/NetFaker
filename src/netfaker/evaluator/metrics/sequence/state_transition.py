"""状态转移评估指标。

计算真实序列与合成序列之间的状态转移概率差异，
使用JS散度评估状态转移的相似性。
"""

from typing import Dict, List, Optional

import numpy as np


class StateTransitionMetrics:
    """状态转移评估指标计算器。
    
    计算真实序列与合成序列之间的状态转移概率差异，
    使用JS散度评估状态转移的相似性。
    """

    def calculate(self, real_states: Optional[List[int]], syn_states: Optional[List[int]]) -> Dict:
        """计算状态转移指标。
        
        Args:
            real_states: 真实状态序列
            syn_states: 合成状态序列
            
        Returns:
            Dict: 包含状态转移JS散度的指标字典
        """
        if real_states is None or syn_states is None:
            return {
                "state_transition_jsd": 0.0,
                "state_transition_available": False
            }
        
        # 确保状态序列长度一致
        min_len = min(len(real_states), len(syn_states))
        real_states = real_states[:min_len]
        syn_states = syn_states[:min_len]
        
        # 计算状态转移矩阵
        real_transitions = self._calculate_transition_matrix(real_states)
        syn_transitions = self._calculate_transition_matrix(syn_states)
        
        # 计算JS散度
        jsd = self._calculate_js_divergence(real_transitions, syn_transitions)
        
        return {
            "state_transition_jsd": jsd,
            "state_transition_available": True
        }
    
    def _calculate_transition_matrix(self, states: List[int]) -> Dict[tuple, float]:
        """计算状态转移矩阵。
        
        Args:
            states: 状态序列
            
        Returns:
            Dict[tuple, float]: 状态转移概率字典，键为(from_state, to_state)，值为概率
        """
        transitions = {}
        state_counts = {}
        
        # 统计状态转移次数
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            
            # 更新状态计数
            state_counts[from_state] = state_counts.get(from_state, 0) + 1
            
            # 更新转移计数
            key = (from_state, to_state)
            transitions[key] = transitions.get(key, 0) + 1
        
        # 计算转移概率
        transition_probs = {}
        for (from_state, to_state), count in transitions.items():
            transition_probs[(from_state, to_state)] = count / state_counts[from_state]
        
        return transition_probs
    
    def _calculate_js_divergence(self, p: Dict[tuple, float], q: Dict[tuple, float]) -> float:
        """计算两个状态转移矩阵之间的JS散度。
        
        Args:
            p: 第一个转移概率字典
            q: 第二个转移概率字典
            
        Returns:
            float: JS散度值，范围[0, 1]
        """
        # 获取所有可能的状态转移
        all_transitions = set(p.keys()).union(set(q.keys()))
        
        # 平滑参数，避免零概率
        epsilon = 1e-10
        
        jsd = 0.0
        
        for transition in all_transitions:
            p_prob = p.get(transition, 0.0) + epsilon
            q_prob = q.get(transition, 0.0) + epsilon
            
            # 计算混合分布
            m_prob = 0.5 * (p_prob + q_prob)
            
            # 计算KL散度
            kl_pm = p_prob * np.log(p_prob / m_prob)
            kl_qm = q_prob * np.log(q_prob / m_prob)
            
            # 累加JS散度
            jsd += 0.5 * (kl_pm + kl_qm)
        
        return jsd
