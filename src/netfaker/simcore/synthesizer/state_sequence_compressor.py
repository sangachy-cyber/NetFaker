"""状态序列压缩器模块。

该模块负责将原始state_id序列压缩为结构化的segments，
用于回放、分析或再生成。
"""

from typing import List, Dict, Any

from loguru import logger


class StateSequenceCompressor:
    """状态序列压缩器类。

    将原始state_id序列压缩为结构化的segments格式。

    示例：
    >>> compressor = StateSequenceCompressor()
    >>> state_ids = [2, 2, 2, 5, 5, 5, 5, 5, 2, 2]
    >>> segments = compressor.compress(state_ids)
    >>> print(segments)
    [{"type": "s2", "duration": 30}, {"type": "s5", "duration": 50}, {"type": "s2", "duration": 20}]
    """

    def __init__(self, window_duration_sec: int = 10):
        """初始化状态序列压缩器。

        Args:
            window_duration_sec: 每个窗口的持续时间，单位秒
        """
        self.window_duration_sec = window_duration_sec

    def compress(self, state_ids: List[int]) -> List[Dict[str, Any]]:
        """将state_id列表压缩为segments格式。

        Args:
            state_ids: 每个窗口对应的state_id列表

        Returns:
            符合segments格式的字典列表

        Examples:
            >>> compressor = StateSequenceCompressor()
            >>> compressor.compress([2, 2, 2])
            [{"type": "s2", "duration": 30}]
            
            >>> compressor.compress([])
            []
        """
        # 处理空输入
        if not state_ids:
            logger.warning("输入state_id列表为空，返回空segments")
            return []

        segments = []
        current_state = state_ids[0]
        current_count = 1

        # 遍历state_id列表，合并连续相同状态
        for state_id in state_ids[1:]:
            if state_id == current_state:
                current_count += 1
            else:
                # 创建新的segment
                segment = {
                    "type": f"s{current_state}",
                    "duration": current_count * self.window_duration_sec
                }
                segments.append(segment)
                
                # 重置当前状态和计数
                current_state = state_id
                current_count = 1

        # 添加最后一个状态的segment
        last_segment = {
            "type": f"s{current_state}",
            "duration": current_count * self.window_duration_sec
        }
        segments.append(last_segment)

        logger.info(f"成功压缩 {len(state_ids)} 个窗口为 {len(segments)} 个segments")
        return segments