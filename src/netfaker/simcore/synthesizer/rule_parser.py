"""规则解析器模块。

负责校验和转换用户提供的状态序列模板，确保其符合输入规范。
"""

import re
from typing import List, Tuple


class RuleParser:
    """规则解析器类。

    用于校验和转换用户提供的状态序列模板。
    """

    @staticmethod
    def parse_template(template: List[dict]) -> List[Tuple[int, int]]:
        """解析并校验状态序列模板。

        Args:
            template: 用户提供的状态序列模板，格式为 [{"type": "s0", "duration": 10}, ...]

        Returns:
            List[Tuple[int, int]]: 解析后的模板，格式为 [(state_id, n_windows), ...]

        Raises:
            ValueError: 当模板格式不符合规范时

        Examples:
            >>> template = [{"type": "s0", "duration": 10}, {"type": "s3", "duration": 20}]
            >>> RuleParser.parse_template(template)
            [(0, 1), (3, 2)]
        """
        result = []

        for item in template:
            # 校验必填字段
            if "type" not in item:
                raise ValueError("模板项缺少 'type' 字段")
            if "duration" not in item:
                raise ValueError("模板项缺少 'duration' 字段")

            # 校验 type 格式
            type_str = item["type"]
            match = re.match(r"^s(\d+)$", type_str)
            if not match:
                raise ValueError(f"无效的 type 格式: {type_str}，预期格式为 's{int}'")

            state_id = int(match.group(1))

            # 校验 duration
            duration = item["duration"]
            if not isinstance(duration, int) or duration <= 0:
                raise ValueError(f"duration 必须是正整数: {duration}")
            if duration % 10 != 0:
                raise ValueError(f"duration 必须是 10 的正整数倍: {duration}")

            # 计算所需窗口数
            n_windows = duration // 10
            result.append((state_id, n_windows))

        return result
