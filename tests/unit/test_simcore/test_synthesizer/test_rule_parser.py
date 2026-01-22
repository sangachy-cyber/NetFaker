"""规则解析器测试模块。

测试规则解析器的模板校验和转换功能。
"""

import pytest
from netfaker.simcore.synthesizer.rule_parser import RuleParser


class TestRuleParser:
    """测试规则解析器类。
    """

    def test_parse_template_valid(self):
        """测试解析有效模板。
        """
        template = [
            {"type": "s0", "duration": 10},
            {"type": "s3", "duration": 20}
        ]
        result = RuleParser.parse_template(template)
        assert result == [(0, 1), (3, 2)]

    def test_parse_template_invalid_type(self):
        """测试解析无效类型模板。
        """
        template = [
            {"type": "invalid", "duration": 10}
        ]
        with pytest.raises(ValueError, match="无效的 type 格式"):
            RuleParser.parse_template(template)

    def test_parse_template_invalid_duration(self):
        """测试解析无效时长模板。
        """
        template = [
            {"type": "s0", "duration": 5}
        ]
        with pytest.raises(ValueError, match="duration 必须是 10 的正整数倍"):
            RuleParser.parse_template(template)

    def test_parse_template_missing_type(self):
        """测试解析缺少 type 字段的模板。
        """
        template = [
            {"duration": 10}
        ]
        with pytest.raises(ValueError, match="模板项缺少 'type' 字段"):
            RuleParser.parse_template(template)

    def test_parse_template_missing_duration(self):
        """测试解析缺少 duration 字段的模板。
        """
        template = [
            {"type": "s0"}
        ]
        with pytest.raises(ValueError, match="模板项缺少 'duration' 字段"):
            RuleParser.parse_template(template)
