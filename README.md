# NetFaker

网络仿真参数生成工具，用于生成各种网络场景下的仿真参数。

## 功能特性

- 🎯 支持多种网络场景仿真
- 📊 基于规则和机器学习的参数生成策略
- 🌐 提供RESTful API接口
- 🛠️ 命令行工具支持
- 📈 可视化仿真结果

## 快速开始

### 安装依赖

```bash
# 安装开发依赖
uv sync --all-extras

# 安装包（使netfaker可导入）
uv pip install -e .
```

### 启动开发服务器

```bash
uv run uvicorn netfaker.app.main:app --reload
```

服务器将在 `http://localhost:8000` 启动。

### 使用命令行工具

```bash
# 列出所有可用策略
uv run python -m scripts.main list

# 生成仿真参数
uv run python -m scripts.main generate --strategy rule_based --scenario video_streaming
```

## 项目结构

```
netfaker/
├── src/netfaker/             # 所有可导入的Python包代码
│   ├── core/                 # 公共模块（配置、日志、异常、工具）
│   ├── app/                  # Web服务层（FastAPI）
│   └── simcore/              # 仿真核心逻辑（含ML与规则策略）
├── scripts/                  # 命令行入口脚本（非包内）
├── tests/                    # 单元测试与集成测试
├── data/                     # 数据资产（原始/处理/输出）
├── pyproject.toml            # 项目元数据与工具配置
├── .gitignore
├── README.md
└── Makefile                  # 开发快捷命令
```

## 开发规范

请参考 [开发规范](docs/DEVELOPMENT_GUIDE.md) 了解项目的编码标准、工具链配置与协作流程。

## 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_strategy.py

# 生成测试覆盖率报告
uv run pytest --cov=src --cov-report=html
```

## 代码风格

```bash
# 自动格式化代码
uv run ruff format .

# 运行代码风格检查
uv run ruff check .
```

## 贡献

欢迎提交Issue和Pull Request！请确保您的代码符合项目的开发规范。

## 许可证

MIT License
