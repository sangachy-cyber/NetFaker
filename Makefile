# NetFaker 开发快捷命令

.PHONY: help install dev-install run test lint format clean coverage

# 默认目标
help:
	@echo "NetFaker 开发命令"
	@echo "=================="
	@echo "install   - 安装运行时依赖"
	@echo "dev-install - 安装开发依赖"
	@echo "run       - 启动开发服务器"
	@echo "test      - 运行所有测试"
	@echo "lint      - 运行代码风格检查"
	@echo "format    - 自动格式化代码"
	@echo "clean     - 清理临时文件"
	@echo "coverage  - 生成测试覆盖率报告"

# 安装运行时依赖
install:
	uv sync --only-group=default

# 安装开发依赖
dev-install:
	uv sync --all-extras

# 安装第三方库
third-party-install:
	uv pip install -e ./third_party/holowan_python_api-2908

# 安装所有依赖（运行时+开发+第三方）
all-install:
	uv sync --all-extras
	uv pip install -e ./third_party/holowan_python_api-2908


# 启动开发服务器
run:
	uv run uvicorn netfaker.app.main:app --host 0.0.0.0 --port 8000 --reload

# 运行所有测试
test:
	uv run pytest

# 运行代码风格检查
lint:
	uv run ruff check .

# 自动格式化代码
format:
	uv run ruff format .

# 清理临时文件
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	rm -rf htmlcov .coverage
	rm -rf dist build *.egg-info

# 生成测试覆盖率报告
coverage:
	uv run pytest --cov=src --cov-report=html
