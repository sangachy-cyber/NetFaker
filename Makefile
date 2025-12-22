.PHONY: install test format clean

install:
	uv sync

test:
	pytest

format:
	ruff format .

clean:
	rm -rf output/*