# Stock Advisor Makefile

.PHONY: help install install-dev install-ui install-ml clean test lint format run-ui run-cli predict docs

# Default target
help:
	@echo "Stock Advisor - Available commands:"
	@echo "  install        Install core dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  install-ui     Install UI dependencies"
	@echo "  install-ml     Install ML dependencies"
	@echo "  install-all    Install all dependencies"
	@echo "  clean          Clean build artifacts"
	@echo "  test           Run tests"
	@echo "  lint           Run linting"
	@echo "  format         Format code"
	@echo "  run-ui         Start web UI"
	@echo "  run-cli        Run CLI interface"
	@echo "  predict        Run prediction script"
	@echo "  docs           Generate documentation"
	@echo "  build          Build package"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-ui:
	pip install -e .[ui]

install-ml:
	pip install -e .[ml]

install-all:
	pip install -e .[dev,ui,ml]

# Development targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest stock_advisor/tests/ -v

test-cov:
	pytest stock_advisor/tests/ -v --cov=stock_advisor --cov-report=html

lint:
	flake8 stock_advisor/ scripts/
	mypy stock_advisor/

format:
	black stock_advisor/ scripts/
	isort stock_advisor/ scripts/

# Runtime targets
run-ui:
	cd . && python scripts/run_ui.py

run-cli:
	cd . && python scripts/main.py

predict:
	cd . && python scripts/predict_3_days.py

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Build targets
build:
	python -m build

docker-build:
	docker build -t stock-advisor .

docker-run:
	docker run -p 8501:8501 stock-advisor

# Setup development environment
setup-dev:
	pip install -e .[dev,ui,ml]
	pre-commit install

# Run all quality checks
check-all: lint test

# Quick development setup
dev-setup: setup-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make run-ui' to start the web interface"