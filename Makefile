.PHONY: help install install-dev test clean demo format lint

help:  ## Show this help message
	@echo "segimage - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the library in development mode
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	python -m pytest tests/ -v

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

demo:  ## Run the demo script
	python demo.py

format:  ## Format code with black
	black src/ tests/ examples/

lint:  ## Run linting checks
	flake8 src/ tests/ examples/

check: format lint test  ## Run all checks (format, lint, test)

build:  ## Build the package
	python -m build

install-cli: install  ## Install and make CLI available
	@echo "CLI commands now available:"
	@echo "  segimage process input.mat output_dir"
	@echo "  segimage process input.mat output_dir -f png"
	@echo "  segimage process input.mat output_dir -f jpg"
	@echo "  segimage formats"
	@echo "  segimage info"
