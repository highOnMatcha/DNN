# Makefile for Pokemon Sprites Project

.PHONY: help install lint lint-fix format test test-verbose clean setup-hooks

# Default target
helci-test:
	pytest tests/ -v --tb=short --cov=src --cov-report=xml --cov-fail-under=70
	@echo "Pokemon Sprites Project - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install all dependencies and development tools"
	@echo "  setup-hooks    Install git hooks for automatic linting"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run all linting checks (no fixes)"
	@echo "  lint-fix       Run all linting checks and auto-fix issues"
	@echo "  format         Format code with black and isort"
	@echo "  check          Run all quality checks (lint + test)"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-verbose   Run tests with verbose output"
	@echo "  test-unit      Run only unit tests"
	@echo "  test-integration Run only integration tests"
	@echo "  test-performance Run only performance tests"
	@echo "  test-coverage  Run tests with coverage analysis"
	@echo "  coverage-report Show coverage report from existing data"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean up temporary files and caches"
	@echo "  requirements   Generate requirements.txt from current environment"

# Installation and setup
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black isort flake8 autoflake mypy pytest pytest-cov pre-commit
	@echo "Dependencies installed successfully"

setup-hooks:
	pre-commit install
	@echo "Git hooks installed successfully"

# Code quality
lint:
	@echo "Running linting checks..."
	./scripts/lint.sh

lint-fix:
	@echo "Running linting checks with auto-fix..."
	./scripts/lint.sh --fix

format:
	@echo "Formatting code..."
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports $$(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*")
	isort --profile=black --line-length=120 $$(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*")
	black --line-length=120 $$(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*")
	@echo "Code formatted successfully"

check:
	@echo "Running auto-fix and quality checks..."
	./scripts/lint.sh --fix || true
	@echo "Running tests with coverage..."
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term --cov-fail-under=70
	@echo ""
	@echo "Coverage report generated in htmlcov/ directory"
	@echo "All quality checks completed!"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

test-verbose:
	@echo "Running tests with verbose output..."
	pytest tests/ -vv --tb=long

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

test-performance:
	@echo "Running performance tests..."
	pytest tests/performance/ -v --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term --cov-fail-under=70
	@echo ""
	@echo "Coverage report generated in htmlcov/ directory"
	@echo "Current coverage requirement: 70% minimum for quality assurance"

coverage-report:
	@echo "Generating coverage report from existing data..."
	coverage report --show-missing
	@echo ""
	@echo "For detailed HTML report, open: htmlcov/index.html"

# Utilities
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "Cleanup completed"

requirements:
	@echo "Generating requirements.txt..."
	pip freeze > requirements.txt
	@echo "requirements.txt updated"

# Development shortcuts
dev-setup: install setup-hooks
	@echo "Development environment setup completed!"

pre-commit: lint-fix test
	@echo "Pre-commit checks completed successfully!"

# CI/CD helpers
ci-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black isort flake8 autoflake mypy pytest pytest-cov

ci-check:
	black --check --line-length=120 .
	isort --check-only --profile=black --line-length=120 .
	flake8 . --max-line-length=120 --ignore=E203,W503,E501 --max-complexity=15 --exclude=__pycache__,*.pyc,.git,build,dist,.venv,venv
	autoflake --check --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports --recursive .

ci-test:
	pytest tests/ -v --tb=short --cov=src --cov-report=xml --cov-fail-under=72
