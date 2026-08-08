.DEFAULT_GOAL := help
UV := uv

.PHONY: help install dev test lint format typecheck check up down logs models clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and pre-commit hooks
	$(UV) sync --extra dev
	$(UV) run pre-commit install || true

dev: ## Run the API with reload
	$(UV) run uvicorn researchagent.api.app:app --reload --host 0.0.0.0 --port 8000

test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint
	$(UV) run ruff check .

format: ## Format
	$(UV) run ruff format .

typecheck: ## Static type check
	$(UV) run mypy researchagent

check: lint typecheck test ## Everything CI runs

up: ## Start the local stack
	docker compose up -d

down: ## Stop the local stack
	docker compose down

logs: ## Tail API logs
	docker compose logs -f api

models: ## Pull the models referenced by config/models.yaml
	./scripts/pull_models.sh

clean: ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
