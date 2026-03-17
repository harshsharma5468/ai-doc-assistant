.PHONY: help dev test lint build up down logs clean deploy-staging

DOCKER_COMPOSE = docker compose -f docker/docker-compose.yml
BACKEND_DIR = backend
FRONTEND_DIR = frontend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────
dev: ## Start backend in dev mode with hot reload
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --port 8000 --log-level debug

dev-ui: ## Start Streamlit UI
	cd $(FRONTEND_DIR) && streamlit run app.py --server.runOnSave true

install: ## Install all dependencies
	pip install -r $(BACKEND_DIR)/requirements.txt
	pip install -r $(FRONTEND_DIR)/requirements.txt
	pip install ruff mypy pytest pytest-asyncio pytest-cov httpx bandit

# ── Testing ────────────────────────────────────────────────────────────────────
test: ## Run tests with coverage
	cd $(BACKEND_DIR) && pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=70

test-fast: ## Run tests without coverage
	cd $(BACKEND_DIR) && pytest tests/ -v -x

test-watch: ## Watch tests
	cd $(BACKEND_DIR) && ptw tests/ -- -v

# ── Linting ───────────────────────────────────────────────────────────────────
lint: ## Run all linters
	ruff check $(BACKEND_DIR)/app
	ruff format --check $(BACKEND_DIR)/app
	bandit -r $(BACKEND_DIR)/app -ll -x $(BACKEND_DIR)/tests

format: ## Auto-format code
	ruff format $(BACKEND_DIR)/app
	ruff check --fix $(BACKEND_DIR)/app

# ── Docker ────────────────────────────────────────────────────────────────────
build: ## Build all Docker images
	$(DOCKER_COMPOSE) build --no-cache

up: ## Start all services
	$(DOCKER_COMPOSE) up -d
	@echo "UI:      http://localhost:8501"
	@echo "API:     http://localhost:8000"
	@echo "Docs:    http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000"

down: ## Stop all services
	$(DOCKER_COMPOSE) down

restart: ## Restart all services
	$(DOCKER_COMPOSE) restart

logs: ## Tail all logs
	$(DOCKER_COMPOSE) logs -f

logs-backend: ## Tail backend logs only
	$(DOCKER_COMPOSE) logs -f backend

health: ## Check service health
	curl -s http://localhost:8000/health/ | python3 -m json.tool

ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

# ── Data ──────────────────────────────────────────────────────────────────────
clean-data: ## Remove vector store and uploads (DESTRUCTIVE)
	@read -p "Delete all data? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -rf data/vectorstore/* data/uploads/*
	@echo "Data cleared"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf $(BACKEND_DIR)/.pytest_cache $(BACKEND_DIR)/.coverage $(BACKEND_DIR)/coverage.xml
	rm -rf $(BACKEND_DIR)/.mypy_cache $(BACKEND_DIR)/.ruff_cache

clean-docker: ## Remove all containers and images
	$(DOCKER_COMPOSE) down --rmi all --volumes
