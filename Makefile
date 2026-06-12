# =============================================================
# Locate Anything Assistant — Makefile
# =============================================================
# All common developer commands.
#
# Usage:
#   make help           Show this help
#   make install        Install all backend + frontend dependencies
#   make dev            Start backend + frontend in development mode
#   make test           Run the full backend test suite
#   make lint           Run ruff linter on all Python files
#   make docker-up      Start full stack with Docker Compose
#   make docker-down    Stop Docker Compose
#   make download       Pre-download model weights
#   make clean          Remove generated files and caches
# =============================================================

.DEFAULT_GOAL := help
.PHONY: help install install-backend install-frontend \
        dev dev-backend dev-frontend \
        test test-verbose test-coverage lint format \
        docker-up docker-down docker-build docker-logs docker-clean \
        download verify-model clean

# ── Colours ──────────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

# ── Paths ────────────────────────────────────────────────────
BACKEND_DIR  := backend
FRONTEND_DIR := frontend
VENV         := $(BACKEND_DIR)/venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
PYTEST       := $(VENV)/bin/pytest

# ── Help ─────────────────────────────────────────────────────

help:
	@echo ""
	@echo "$(CYAN)Locate Anything Assistant$(RESET)"
	@echo ""
	@echo "$(YELLOW)Setup:$(RESET)"
	@echo "  make install          Install backend venv + frontend node_modules"
	@echo "  make install-backend  Install backend only"
	@echo "  make install-frontend Install frontend only"
	@echo "  make download         Pre-download model weights from HuggingFace"
	@echo ""
	@echo "$(YELLOW)Development:$(RESET)"
	@echo "  make dev              Start both backend and frontend (two terminals)"
	@echo "  make dev-backend      Start uvicorn on :8000"
	@echo "  make dev-frontend     Start Vite dev server on :5173"
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@echo "  make test             Run pytest (fast, no GPU)"
	@echo "  make test-verbose     Run pytest with verbose output"
	@echo "  make test-coverage    Run pytest with HTML coverage report"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Run ruff formatter"
	@echo ""
	@echo "$(YELLOW)Docker:$(RESET)"
	@echo "  make docker-build     Build backend + frontend images"
	@echo "  make docker-up        Start full stack (docker compose up -d)"
	@echo "  make docker-up-gpu    Start with GPU profile"
	@echo "  make docker-down      Stop and remove containers"
	@echo "  make docker-logs      Follow logs for all services"
	@echo "  make docker-clean     Remove images + volumes"
	@echo ""
	@echo "$(YELLOW)Utilities:$(RESET)"
	@echo "  make verify-model     Verify model weights are fully downloaded"
	@echo "  make clean            Remove caches, __pycache__, dist/"
	@echo ""

# ── Installation ─────────────────────────────────────────────

install: install-backend install-frontend
	@echo "$(GREEN)[OK] All dependencies installed.$(RESET)"

install-backend:
	@echo "$(CYAN)Creating Python virtual environment...$(RESET)"
	python3.11 -m venv $(VENV)
	@echo "$(CYAN)Installing PyTorch (CPU — override for GPU)...$(RESET)"
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "$(CYAN)Installing backend requirements...$(RESET)"
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt
	$(PIP) install pytest pytest-asyncio httpx pytest-cov ruff
	@echo "$(GREEN)[OK] Backend ready. Activate with: source $(VENV)/bin/activate$(RESET)"

install-frontend:
	@echo "$(CYAN)Installing frontend dependencies...$(RESET)"
	cd $(FRONTEND_DIR) && npm install
	@echo "$(GREEN)[OK] Frontend ready.$(RESET)"

install-gpu:
	@echo "$(CYAN)Installing PyTorch with CUDA 12.1...$(RESET)"
	python3.11 -m venv $(VENV)
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt

# ── Development servers ───────────────────────────────────────

dev:
	@echo "$(YELLOW)Start the backend and frontend in separate terminals:$(RESET)"
	@echo "  Terminal 1: make dev-backend"
	@echo "  Terminal 2: make dev-frontend"

dev-backend:
	@echo "$(CYAN)Starting FastAPI backend on http://localhost:8000 ...$(RESET)"
	@echo "$(YELLOW)Swagger UI: http://localhost:8000/docs$(RESET)"
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend:
	@echo "$(CYAN)Starting Vite dev server on http://localhost:5173 ...$(RESET)"
	cd $(FRONTEND_DIR) && npm run dev

# ── Tests ────────────────────────────────────────────────────

test:
	@echo "$(CYAN)Running pytest (no GPU required)...$(RESET)"
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  pytest tests/ -m "not gpu and not slow" --tb=short -q

test-verbose:
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  pytest tests/ -m "not gpu and not slow" -v --tb=long

test-coverage:
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  pytest tests/ -m "not gpu and not slow" \
	    --cov=app \
	    --cov-report=html:htmlcov \
	    --cov-report=term-missing
	@echo "$(GREEN)[OK] Coverage report: $(BACKEND_DIR)/htmlcov/index.html$(RESET)"

lint:
	@echo "$(CYAN)Running ruff linter...$(RESET)"
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  ruff check app/ tests/ --select E,F,W,I --ignore E501

format:
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  ruff format app/ tests/

# ── Docker ───────────────────────────────────────────────────

docker-build:
	@echo "$(CYAN)Building Docker images...$(RESET)"
	docker compose build
	@echo "$(GREEN)[OK] Images built.$(RESET)"

docker-up:
	@echo "$(CYAN)Starting full stack (CPU)...$(RESET)"
	@echo "$(YELLOW)Backend:  http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(RESET)"
	@echo "$(YELLOW)Swagger:  http://localhost:8000/docs$(RESET)"
	docker compose up -d
	@echo "$(GREEN)[OK] Stack started. Run 'make docker-logs' to follow logs.$(RESET)"

docker-up-gpu:
	@echo "$(CYAN)Starting full stack (GPU)...$(RESET)"
	docker compose --profile gpu up -d

docker-down:
	docker compose down
	@echo "$(GREEN)[OK] Stack stopped.$(RESET)"

docker-logs:
	docker compose logs -f

docker-clean:
	docker compose down -v --rmi all
	@echo "$(GREEN)[OK] All containers, images, and volumes removed.$(RESET)"

# ── Model management ─────────────────────────────────────────

download:
	@echo "$(CYAN)Pre-downloading nvidia/LocateAnything-3B...$(RESET)"
	@echo "$(YELLOW)Make sure HF_TOKEN is set in your environment.$(RESET)"
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  python scripts/download_model.py

verify-model:
	cd $(BACKEND_DIR) && \
	  source venv/bin/activate && \
	  python scripts/download_model.py --verify-only

# ── Cleanup ──────────────────────────────────────────────────

clean:
	@echo "$(CYAN)Cleaning generated files...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete 2>/dev/null || true
	find . -type f -name "*.pyo"     -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "coverage.xml"  -delete 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/dist 2>/dev/null || true
	@echo "$(GREEN)[OK] Clean complete.$(RESET)"
