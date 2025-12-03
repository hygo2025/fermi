PYTHON_VERSION = 3.9
VENV_DIR = .venv
REQUIREMENTS_FILE = requirements.txt
ACTIVATE = . $(VENV_DIR)/bin/activate

PYTHONPATH ?= $(PWD)/src:$(PWD)/session-rec-lib
SRC_DIR = src
SCRIPTS_DIR = scripts

export PYTHONUNBUFFERED = 1
export PYTHONPATH

.DEFAULT_GOAL := help

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

check_python_version:
	@installed_version=$$(pyenv versions --bare | grep $(PYTHON_VERSION)); \
	if [ -z "$$installed_version" ]; then \
		echo "Python version $(PYTHON_VERSION) not found. Installing..."; \
		pyenv install $(PYTHON_VERSION); \
		pyenv global $(PYTHON_VERSION); \
	else \
		echo "Python version $(PYTHON_VERSION) is already installed."; \
	fi

$(VENV_DIR): check_python_version
	@echo "Creating virtual environment in $(VENV_DIR)..."
	pyenv local $(PYTHON_VERSION)
	pyenv exec python -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r $(REQUIREMENTS_FILE)

install: $(VENV_DIR) ## Install Python dependencies only
	@echo "✓ Python dependencies installed"

setup: install-benchmark ## Complete setup (alias for install-benchmark)
	@echo "✓ Setup complete! Run 'make status' to verify"

install-benchmark: install ## Install benchmark (dependencies + session-rec submodule)
	@echo "Installing benchmark environment..."
	@echo "Initializing git submodules..."
	@git submodule update --init --recursive
	@echo "✓ Session-rec submodule ready (imported via PYTHONPATH)"
	@echo "✓ Benchmark environment ready"

update: ## Update Python dependencies
	$(ACTIVATE) && pip install --upgrade -r $(REQUIREMENTS_FILE)

##@ Benchmark - Pattern Mining Models

test-ar: ## Run AR (Association Rules)
	@echo "Running AR (Association Rules)..."
	$(ACTIVATE) && python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/ar.yml

test-markov: ## Run Markov Chain
	@echo "Running Markov Chain..."
	$(ACTIVATE) && python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/markov.yml

test-sr: ## Run SR (Sequential Rules)
	@echo "Running SR (Sequential Rules)..."
	$(ACTIVATE) && python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/sr.yml

test-pattern-mining: ## Run all pattern mining models in parallel
	@echo "Running all pattern mining models in parallel..."
	@mkdir -p logs
	@$(ACTIVATE) && \
	python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/ar.yml > logs/ar.log 2>&1 & \
	python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/markov.yml > logs/markov.log 2>&1 & \
	python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pattern_mining/sr.yml > logs/sr.log 2>&1 & \
	wait
	@echo "✓ All pattern mining models complete. Check logs/ directory for outputs."

##@ Benchmark - Run All

run-all-baselines: test-pattern-mining ## Run all implemented models

##@ Data Preparation

prepare-data: ## Prepare dataset with Spark (30 days)
	@echo "Preparing dataset with Spark..."
	@if [ ! -f .env ]; then echo "ERROR: .env file not found. Create it with BASE_PATH variable."; exit 1; fi
	$(ACTIVATE) && python data/prepare_dataset.py \
		--start-date 2024-04-01 \
		--end-date 2024-04-30

##@ Cleanup

clean: ## Remove virtual environment and generated files
	rm -rf $(VENV_DIR)
	rm -rf src/results/*
	rm -rf data/processed/
	rm -rf data/session_rec_format/
	@echo "✓ Cleaned up"

clean-all: clean ## Remove everything including session-rec lib
	rm -rf session-rec-lib
	@echo "✓ Full cleanup complete"

##@ Development

status: ## Show project status
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║         Fermi - Session-Rec Benchmark Status            ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Python version: $(PYTHON_VERSION)"
	@if [ -d "$(VENV_DIR)" ]; then echo "✓ Virtual environment: installed"; else echo "✗ Virtual environment: not found (run 'make install')"; fi
	@if [ -d "session-rec-lib" ]; then echo "✓ Session-rec: installed"; else echo "✗ Session-rec: not found (run 'make install-benchmark')"; fi
	@if [ -d "data/session_rec_format" ]; then echo "✓ Data: prepared"; else echo "✗ Data: not prepared (run 'make prepare-data')"; fi
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install-benchmark  (first time only)"
	@echo "  2. make prepare-data       (prepare dataset)"
	@echo "  3. make test-sr            (test with Sequential Rules)"
	@echo ""

.PHONY: help install install-benchmark update test-ar test-markov test-sr test-pattern-mining run-all-baselines prepare-data clean clean-all status
