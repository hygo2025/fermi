PYTHON_VERSION = 3.9
VENV_DIR = .venv
REQUIREMENTS_FILE = requirements.txt
ACTIVATE = . $(VENV_DIR)/bin/activate

PYTHONPATH ?= $(PWD)/src
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

install-benchmark: install ## Install benchmark (dependencies + session-rec)
	@echo "Installing benchmark environment..."
	@cd $(SCRIPTS_DIR) && ./install.sh
	@echo "✓ Benchmark environment ready"

update: ## Update Python dependencies
	$(ACTIVATE) && pip install --upgrade -r $(REQUIREMENTS_FILE)

##@ Benchmark

test-pop: ## Run quick test with POP baseline
	@echo "Running POP baseline test..."
	$(ACTIVATE) && python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/pop_only.yml

run-benchmark: ## Run full benchmark (all models)
	@echo "Running full benchmark..."
	$(ACTIVATE) && python $(SRC_DIR)/run_session_rec.py --config $(SRC_DIR)/configs/session_rec_config.yml

prepare-data: ## Prepare sample dataset (14 days)
	@echo "Preparing sample dataset..."
	$(ACTIVATE) && python $(SRC_DIR)/prepare_dataset.py \
		--start-date 2024-03-01 \
		--end-date 2024-03-15

convert-data: ## Convert data to session-rec format
	@echo "Converting to session-rec format..."
	$(ACTIVATE) && python $(SRC_DIR)/convert_to_session_rec.py

##@ Cleanup

clean: ## Remove virtual environment and generated files
	rm -rf $(VENV_DIR)
	rm -rf src/results/*
	rm -rf data/processed_sample
	rm -rf data/session_rec_format
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
	@echo "  2. make prepare-data       (prepare sample dataset)"
	@echo "  3. make convert-data       (convert to session-rec format)"
	@echo "  4. make test-pop           (test with POP baseline)"
	@echo ""

.PHONY: help install install-benchmark update test-pop run-benchmark prepare-data convert-data clean clean-all status
