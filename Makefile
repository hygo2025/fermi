.DEFAULT_GOAL := help
.PHONY: help install prepare-raw-data data benchmark tune eval-only api clean format

COLOR_RESET   = \033[0m
COLOR_CYAN    = \033[36m
COLOR_YELLOW  = \033[33m
COLOR_GREEN   = \033[32m

# -----------------------------------------------------------------------------
# HELP SYSTEM
# -----------------------------------------------------------------------------
help: ## Show available commands
	@echo ""
	@echo "Available Commands"
	@echo "----------------------------------------------------------------"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(COLOR_YELLOW)<target>$(COLOR_RESET)\n"} \
	/^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOR_CYAN)%-25s$(COLOR_RESET) %s\n", $$1, $$2 } \
	/^##@/ { printf "\n$(COLOR_GREEN)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

# -----------------------------------------------------------------------------
##@ Setup & Installation
# -----------------------------------------------------------------------------
install: ## Install all project dependencies
	@echo "[INFO] Installing dependencies..."
	pip install -e .

# -----------------------------------------------------------------------------
##@ Data Pipeline
# -----------------------------------------------------------------------------
prepare-raw-data: ## Process raw data (listings + events)
	@echo "[INFO] Processing raw data..."
	python src/data_preparation/prepare_raw_data.py
	@echo "[INFO] Done."

data: ## Prepare dataset for RecBole
	@echo "[INFO] Starting data pipeline..."
	python src/data_preparation/recbole_data_pipeline.py
	@echo "[INFO] Done."

# -----------------------------------------------------------------------------
##@ Benchmark Execution
#@GROUP_NAME="run_$(shell date +%m-%d_%H-%M)"; \
# -----------------------------------------------------------------------------
benchmark: ## Run benchmark. Optional: MODEL=...
	@GROUP_NAME="run_final"; \
	echo "[INFO] W&B Group: $$GROUP_NAME"; \
	if [ -n "$(MODEL)" ]; then \
		WANDB_RUN_GROUP="$$GROUP_NAME" ./scripts/run_benchmark.sh "$(MODEL)"; \
	else \
		WANDB_RUN_GROUP="$$GROUP_NAME" ./scripts/run_benchmark.sh; \
	fi

eval-only: ## Evaluate a saved checkpoint. MODEL=... CHECKPOINT=... EVAL_BATCH_SIZE=...
	@if [ -z "$(MODEL)" ] || [ -z "$(CHECKPOINT)" ]; then \
		echo "[ERROR] MODEL and CHECKPOINT are required."; \
		echo "[INFO] Usage: make eval-only MODEL=TransRec CHECKPOINT=outputs/saved/TransRec-....pth EVAL_BATCH_SIZE=1"; \
		exit 1; \
	fi
	@EVAL_BATCH_SIZE_ARG="$(if $(EVAL_BATCH_SIZE),--eval-batch-size $(EVAL_BATCH_SIZE),)"; \
	WANDB_GROUP_ARG="$(if $(WANDB_GROUP),--wandb-group $(WANDB_GROUP),)"; \
	python src/eval_only.py --model $(MODEL) --checkpoint $(CHECKPOINT) $$EVAL_BATCH_SIZE_ARG $$WANDB_GROUP_ARG

# -----------------------------------------------------------------------------
##@ Hyperparameter Tuning
# -----------------------------------------------------------------------------
tune: ## Run hyperparameter tuning. MODEL=... or empty for all
	@if [ -n "$(MODEL)" ]; then \
		DATASET_ARG="$(if $(DATASET),--dataset $(DATASET),)"; \
		ALGO_ARG="$(if $(ALGO),--algo $(ALGO),)"; \
		MAX_EVALS_ARG="$(if $(MAX_EVALS),--max-evals $(MAX_EVALS),)"; \
		EARLY_STOP_ARG="$(if $(EARLY_STOP),--early-stop $(EARLY_STOP),)"; \
		COOLDOWN_ARG="$(if $(COOLDOWN),--cooldown $(COOLDOWN),)"; \
		OUTPUT_ARG="$(if $(OUTPUT),--output $(OUTPUT),)"; \
		echo "[INFO] Running tuning for $(MODEL) $$MAX_EVALS_ARG $$ALGO_ARG $$COOLDOWN_ARG"; \
		python src/hyperparameter_tuning.py --model $(MODEL) $$DATASET_ARG $$ALGO_ARG $$MAX_EVALS_ARG $$EARLY_STOP_ARG $$COOLDOWN_ARG $$OUTPUT_ARG; \
	else \
		echo "[INFO] No MODEL specified, running tune_remaining_models.sh for all models"; \
		echo "[INFO] MAX_EVALS=$(or $(MAX_EVALS),150), ALGO=$(or $(ALGO),bayes), COOLDOWN=$(or $(COOLDOWN),60)"; \
		MAX_EVALS=$(or $(MAX_EVALS),150) ALGO=$(or $(ALGO),bayes) COOLDOWN=$(or $(COOLDOWN),60) DATASET=$(DATASET) ./scripts/tune_remaining_models.sh; \
	fi

# -----------------------------------------------------------------------------
##@ API Server
# -----------------------------------------------------------------------------
api: ## Start API server (Requires: MODEL=filename.pth)
	@if [ -z "$(MODEL)" ]; then \
		echo "[ERROR] MODEL argument required."; \
		echo "[INFO] Usage: make api MODEL=GRU4Rec"; \
		exit 1; \
	fi
	@echo "[INFO] Starting API with model: $(MODEL)..."
	python src/api/app.py --model $(MODEL)

# -----------------------------------------------------------------------------
##@ Maintenance
# -----------------------------------------------------------------------------
clean: ## Remove cache, logs and checkpoints
	@echo "[INFO] Cleaning artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/results/* 2>/dev/null || true
	rm -rf outputs/logs/* 2>/dev/null || true
	rm -rf outputs/saved/realestate-SequentialDataset.pth* 2>/dev/null || true
	rm -rf log_tensorboard/* 2>/dev/null || true
	rm -rf log/* 2>/dev/null || true

# -----------------------------------------------------------------------------
##@ Development
# -----------------------------------------------------------------------------
format: ## Format source code (black)
	black src/ --line-length=100
