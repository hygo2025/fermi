.DEFAULT_GOAL := help
.PHONY: help install prepare-raw-data data benchmark tune api clean format

COLOR_RESET   = \033[0m
COLOR_CYAN    = \033[36m
COLOR_YELLOW  = \033[33m
COLOR_GREEN   = \033[32m

# -----------------------------------------------------------------------------
# HELP SYSTEM
# -----------------------------------------------------------------------------
help: ## Exibe esta mensagem de ajuda
	@echo ""
	@echo "Fermi Benchmark - Comandos Disponíveis"
	@echo "----------------------------------------------------------------"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(COLOR_YELLOW)<target>$(COLOR_RESET)\n"} \
	/^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOR_CYAN)%-25s$(COLOR_RESET) %s\n", $$1, $$2 } \
	/^##@/ { printf "\n$(COLOR_GREEN)%s$(COLOR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

# -----------------------------------------------------------------------------
##@ Setup & Instalação
# -----------------------------------------------------------------------------
install: ## Instala todas as dependências do projeto em modo editável
	@echo "[INFO] Installing dependencies..."
	pip install -e .

# -----------------------------------------------------------------------------
##@ Pipeline de Dados
# -----------------------------------------------------------------------------
prepare-raw-data: ## Processa dados brutos (listings + events) - Etapa 0
	@echo "[INFO] Processing raw data (listings + events)..."
	python src/data_preparation/prepare_raw_data.py
	@echo "[INFO] Raw data processing complete."

data: ## Prepara o dataset para o RecBole (Global Temporal Leave-One-Out)
	@echo "[INFO] Starting data preparation pipeline..."
	python src/data_preparation/recbole_data_pipeline.py
	@echo "[INFO] Dataset preparation complete."

# -----------------------------------------------------------------------------
##@ Execução de Benchmark
# -----------------------------------------------------------------------------
benchmark: ## Executa benchmark. Opcional: MODEL=... (vazio = todos os modelos)
	@if [ -n "$(MODEL)" ]; then \
		./scripts/run_benchmark.sh "$(MODEL)"; \
	else \
		./scripts/run_benchmark.sh; \
	fi

# -----------------------------------------------------------------------------
##@ Hyperparameter Tuning
# -----------------------------------------------------------------------------
tune: ## Executa hyperparameter tuning. MODEL=... para um modelo, vazio para todos
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
api: ## Inicia servidor API (Requer: MODEL=nome_do_modelo ou path.pth)
	@if [ -z "$(MODEL)" ]; then \
		echo "[ERROR] MODEL argument is required."; \
		echo "[INFO] Usage: make api MODEL=GRU4Rec"; \
		echo "[INFO]    or: make api MODEL=outputs/saved/GRU4Rec-Dec-31-2024_12-34-56.pth"; \
		exit 1; \
	fi
	@echo "[INFO] Starting API with model: $(MODEL)..."
	python src/api/app.py --model $(MODEL)

# -----------------------------------------------------------------------------
##@ Manutenção e Limpeza
# -----------------------------------------------------------------------------
clean: ## Remove cache, logs, resultados e checkpoints salvos (Reset total)
	@echo "[INFO] Cleaning all artifacts (results, logs, checkpoints)..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/results/* 2>/dev/null || true
	rm -rf outputs/logs/* 2>/dev/null || true
	rm -rf outputs/saved/* 2>/dev/null || true
	rm -rf log_tensorboard/* 2>/dev/null || true
	rm -rf log/* 2>/dev/null || true
	rm -rf wandb/* 2>/dev/null || true

# -----------------------------------------------------------------------------
##@ Desenvolvimento
# -----------------------------------------------------------------------------
format: ## Formata o código fonte (black)
	black src/ --line-length=100