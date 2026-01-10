.DEFAULT_GOAL := help
.PHONY: help install data benchmark clean clean-all test lint format


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
	python src/pipeline/recbole_data_pipeline.py
	@echo "[INFO] Dataset preparation complete."

data-custom: ## Prepara dados com intervalo customizado (Requer START_DATE e END_DATE)
	@if [ -z "$(START_DATE)" ] || [ -z "$(END_DATE)" ]; then \
		echo "[ERROR] START_DATE and END_DATE arguments are required."; \
		exit 1; \
	fi
	python src/pipeline/recbole_data_pipeline.py \
		--start-date $(START_DATE) \
		--end-date $(END_DATE)

# -----------------------------------------------------------------------------
##@ Execução de Benchmark
# -----------------------------------------------------------------------------
benchmark: ## Executa modelos. Opcional: MODELS='...' DATASET='...'
	@MODELS_ARG=$(or $(MODELS),all); \
	DATASET_ARG=$(or $(DATASET),); \
	echo "[INFO] Starting benchmark execution (Models: $$MODELS_ARG)..."; \
	if [ -n "$$DATASET_ARG" ]; then \
		python src/run_benchmark.py --models $$MODELS_ARG --dataset $$DATASET_ARG; \
	else \
		python src/run_benchmark.py --models $$MODELS_ARG; \
	fi

benchmark-neurais: ## Executa apenas modelos baseados em Redes Neurais (GRU4Rec, NARM, etc.)
	@echo "[INFO] Running neural models benchmark..."
	python src/run_benchmark.py --models neurais

benchmark-baselines: ## Executa apenas modelos Baseline (Random, POP, etc.)
	@echo "[INFO] Running baselines benchmark..."
	python src/run_benchmark.py --models baselines

benchmark-factor: ## Executa apenas modelos de Fatoração (FPMC, FOSSIL)
	@echo "[INFO] Running factorization models benchmark..."
	python src/run_benchmark.py --models factorization

benchmark-quick: ## Executa teste rápido (GRU4Rec) para validação de sanidade
	@echo "[INFO] Running quick sanity check (GRU4Rec)..."
	python src/run_benchmark.py --models GRU4Rec

# -----------------------------------------------------------------------------
##@ Análise de Resultados
# -----------------------------------------------------------------------------
aggregate: ## Agrega os resultados da execução mais recente em um CSV único
	@LAST_DIR=$$(ls -td outputs/results/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_DIR" ]; then \
		echo "[ERROR] No results found in outputs/results/"; \
		exit 1; \
	fi; \
	echo "[INFO] Processing results from: $$LAST_DIR"; \
	python src/aggregate_results.py \
		--input "$${LAST_DIR%/}" \
		--output "$${LAST_DIR%/}/aggregated_results.csv"

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

api-dev: ## Inicia API em modo desenvolvimento com auto-reload
	@if [ -z "$(MODEL)" ]; then \
		echo "[ERROR] MODEL argument is required."; \
		exit 1; \
	fi
	@echo "[INFO] Starting API (dev mode) with model: $(MODEL)..."
	python src/api/app.py --model $(MODEL) --reload

# -----------------------------------------------------------------------------
##@ Manutenção e Limpeza
# -----------------------------------------------------------------------------
clean: ## Remove arquivos de cache do Python (__pycache__, .pyc)
	@echo "[INFO] Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove cache, logs, resultados e checkpoints salvos (Reset total)
	@echo "[INFO] Cleaning all artifacts (results, logs, checkpoints)..."
	rm -rf outputs/results/* 2>/dev/null || true
	rm -rf outputs/logs/* 2>/dev/null || true
	rm -rf outputs/saved/* 2>/dev/null || true
	rm -rf log_tensorboard/* 2>/dev/null || true

# -----------------------------------------------------------------------------
##@ Desenvolvimento
# -----------------------------------------------------------------------------
format: ## Formata o código fonte (black)
	black src/ --line-length=100