# =============================================================================
# FERMI - SESSION-BASED RECOMMENDATION BENCHMARK
# =============================================================================
# Arquitetura: Python-First (Wrapper)
# Descrição: Orquestração de pipelines de dados e execução de modelos RecBole.
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help install data benchmark clean clean-all test lint format

# Cores para o terminal (opcional, para legibilidade do help)
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
data: ## Prepara o dataset para o RecBole (Global Temporal Leave-One-Out)
	@echo "[INFO] Starting data preparation pipeline..."
	python src/pipeline/prepare_data.py --config config/project_config.yaml
	@echo "[INFO] Dataset preparation complete."

data-custom: ## Prepara dados com intervalo customizado (Requer START_DATE e END_DATE)
	@if [ -z "$(START_DATE)" ] || [ -z "$(END_DATE)" ]; then \
		echo "[ERROR] START_DATE and END_DATE arguments are required."; \
		exit 1; \
	fi
	python src/pipeline/prepare_data.py \
		--config config/project_config.yaml \
		--start-date $(START_DATE) \
		--end-date $(END_DATE)

# -----------------------------------------------------------------------------
##@ Execução de Benchmark
# -----------------------------------------------------------------------------
benchmark: ## Executa modelos. Opcional: MODELS='...' DATASET='...'
	@MODELS_ARG=$(or $(MODELS),all); \
	DATASET_ARG=$(or $(DATASET),realestate); \
	echo "[INFO] Starting benchmark execution (Models: $$MODELS_ARG | Dataset: $$DATASET_ARG)..."; \
	python src/run_benchmark.py \
		--models $$MODELS_ARG \
		--dataset $$DATASET_ARG \
		--config config/project_config.yaml

benchmark-neurais: ## Executa apenas modelos baseados em Redes Neurais (GRU4Rec, NARM, etc.)
	@echo "[INFO] Running neural models benchmark..."
	python src/run_benchmark.py --models neurais --config config/project_config.yaml

benchmark-baselines: ## Executa apenas modelos Baseline (Random, POP, etc.)
	@echo "[INFO] Running baselines benchmark..."
	python src/run_benchmark.py --models baselines --config config/project_config.yaml

benchmark-factor: ## Executa apenas modelos de Fatoração (FPMC, FOSSIL)
	@echo "[INFO] Running factorization models benchmark..."
	python src/run_benchmark.py --models factorization --config config/project_config.yaml

benchmark-quick: ## Executa teste rápido (GRU4Rec) para validação de sanidade
	@echo "[INFO] Running quick sanity check (GRU4Rec)..."
	python src/run_benchmark.py --models GRU4Rec --config config/project_config.yaml

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
test: ## Executa a suíte de testes unitários
	pytest tests/ -v

lint: ## Executa verificação de estilo (flake8)
	flake8 src/ --max-line-length=100 --ignore=E501,W503

format: ## Formata o código fonte (black)
	black src/ --line-length=100