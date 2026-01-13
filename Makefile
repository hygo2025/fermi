.DEFAULT_GOAL := help
.PHONY: help install data benchmark clean clean-all test lint format tune tune-all tune-smoke

TUNABLE_MODELS := GRU4Rec NARM STAMP SASRec FPMC FOSSIL BERT4Rec SRGNN Caser GCSAN

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
benchmark: ## Executa modelos. Opcional: MODEL=... ou MODELS='...' DATASET='...'
	@if [ -n "$(MODEL)" ]; then \
		echo "[INFO] Running benchmark for $(MODEL)"; \
		python src/run_benchmark.py --model $(MODEL); \
	else \
		@MODELS_ARG=$(or $(MODELS),all); \
		DATASET_ARG=$(or $(DATASET),); \
		echo "[INFO] Starting benchmark execution (Models: $$MODELS_ARG)..."; \
		if [ -n "$$DATASET_ARG" ]; then \
			python src/run_benchmark.py --models $$MODELS_ARG --dataset $$DATASET_ARG; \
		else \
			python src/run_benchmark.py --models $$MODELS_ARG; \
		fi; \
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

tune-all: tune ## Alias para 'make tune' (roda todos os modelos)

# -----------------------------------------------------------------------------
##@ Hyperparameter Tuning - Smoke Test
# -----------------------------------------------------------------------------
tune-smoke: ## Executa um trial rápido (smoke test) para cada modelo tunável
	@DATASET_ARG="$(if $(DATASET),--dataset $(DATASET),)"; \
	ALGO_ARG="$(if $(ALGO),--algo $(ALGO),)"; \
	MAX_EVALS_ARG="$(if $(MAX_EVALS),--max-evals $(MAX_EVALS),--max-evals 1)"; \
	EARLY_STOP_ARG="$(if $(EARLY_STOP),--early-stop $(EARLY_STOP),--early-stop 1)"; \
	COOLDOWN_ARG="$(if $(COOLDOWN),--cooldown $(COOLDOWN),)"; \
	OUTPUT_ARG="$(if $(OUTPUT),--output $(OUTPUT),)"; \
	MODEL_COUNT=$$(printf "%s\n" "$(TUNABLE_MODELS)" | wc -w | awk '{print $$1}'); \
	echo "[INFO] Executando $$MODEL_COUNT smoke tests (modelos: $(TUNABLE_MODELS))"; \
	for model in $(TUNABLE_MODELS); do \
		echo "[INFO] >>> Smoke tuning $$model $$MAX_EVALS_ARG $$ALGO_ARG $$COOLDOWN_ARG"; \
		python src/hyperparameter_tuning.py --model $$model $$DATASET_ARG $$ALGO_ARG $$MAX_EVALS_ARG $$EARLY_STOP_ARG $$COOLDOWN_ARG $$OUTPUT_ARG || exit 1; \
	done


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
	rm -rf log/* 2>/dev/null || true

# -----------------------------------------------------------------------------
##@ Desenvolvimento
# -----------------------------------------------------------------------------
format: ## Formata o código fonte (black)
	black src/ --line-length=100