.PHONY: help install clean clean-results prepare-raw-data prepare-data prepare-simple-data convert-recbole convert-recbole-simple run-all aggregate-last run-neurais run-factorization run-baselines run-gru4rec run-narm run-stamp run-sasrec run-fpmc run-fossil run-random run-pop run-rpop run-spop run-gru4rec-parallel run-parallel analyze-model pipeline-complete

help:
	@echo "Fermi - Session-Based Recommendation Benchmark"
	@echo ""
	@echo "Pipeline Completo:"
	@echo "  make pipeline-complete    - [PIPELINE COMPLETO] Executar todo pipeline (prepare → run-all)"
	@echo ""
	@echo "Pipeline de Dados (em ordem):"
	@echo "  make prepare-raw-data     - [ETAPA 1] Processar dados brutos (listings + events)"
	@echo "  make prepare-data         - [ETAPA 2] Criar sliding window splits (PySpark)"
	@echo "  make prepare-simple-data  - [ETAPA 2 ALT] Preparar dados sem sliding windows (train/test simples)"
	@echo "  make convert-recbole      - [ETAPA 3] Converter para formato RecBole"
	@echo "  make convert-recbole-simple - [ETAPA 3 ALT] Converter dados simples para RecBole"
	@echo ""
	@echo "Experimentos (Sequencial):"
	@echo "  make run-all              - Executar todos modelos em todos slices (1 por vez)"
	@echo ""
	@echo "Experimentos (Paralelo):"
	@echo "  make run-parallel         - Executar modelo em 2 slices paralelos (com checkpoints)"
	@echo "  make run-gru4rec-parallel - Executar GRU4Rec em 2 slices paralelos"
	@echo ""
	@echo "Modelos por Categoria:"
	@echo "  make run-neurais          - Executar todos modelos neurais (GRU4Rec, NARM, STAMP, SASRec)"
	@echo "  make run-factorization    - Executar todos modelos de fatoração (FPMC, FOSSIL)"
	@echo "  make run-baselines        - Executar todos baselines (Random, POP, RPOP, SPOP)"
	@echo ""
	@echo "Modelos Neurais:"
	@echo "  make run-gru4rec          - Executar apenas GRU4Rec em todos slices"
	@echo "  make run-narm             - Executar apenas NARM em todos slices"
	@echo "  make run-stamp            - Executar apenas STAMP em todos slices"
	@echo "  make run-sasrec           - Executar apenas SASRec em todos slices"
	@echo ""
	@echo "Modelos de Fatoração:"
	@echo "  make run-fpmc             - Executar apenas FPMC em todos slices"
	@echo "  make run-fossil           - Executar apenas FOSSIL em todos slices"
	@echo ""
	@echo "Baselines (Individuais):"
	@echo "  make run-random           - Executar Random em todos slices"
	@echo "  make run-pop              - Executar POP em todos slices"
	@echo "  make run-rpop             - Executar RPOP em todos slices"
	@echo "  make run-spop             - Executar SPOP em todos slices"
	@echo ""
	@echo "Analise:"
	@echo "  make analyze-model        - Analisar recomendacoes (requer MODEL_PATH e opcionalmente SLICE)"
	@echo ""
	@echo "Resultados:"
	@echo "  make aggregate-last       - Agregar último resultado"
	@echo ""
	@echo "Utilidades:"
	@echo "  make install              - Instalar dependências"
	@echo "  make clean                - Limpar cache e temp files"
	@echo "  make clean-results        - Limpar resultados e logs"

install:
	pip install -e .

# Pipeline de dados
prepare-raw-data:
	@echo "[ETAPA 1/3] Processando dados brutos (listings + events)..."
	@echo "Isso pode levar alguns minutos..."
	export $(shell cat .env) && python src/data_preparation/prepare_raw_data.py

prepare-data:
	@echo "[ETAPA 2/3] Criando sliding window splits..."
	python src/preprocessing/sliding_window_pipeline.py \
		--input /home/hygo2025/Documents/data/processed_data/enriched_events \
		--output outputs/data/sliding_window \
		--start-date 2024-03-01 \
		--n-days 30

prepare-simple-data:
	@echo "[ETAPA 2/3 - SIMPLES] Preparando dados sem sliding windows..."
	python src/preprocessing/simple_data_pipeline.py \
		--input /home/hygo2025/Documents/data/processed_data/enriched_events \
		--output outputs/data/simple_data \
		--start-date 2024-03-01 \
		--n-days 30 \
		--train-ratio 0.8

convert-recbole:
	@echo "[ETAPA 3/3] Convertendo para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input outputs/data/sliding_window \
		--output outputs/data/recbole

convert-recbole-simple:
	@echo "[ETAPA 3/3] Convertendo dados simples para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input outputs/data/simple_data \
		--output outputs/data/recbole_simple

convert-recbole-simple:
	@echo "[ETAPA 3/3] Convertendo para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input outputs/data/simple_data \
		--output outputs/data/recbole_simple

# Experimentos - Sequencial
run-all:
	@echo "Executando todos experimentos (slices em paralelo por modelo)..."
	@chmod +x scripts/run_all_experiments.sh
	@./scripts/run_all_experiments.sh

# Experimentos - Paralelo (2 slices por vez)
run-parallel:
	@echo "Uso: make run-parallel MODEL=<modelo>"
	@echo "Exemplo: make run-parallel MODEL=GRU4Rec"
	@if [ -z "$(MODEL)" ]; then \
		echo "ERRO: Especifique o modelo com MODEL=<nome>"; \
		echo "Modelos disponíveis: GRU4Rec, NARM, STAMP, SASRec, FPMC, FOSSIL"; \
		exit 1; \
	fi
	@chmod +x scripts/run_parallel_gpu.sh
	@./scripts/run_parallel_gpu.sh $(MODEL) yes

run-gru4rec-parallel:
	@echo "Executando GRU4Rec em 2 slices paralelos (com checkpoints)..."
	@chmod +x scripts/run_parallel_gpu.sh
	@./scripts/run_parallel_gpu.sh GRU4Rec yes

# Neurais
run-gru4rec-quick:
	@echo "Executando GRU4Rec em 1 slice (slice1)..."
	python src/run_experiments.py --models GRU4Rec --slices 1 --save-checkpoints

run-gru4rec:
	@echo "Executando GRU4Rec em todos os slices (sequencial)..."
	python src/run_experiments.py --models GRU4Rec --all-slices --save-checkpoints

run-narm:
	@echo "Executando NARM em todos os slices (sequencial)..."
	python src/run_experiments.py --models NARM --all-slices

run-stamp:
	@echo "Executando STAMP em todos os slices (sequencial)..."
	python src/run_experiments.py --models STAMP --all-slices

run-sasrec:
	@echo "Executando SASRec em todos os slices (sequencial)..."
	python src/run_experiments.py --models SASRec --all-slices

# Baselines
run-random:
	@echo "Executando Random em todos os slices (sequencial)..."
	python src/run_experiments.py --models Random --all-slices

run-pop:
	@echo "Executando POP em todos os slices (sequencial)..."
	python src/run_experiments.py --models POP --all-slices

run-rpop:
	@echo "Executando RPOP em todos os slices (sequencial)..."
	python src/run_experiments.py --models RPOP --all-slices

run-spop:
	@echo "Executando SPOP em todos os slices (sequencial)..."
	python src/run_experiments.py --models SPOP --all-slices

# Factorization Models
run-fpmc:
	@echo "Executando FPMC em todos os slices (sequencial)..."
	python src/run_experiments.py --models FPMC --all-slices

run-fossil:
	@echo "Executando FOSSIL em todos os slices (sequencial)..."
	python src/run_experiments.py --models FOSSIL --all-slices


# Agregação manual do último resultado
aggregate-last:
	@echo "Agregando último resultado..."
	@LAST_DIR=$$(ls -td outputs/results/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_DIR" ]; then \
		echo "ERROR: Nenhum resultado encontrado em outputs/results/"; \
		exit 1; \
	fi; \
	echo "Processando: $$LAST_DIR"; \
	python src/aggregate_results.py \
		--input "$${LAST_DIR%/}" \
		--output "$${LAST_DIR%/}/aggregated_results.csv"

# Analise de recomendacoes
# Uso: make analyze-model MODEL_PATH=outputs/saved/GRU4Rec.pth [NUM_SESSIONS=5] [TOP_K=10] [SLICE=slice2]
analyze-model:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "ERROR: MODEL_PATH nao especificado"; \
		echo "Uso: make analyze-model MODEL_PATH=outputs/saved/GRU4Rec.pth"; \
		exit 1; \
	fi
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "ERROR: Modelo nao encontrado: $(MODEL_PATH)"; \
		exit 1; \
	fi
	@SLICE=$(or $(SLICE),slice1); \
	OUTPUT_DIR="outputs/analysis/$$(basename $(MODEL_PATH) .pth)"; \
	echo "Analisando modelo: $(MODEL_PATH)"; \
	echo "Slice: $$SLICE"; \
	echo "Output: $$OUTPUT_DIR"; \
	python src/exploration/analyze_recommendations.py \
		--model_path $(MODEL_PATH) \
		--features_path /home/hygo2025/Documents/data/processed_data/listings/part-00000-147c4e9e-f355-4a0f-92b2-9701e8657b41-c000.snappy.parquet \
		--test_data_path outputs/data/recbole/realestate_$$SLICE/realestate_$$SLICE.test.inter \
		--num_sessions $(or $(NUM_SESSIONS),5) \
		--top_k $(or $(TOP_K),10) \
		--output_dir $$OUTPUT_DIR

clean-results:
	rm -rf outputs/results/* 2>/dev/null || true
	rm -rf outputs/logs/* 2>/dev/null || true
	rm -rf outputs/saved/* 2>/dev/null || true
	rm -rf log_tensorboard/* 2>/dev/null || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true


# Grouped model execution
run-neurais:
	@echo "Executando todos modelos neurais em todos os slices..."
	@./scripts/run_all_experiments.sh GRU4Rec NARM STAMP SASRec

run-baselines:
	@echo "Executando todos baselines em todos os slices (sequencial)..."
	python src/run_experiments.py --models Random POP RPOP SPOP --all-slices

run-factorization:
	@echo "Executando todos modelos de fatoracao em todos os slices..."
	@./scripts/run_all_experiments.sh FPMC FOSSIL

# Pipeline completo - executar tudo em sequencia
pipeline-complete:
	@echo "========================================"
	@echo "PIPELINE COMPLETO - Fermi Benchmark"
	@echo "========================================"
	@echo ""
	@echo "Este comando executará em sequência:"
	@echo "  1. Processamento de dados brutos"
	@echo "  2. Criação de sliding window splits"
	@echo "  3. Conversão para formato RecBole"
	@echo "  4. Execução de todos os modelos"
	@echo ""
	@echo "Tempo estimado: 40-60 horas (com early stopping)"
	@echo ""
	@read -p "Deseja continuar? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "[1/4] Processando dados brutos..."
	@$(MAKE) prepare-raw-data
	@echo ""
	@echo "[2/4] Criando sliding window splits..."
	@$(MAKE) prepare-data
	@echo ""
	@echo "[3/4] Convertendo para RecBole..."
	@$(MAKE) convert-recbole
	@echo ""
	@echo "[4/4] Executando todos os modelos..."
	@$(MAKE) run-all
	@echo ""
	@echo "========================================"
	@echo "PIPELINE COMPLETO - CONCLUÍDO!"
	@echo "========================================"
	@echo "Resultados disponíveis em: outputs/results/"
