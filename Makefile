.PHONY: help install clean clean-results prepare-raw-data prepare-data convert-recbole run-all aggregate-last run-neurais run-factorization run-baselines run-gru4rec run-narm run-stamp run-sasrec run-fpmc run-fossil run-random run-pop run-rpop run-spop run-gru4rec-parallel run-parallel analyze-model

help:
	@echo "Fermi - Session-Based Recommendation Benchmark"
	@echo ""
	@echo "Pipeline de Dados (em ordem):"
	@echo "  make prepare-raw-data     - [ETAPA 1] Processar dados brutos (listings + events)"
	@echo "  make prepare-data         - [ETAPA 2] Criar sliding window splits (PySpark)"
	@echo "  make convert-recbole      - [ETAPA 3] Converter para formato RecBole"
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

convert-recbole:
	@echo "[ETAPA 3/3] Convertendo para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input outputs/data/sliding_window \
		--output outputs/data/recbole

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
		--features_path /home/hygo2025/Documents/data/processed_data/listings/part-00000-bf98a429-f6d4-47a3-a1c0-94dc622ae21a-c000.snappy.parquet \
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
