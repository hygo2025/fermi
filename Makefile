.PHONY: help install clean clean-results prepare-data convert-recbole run-all aggregate-last run-gru4rec run-narm run-stamp run-sasrec run-fpmc run-fossil run-baselines run-random run-pop run-rpop run-spop

help:
	@echo "Fermi - Session-Based Recommendation Benchmark"
	@echo ""
	@echo "Pipeline de Dados:"
	@echo "  make prepare-data         - Criar sliding window splits (PySpark)"
	@echo "  make convert-recbole      - Converter para formato RecBole"
	@echo ""
	@echo "Experimentos (Sequencial):"
	@echo "  make run-all              - Executar todos modelos em todos slices (1 por vez)"
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
	@echo "Baselines (Sequencial):"
	@echo "  make run-baselines        - Executar todos baselines em todos slices"
	@echo "  make run-random           - Executar Random em todos slices"
	@echo "  make run-pop              - Executar POP em todos slices"
	@echo "  make run-rpop             - Executar RPOP em todos slices"
	@echo "  make run-spop             - Executar SPOP em todos slices"
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
prepare-data:
	@echo "Criando sliding window splits..."
	python src/preprocessing/sliding_window_pipeline.py \
		--input /home/hygo2025/Documents/data/processed_data/enriched_events \
		--output outputs/data/sliding_window \
		--start-date 2024-03-01 \
		--n-days 30

convert-recbole:
	@echo "Convertendo para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input outputs/data/sliding_window \
		--output outputs/data/recbole

# Experimentos - Sequencial
run-all:
	@echo "Executando todos experimentos (slices em paralelo por modelo)..."
	@chmod +x scripts/run_all_experiments.sh
	@./scripts/run_all_experiments.sh

run-gru4rec:
	@echo "Executando GRU4Rec em todos os slices (sequencial)..."
	python src/run_experiments.py --models GRU4Rec --all-slices

run-narm:
	@echo "Executando NARM em todos os slices (sequencial)..."
	python src/run_experiments.py --models NARM --all-slices

run-stamp:
	@echo "Executando STAMP em todos os slices (sequencial)..."
	python src/run_experiments.py --models STAMP --all-slices

run-sasrec:
	@echo "Executando SASRec em todos os slices (sequencial)..."
	python src/run_experiments.py --models SASRec --all-slices

# Baselines - Sequencial
run-baselines:
	@echo "Executando todos baselines em todos os slices (sequencial)..."
	python src/run_experiments.py --models Random POP RPOP SPOP --all-slices

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

# Factorization Models - Sequencial
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

clean-results:
	rm -rf outputs/results/* 2>/dev/null || true
	rm -rf outputs/logs/* 2>/dev/null || true
	rm -rf outputs/saved/* 2>/dev/null || true
	rm -rf log_tensorboard/* 2>/dev/null || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

