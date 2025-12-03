.PHONY: help install clean prepare-data convert-recbole run-all aggregate-results run-gru4rec run-narm run-stamp run-sasrec

help:
	@echo "Fermi - Session-Based Recommendation Benchmark"
	@echo ""
	@echo "Pipeline de Dados:"
	@echo "  make prepare-data      - Criar sliding window splits (PySpark)"
	@echo "  make convert-recbole   - Converter para formato RecBole"
	@echo ""
	@echo "Experimentos:"
	@echo "  make run-all           - Executar todos modelos em todos slices"
	@echo "  make run-gru4rec       - Executar apenas GRU4Rec em todos slices"
	@echo "  make run-narm          - Executar apenas NARM em todos slices"
	@echo "  make run-stamp         - Executar apenas STAMP em todos slices"
	@echo "  make run-sasrec        - Executar apenas SASRec em todos slices"
	@echo "  make aggregate-results - Agregar resultados (média ± std)"
	@echo ""
	@echo "Utilidades:"
	@echo "  make install           - Instalar dependências"
	@echo "  make clean             - Limpar cache e temp files"

install:
	pip install -r requirements.txt

# Pipeline de dados
prepare-data:
	@echo "Criando sliding window splits..."
	python src/preprocessing/sliding_window_pipeline.py \
		--input /home/hygo2025/Documents/data/processed_data/enriched_events \
		--output data/sliding_window \
		--start-date 2024-03-01 \
		--n-days 30

convert-recbole:
	@echo "Convertendo para formato RecBole..."
	python src/preprocessing/recbole_converter.py \
		--input data/sliding_window \
		--output recbole_data

# Experimentos
run-all:
	@echo "Executando todos experimentos..."
	python src/run_experiments.py --all-slices

run-gru4rec:
	@echo "Executando GRU4Rec em todos os slices..."
	python src/run_experiments.py --models GRU4Rec --all-slices

run-narm:
	@echo "Executando NARM em todos os slices..."
	python src/run_experiments.py --models NARM --all-slices

run-stamp:
	@echo "Executando STAMP em todos os slices..."
	python src/run_experiments.py --models STAMP --all-slices

run-sasrec:
	@echo "Executando SASRec em todos os slices..."
	python src/run_experiments.py --models SASRec --all-slices

aggregate-results:
	@echo "Agregando resultados..."
	python src/aggregate_results.py \
		--input results \
		--output results/aggregated_results.csv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
