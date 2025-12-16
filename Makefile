.PHONY: help install clean clean-results prepare-data convert-recbole run-all aggregate-results run-gru4rec run-narm run-stamp run-sasrec run-gru4rec-parallel run-narm-parallel run-stamp-parallel run-sasrec-parallel run-baselines run-random run-pop run-rpop run-spop

help:
	@echo "Fermi - Session-Based Recommendation Benchmark"
	@echo ""
	@echo "Pipeline de Dados:"
	@echo "  make prepare-data         - Criar sliding window splits (PySpark)"
	@echo "  make convert-recbole      - Converter para formato RecBole"
	@echo ""
	@echo "Experimentos (Sequencial):"
	@echo "  make run-all              - Executar todos modelos em todos slices (1 por vez)"
	@echo "  make test-quick           - TESTE RÁPIDO: GRU4Rec em 2 slices (~5-10min)"
	@echo "  make run-gru4rec          - Executar apenas GRU4Rec em todos slices"
	@echo "  make run-narm             - Executar apenas NARM em todos slices"
	@echo "  make run-stamp            - Executar apenas STAMP em todos slices"
	@echo "  make run-sasrec           - Executar apenas SASRec em todos slices"
	@echo ""
	@echo "Baselines (Sequencial):"
	@echo "  make run-baselines        - Executar todos baselines em todos slices"
	@echo "  make run-random           - Executar Random em todos slices"
	@echo "  make run-pop              - Executar POP em todos slices"
	@echo "  make run-rpop             - Executar RPOP em todos slices"
	@echo "  make run-spop             - Executar SPOP em todos slices"
	@echo ""
	@echo "Experimentos (Paralelo):"
	@echo "  make run-gru4rec-parallel      - Executar GRU4Rec com slices 1,2,3 em paralelo"
	@echo "  make run-narm-parallel         - Executar NARM com slices 1,2,3 em paralelo"
	@echo "  make run-stamp-parallel        - Executar STAMP com slices 1,2,3 em paralelo"
	@echo "  make run-sasrec-parallel       - Executar SASRec com slices 1,2,3 em paralelo"
	@echo ""
	@echo "Resultados:"
	@echo "  make aggregate-results    - Agregar resultados (média ± std)"
	@echo ""
	@echo "Utilidades:"
	@echo "  make install              - Instalar dependências"
	@echo "  make clean                - Limpar cache e temp files"

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

# Experimentos - Paralelo (3 slices simultâneos)
run-gru4rec-parallel:
	@echo "Executando GRU4Rec com 3 slices em paralelo..."
	@chmod +x scripts/run_parallel.sh
	@./scripts/run_parallel.sh GRU4Rec "1 2 3"
	@echo "Slices 1,2,3 iniciados. Aguarde a conclusão com 'wait' ou monitore com 'nvidia-smi'"

run-narm-parallel:
	@echo "Executando NARM com 3 slices em paralelo..."
	@chmod +x scripts/run_parallel.sh
	@./scripts/run_parallel.sh NARM "1 2 3"
	@echo "Slices 1,2,3 iniciados. Aguarde a conclusão com 'wait' ou monitore com 'nvidia-smi'"

run-stamp-parallel:
	@echo "Executando STAMP com 3 slices em paralelo..."
	@chmod +x scripts/run_parallel.sh
	@./scripts/run_parallel.sh STAMP "1 2 3"
	@echo "Slices 1,2,3 iniciados. Aguarde a conclusão com 'wait' ou monitore com 'nvidia-smi'"

run-sasrec-parallel:
	@echo "Executando SASRec com 3 slices em paralelo..."
	@chmod +x scripts/run_parallel.sh
	@./scripts/run_parallel.sh SASRec "1 2 3"
	@echo "Slices 1,2,3 iniciados. Aguarde a conclusão com 'wait' ou monitore com 'nvidia-smi'"

aggregate-results:
	@echo "Este comando não é mais necessário!"
	@echo "A agregação é feita automaticamente ao final de 'make run-all'"
	@echo "Os resultados estão em: outputs/results/YYYYMMDD_HHMMSS/"

test-quick:
	@echo "Executando teste rápido (GRU4Rec em slices 1-2)..."
	@echo "Configuração: 10 épocas, batch_size=512"
	python src/run_experiments.py \
		--models GRU4Rec \
		--slices 1 2 \
		--epochs 2 \
		--batch-size 512

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
