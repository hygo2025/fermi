help:
	@echo "════════════════════════════════════════════════════════════"
	@echo "Fermi - RecBole Benchmark"
	@echo "════════════════════════════════════════════════════════════"
	@echo "Data Setup:"
	@echo "  make prepare-data       # Convert session_rec_format to RecBole"
	@echo ""
	@echo "Neural Models:"
	@echo "  make test-gru4rec    make test-narm"
	@echo "  make test-stamp      make test-srgnn"
	@echo ""
	@echo "KNN Models:"
	@echo "  make test-itemknn    make test-sknn"
	@echo ""
	@echo "Baselines:"
	@echo "  make test-pop"
	@echo ""
	@echo "Run All:"
	@echo "  make run-all"

# Convert existing session_rec_format data to RecBole format
prepare-data:
	@echo "Converting session_rec_format to RecBole format..."
	. .venv/bin/activate && python src/data/convert_to_recbole.py
	@echo "✅ Data preparation complete!"

test-gru4rec:
	@mkdir -p logs/neural
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/neural/gru4rec.yaml

test-narm:
	@mkdir -p logs/neural
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/neural/narm.yaml

test-stamp:
	@mkdir -p logs/neural
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/neural/stamp.yaml

test-srgnn:
	@mkdir -p logs/neural
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/neural/srgnn.yaml

test-itemknn:
	@mkdir -p logs/knn
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/knn/itemknn.yaml

test-sknn:
	@mkdir -p logs/knn
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/knn/sknn.yaml

test-pop:
	@mkdir -p logs/baselines
	. .venv/bin/activate && python src/run_recbole.py -c src/configs/baselines/pop.yaml

run-all: test-gru4rec test-narm test-stamp test-srgnn test-itemknn test-sknn test-pop
	@echo "All 7 models complete!"
