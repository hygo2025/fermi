# Fermi - Session-Based Recommendation Benchmark

Benchmark implementation based on Domingues et al. (2025) methodology for real estate session-based recommendations.

**Reference Paper:**  
"A large scale benchmark for session-based recommendations on the legal domain"  
Domingues et al. (2025) - Artificial Intelligence and Law

## Overview

Este projeto implementa e avalia múltiplos modelos de recomendação baseados em sessão usando dados reais de interações de usuários com listagens de imóveis. O objetivo é predizer o próximo imóvel que um usuário vai interagir baseado na sequência de interações da sessão atual.

### Diferenças do Artigo Original

| Aspecto   | Artigo (Jusbrasil)             | Nosso Benchmark                    |
|-----------|--------------------------------|------------------------------------|
| Domínio   | Legal (documentos jurídicos)   | Imobiliário (listings)             |
| Itens     | Documentos                     | Imóveis                            |
| Framework | session-rec                    | session-rec (mesmo)                |
| Métricas  | Recall@K, MRR@K, Coverage      | Recall@K, MRR@K, Coverage (mesmas) |

## Estrutura do Projeto

```
fermi/
├── src/
│   ├── run_session_rec.py           # Main benchmark runner
│   └── configs/
│       └── pattern_mining/          # AR, Markov, SR
│
├── data/
│   ├── prepare_dataset.py           # Spark-based data preparation
│   └── convert_to_session_rec.py    # Format conversion
│
├── session-rec-lib/                 # Framework (git submodule)
│   ├── algorithms/                  # Model implementations
│   └── evaluation/                  # Metrics and evaluation
│
├── scripts/
│   └── install.sh                   # Automated installation
│
├── utils/
│   └── spark_session.py             # Spark configuration
│
├── .env                             # Environment variables
├── requirements.txt
├── Makefile
└── README.md
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
BASE_PATH=/home/hygo2025/Documents/data
JAVA_HOME=/opt/jdk/amazon-corretto-21
PYTHONUNBUFFERED=1
```

Note: `BASE_PATH` points to where your raw data is stored.

## Quick Start

### 1. Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/hygo2025/fermi.git
cd fermi

# Configure environment variables
cp .env.example .env
# Edit .env with your paths (BASE_PATH, JAVA_HOME)

# Complete setup (Python 3.9, venv, dependencies, session-rec)
make setup

# Verify installation
make status
```

Setup steps:
- Checks/installs Python 3.9
- Creates virtual environment (.venv)
- Installs all dependencies from requirements.txt
- Initializes session-rec-lib submodule
- Configures PYTHONPATH

### 2. Prepare Dataset

```bash
# For quick testing (2 days: 2024-04-01 to 2024-04-02)
make prepare-data START_DATE=2024-04-01 END_DATE=2024-04-02

# For full benchmark (30 days with 5 slices: 2024-04-01 to 2024-04-30)
# Following paper methodology: 5 slices of 6 days (5 train + 1 test)
make prepare-data START_DATE=2024-04-01 END_DATE=2024-04-30

# Or manually
python data/prepare_dataset.py \
    --start-date 2024-04-01 \
    --end-date 2024-04-30 \
    --output ./session_rec_format/realestate
```

What is done:
- Loads events from BASE_PATH using PySpark
- Creates sessions (30min timeout)
- Removes short sessions (<2 events) and rare items (<5 occurrences)
- Temporal split: configured in YAML
- Saves in Parquet (fast!) in session-rec format

Expected output:
```
session_rec_format/realestate/
  ├── realestate_train_full.parquet
  └── realestate_test.parquet
```

### Evaluation Methodology (Following Paper)

Following Domingues et al. (2025), we use a **slicing protocol** for robust evaluation:

**30-day dataset split into 5 slices:**
- Each slice: 6 consecutive days
  - Train: 5 days
  - Test: 1 day
- Results averaged across all 5 slices

This approach:
- Tests model robustness across different time periods
- Maintains chronological order of interactions
- Provides statistical variance in results

**Example for 2024-04-01 to 2024-04-30:**
```
Slice 1: Train 2024-04-01 to 2024-04-05 → Test 2024-04-06
Slice 2: Train 2024-04-07 to 2024-04-11 → Test 2024-04-12
Slice 3: Train 2024-04-13 to 2024-04-17 → Test 2024-04-18
Slice 4: Train 2024-04-19 to 2024-04-23 → Test 2024-04-24
Slice 5: Train 2024-04-25 to 2024-04-29 → Test 2024-04-30
```

Configured in YAML as:
```yaml
type: slices
data:
  slices: 5
  num_days_train: 5
  num_days_test: 1
```

### 3. Run Benchmark

Pattern Mining Models (from paper):
```bash
# Run individually
make test-ar      # Association Rules (AR)
make test-markov  # Markov Chain
make test-sr      # Sequential Rules (SR)

# Run all pattern mining in parallel (with logs)
make test-pattern-mining
# Logs written to: logs/ar.log, logs/markov.log, logs/sr.log
# Monitor with: tail -f logs/sr.log
```

KNN Models (from paper):
```bash
# Run individually
make test-iknn    # Item-KNN
make test-sknn    # Session-KNN
make test-vsknn   # Vector Multiplication Session-KNN
make test-stan    # Sequence and Time-aware Neighborhood
make test-vstan   # VSKNN + STAN

# Run all KNN models in parallel (with logs)
make test-knn
# Logs written to: logs/iknn.log, logs/sknn.log, logs/vsknn.log, logs/stan.log, logs/vstan.log
# Monitor with: tail -f logs/stan.log
```

Run all implemented models:
```bash
make run-all-baselines  # Runs Pattern Mining + KNN
```

Expected time (per slice):
- Data loading: ~2s (optimized Parquet)
- Fit SR: ~10s
- Evaluation: varies by model
- Total: 5 slices × (fit + eval) time

## Data

Raw data is located at `/home/hygo2025/Documents/data/processed_data/`:
- **events/** - User events (~25M events, 182 days)
- **listings/** - Property catalog (187k properties)

Preparation pipeline:
1. Filter events by period
2. Create sessions (30min inactivity timeout)
3. Remove short sessions (<2 events) and rare items (<5 occurrences)
4. Temporal split (last 2 days = test, rest = train)
5. Convert to session-rec format (Parquet)

## Framework: Session-Rec

We use **session-rec**, the same framework used in the original paper:

- **Fork Python 3.9+:** https://github.com/hygo2025/session-rec-3-9
- **Original:** https://github.com/rn5l/session-rec

### Applied Fixes in Fork

1. `time.clock()` to `time.perf_counter()` (removed in Python 3.8)
2. `yaml.load()` to `yaml.load(Loader=FullLoader)` (security)
3. `Pop.fit()` signature fix
4. Telegram notifications disabled
5. Parquet format support in data loader

### Why Session-Rec?

- Same framework as the paper (comparability)
- 20+ session-based models implemented
- Standardized metrics
- Established benchmark in literature

## Models (following Domingues et al. 2025)

### Implemented Models

#### Pattern Mining
- **ar** - Association Rules (co-occurrence, pruning=20)
- **markov** - First-order Markov Chain (pruning=20)
- **sr** - Sequential Rules (steps=10, weighting='div', pruning=20)

**Configuration**: All pattern mining models use the standard signature `fit(data, test=None)` and don't require wrappers.

#### Neural Networks (Deep Learning)
- **gru4rec** - GRU4Rec (Gated Recurrent Units, loss='top1-max', dropout=0.1, lr=0.08)
- **narm** - NARM (Neural Attentive Recommendation Machine, epochs=20, hidden_units=100)
- **stamp** - STAMP (Short-Term Attention Memory Priority, n_epochs=10, attention-based)
- **sgnn** - SGNN (Session-based Graph Neural Network, GNN with graph structure)

**Configuration**: Neural models are pre-implemented in session-rec-lib and don't require wrappers.

#### Nearest Neighbors (KNN)
- **iknn** - Item k-NN (cosine similarity on last item) - **⚠️ Requires wrapper**
- **sknn** - Session-based k-NN (whole session comparison) - **⚠️ Requires wrapper**
- **vsknn** - Vector Multiplication Session-based k-NN (with linear decay)
- **stan** - Sequence and Time-aware Neighborhood (position + recency + neighbor position)
- **vstan** - VSKNN + STAN fusion (all features combined with IDF weighting)

**Note**: IKNN and SKNN require wrappers in `src/models/knn/` due to incompatibilities with the session-rec framework. See [Wrappers Section](#wrappers) for details.

### Model Wrappers {#wrappers}

Two KNN models require wrappers to fix incompatibilities with session-rec:

#### 1. IKNN Wrapper (`src/models/knn/iknn.py`)
**Problem**: Original `fit(data)` signature incompatible with framework's `fit(train, test)` call.  
**Solution**: Wrapper accepts `test=None` parameter and ignores it.

```python
# src/models/knn/iknn.py
from algorithms.knn.iknn import ItemKNN as BaseItemKNN

class ItemKNN(BaseItemKNN):
    def fit(self, data, test=None):
        super().fit(data)  # Ignores test parameter
```

#### 2. SKNN Wrapper (`src/models/knn/sknn.py`)
**Problem**: `sessions_for_item()` returns `None` for unseen items, causing `TypeError: set | None`.  
**Solution**: Wrapper returns empty `set()` instead of `None`.

```python
# src/models/knn/sknn.py
from algorithms.knn.sknn import ContextKNN as BaseContextKNN

class ContextKNN(BaseContextKNN):
    def sessions_for_item(self, item_id):
        return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()
```

**Architecture**:
```
src/models/knn/          # Wrappers in project
  ├── iknn.py           # Fix: fit signature
  └── sknn.py           # Fix: sessions_for_item → None

session-rec-lib/algorithms/
  └── models -> ../../src/models  # Symlink (created by make install-benchmark)
```

**How it works**:
1. Config uses: `models.knn.sknn.ContextKNN`
2. Framework adds prefix: `algorithms.models.knn.sknn.ContextKNN`
3. Symlink resolves: `session-rec-lib/algorithms/models` → `src/models`
4. Wrapper is loaded and fixes the bug

**Setup**: Symlink is created automatically by `make install-benchmark`. For manual setup: `./scripts/setup_wrappers.sh`

### Future Work

#### Non-Personalized Baselines (Future Work)
- **pop** - Popularity (global item frequency)
- **random** - Random list (lower bound)
- **rpop** - Recent Popularity (last n days)
- **spop** - Session Popularity (frequency in session)

#### Matrix Factorization (Future Work)
- **fism** - Factored Item Similarity Models
- **fossil** - Factorized Personalized Markov Chains

#### Neural Networks (Future Work)
- **gru4rec** - Gated Recurrent Units for Recommendations
- **narm** - Neural Attentive Recommendation Machine
- **stamp** - Short-Term Attention Memory Priority
- **nextitnet** - Dilated Convolutions
- **sasrec** - Self-Attentive Sequential Recommendation

## Evaluation Metrics (following Domingues et al. 2025)

We follow exactly the methodology from the paper:

- **Recall@K** - Hit rate in top-K recommendations
- **MRR@K** - Mean Reciprocal Rank (position of correct item)

Configuration:
- K in {5, 10, 20} - Recommendation list sizes
- Evaluation: next-item prediction
- Protocol: Given 1 Predict 1 (each session event used as prediction point)

### Metric Interpretation

- **Recall@20 = 0.15**: In 15% of predictions, the correct item is in top-20
- **MRR@20 = 0.05**: On average, the correct item appears at position 20 (1/0.05)
- **MRR@20 > Recall@20**: Impossible (MRR <= Recall)

## Complete Pipeline

1. **Preparation:** Filter events, create sessions, temporal split
2. **Conversion:** Raw data to session-rec format (Parquet)
3. **Training:** Train models with training data
4. **Evaluation:** Next-item prediction on test sessions
5. **Analysis:** Compare metrics between models

## Useful Commands (Makefile)

### Installation
```bash
make install-benchmark  # Setup complete environment
make status            # Check installation status
```

### Data Preparation
```bash
make prepare-data      # Prepare dataset with Spark
make convert-data      # Convert to session-rec format
```

### Run Benchmarks
```bash
# Pattern mining models
make test-ar           # Association Rules
make test-markov       # Markov Chain
make test-sr           # Sequential Rules

# KNN models
make test-iknn         # Item-KNN
make test-sknn         # Session-KNN
make test-vsknn        # Vector Multiplication Session-KNN
make test-stan         # STAN
make test-vstan        # VSTAN

# Run all in parallel
make test-pattern-mining  # All pattern mining
make test-knn             # All KNN models
make run-all-baselines    # Everything
```

### Cleanup
```bash
make clean             # Remove results and processed data
make clean-all         # Full cleanup including session-rec
```

## Data Format

### Input (BASE_PATH)
```
/home/hygo2025/Documents/data/processed_data/
├── events/                    # ~25M events, 182 days
│   └── YYYY/MM/DD/*.parquet
└── listings/                  # 187k properties
    └── *.parquet
```

### Output (session-rec format)
```
session_rec_format/realestate/
├── realestate_train_full.parquet
└── realestate_test.parquet

Columns: SessionId, ItemId, Time
```

Parquet Schema:
- `SessionId`: int64 - Unique session ID
- `ItemId`: int32 - Property ID
- `Time`: timestamp - Unix timestamp of interaction

## Troubleshooting

### Common Issues

#### Error: `AttributeError: 'VMContextKNN' object has no attribute 'div_score_score'`
```
AttributeError: 'VMContextKNN' object has no attribute 'div_score_score'
```
**Cause**: VSKNN config has incorrect `weighting_score` parameter.  
**Solution**: Use `weighting_score: div` (not `div_score`). The code appends `_score` automatically.  
**Verify**: Check `src/configs/knn/vsknn.yml` uses `weighting_score: div`

**Available options**: `div`, `same`, `linear`, `log`, `quadratic`

#### Error: `fit() takes 2 positional arguments but 3 were given`
```
TypeError: fit() takes 2 positional arguments but 3 were given
```
**Cause**: IKNN model has incompatible `fit()` signature.  
**Solution**: Use wrapper. Config should use `models.knn.iknn.ItemKNN` (not `knn.iknn.ItemKNN`).  
**Verify**: Check `src/configs/knn/iknn.yml` uses `class: models.knn.iknn.ItemKNN`

#### Error: `TypeError: unsupported operand type(s) for |: 'set' and 'NoneType'`
```
TypeError: unsupported operand type(s) for |: 'set' and 'NoneType'
```
**Cause**: SKNN's `sessions_for_item()` returns `None` for unseen items.  
**Solution**: Use wrapper. Config should use `models.knn.sknn.ContextKNN`.  
**Verify**: Check `src/configs/knn/sknn.yml` uses `class: models.knn.sknn.ContextKNN`

#### Error: `ModuleNotFoundError: No module named 'algorithms.models'`
```
ModuleNotFoundError: No module named 'algorithms.models'
```
**Cause**: Symlink `session-rec-lib/algorithms/models` not created.  
**Solution**: Run `make install-benchmark` or `./scripts/setup_wrappers.sh`  
**Verify**: `ls -la session-rec-lib/algorithms/models` should show symlink

### Other Issues

#### Error: `time.clock()` not found
```
AttributeError: module 'time' has no attribute 'clock'
```
Cause: Python 3.8+ removed `time.clock()`  
Solution: Our fork already fixes this (`time.perf_counter()`). Run `make setup`

### Error: `fit() takes 2 positional arguments but 3 were given`
```
TypeError: fit() takes 2 positional arguments but 3 were given
```
Cause: Incompatible `fit()` method signature  
Solution: Already fixed in our fork. Make sure to use correct submodule

### Error: `predict_next() got an unexpected keyword argument 'timestamp'`
```
TypeError: predict_next() got an unexpected keyword argument 'timestamp'
```
Cause: POP model doesn't accept `timestamp` parameter  
Solution: Fixed in fork - adds `**kwargs` for compatibility

### Slow data loading (>20 minutes)
```
START load data
Loading train from: .../realestate_train_full.txt
[waiting...]
```
Cause: TSV/TXT is slow for 27M+ lines  
Solution: We now use Parquet (loads in ~2 seconds)

### Session-rec not found
```
ModuleNotFoundError: No module named 'evaluation'
```
Solution:
```bash
# PYTHONPATH is configured by make, but if needed:
export PYTHONPATH=/home/hygo2025/Development/projects/fermi/session-rec-lib:$PYTHONPATH
```

### Submodule not initialized
```
fatal: No url found for submodule path 'session-rec-lib'
```
Solution:
```bash
git submodule update --init --recursive
```

### PyArrow not installed
```
ModuleNotFoundError: No module named 'pyarrow'
```
Solution:
```bash
pip install pyarrow  # Already in requirements.txt
```

### Spark cannot find JAVA_HOME
```
Exception: Java gateway process exited before sending its port number
```
Solution: Configure JAVA_HOME in `.env`:
```bash
JAVA_HOME=/opt/jdk/amazon-corretto-21  # or your JDK path
```

## References

Main Paper:
```
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2025).
A large scale benchmark for session-based recommendations on the legal domain.
Artificial Intelligence and Law, 33, 43-78.
DOI: 10.1007/s10506-023-09378-3
```

Session-Rec Framework:
```
Ludewig, M., & Jannach, D. (2018).
Evaluation of session-based recommendation algorithms.
User Modeling and User-Adapted Interaction, 28(4-5), 331-390.
```

## License

This project is part of academic research.

---

Created: December 2, 2024  
Based on: Domingues et al. (2025)  
Framework: [session-rec](https://github.com/hygo2025/session-rec-3-9) (fork Python 3.9+)  
Last updated: December 2, 2024
