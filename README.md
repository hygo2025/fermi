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
│       ├── non_personalized/        # Random, POP, RPOP, SPOP
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

# For full benchmark (14 days: 2024-04-01 to 2024-04-14)
make prepare-data START_DATE=2024-04-01 END_DATE=2024-04-14

# Or manually
python data/prepare_dataset.py \
    --start-date 2024-04-01 \
    --end-date 2024-04-14 \
    --output ./session_rec_format/realestate
```

What is done:
- Loads ~30M events from BASE_PATH using PySpark
- Creates sessions (30min timeout)
- Removes short sessions (<2 events)
- Temporal split: last 2 days = test, rest = train
- Saves in Parquet (fast!) in session-rec format

Expected output:
```
session_rec_format/realestate/
  ├── realestate_train_full.parquet  (~27M events)
  └── realestate_test.parquet        (~3.6M events)
```

### 3. Run Benchmark

Non-Personalized Models (from paper):
```bash
# Run individually
make test-pop     # Popularity (POP)
make test-random  # Random (lower bound)
make test-rpop    # Recent Popularity
make test-spop    # Session Popularity

# Run all non-personalized
make test-non-personalized
```

Pattern Mining Models (from paper):
```bash
# Run individually
make test-ar      # Association Rules (AR)
make test-markov  # Markov Chain
make test-sr      # Sequential Rules (SR)

# Run all pattern mining
make test-pattern-mining
```

Run all baselines:
```bash
make run-all-baselines
```

Expected time:
- Data loading: ~2s (optimized Parquet)
- Fit POP: ~0.3s
- Evaluation: ~3h (3.6M events, event-by-event)

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

### Non-Personalized Baselines
- **pop** - Popularity (global item frequency)
- **random** - Random list (lower bound)
- **rpop** - Recent Popularity (last n days)
- **spop** - Session Popularity (frequency in session)

### Pattern Mining
- **ar** - Association Rules (co-occurrence)
- **markov** - First-order Markov Chain
- **sr** - Sequential Rules (with decay function)

### Nearest Neighbors (Future Work)
- **iknn** - Item k-NN (cosine similarity)
- **sknn** - Session-based k-NN
- **vsknn** - Vector Multiplication Session-based k-NN
- **stan** - Sequence and Time-aware Neighborhood
- **sfsknn** - Session-based Factorized k-NN

### Matrix Factorization (Future Work)
- **fism** - Factored Item Similarity Models
- **fossil** - Factorized Personalized Markov Chains

### Neural Networks (Future Work)
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
# Individual non-personalized models
make test-pop          # Popularity
make test-recent       # Most recent items
make test-random       # Random baseline
make test-rpop         # Recent popularity
make test-spop         # Session popularity

# Run all non-personalized at once
make test-non-personalized
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

### Error: `time.clock()` not found
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
