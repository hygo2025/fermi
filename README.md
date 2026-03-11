# Session-Based Recommendation Benchmark

## Requisitos

- Python 3.9+
- Java 11+
- 16GB+ RAM

## Instalação

```bash
pip install -e .
```

## Configuração

Edite `config/project_config.yaml` com os caminhos dos dados:

```yaml
raw_data:
  events_processed_path: /path/to/events
  listings_processed_path: /path/to/listings
  
data_preparation:
  start_date: "2024-05-01"
  end_date: "2024-05-30"
  min_session_length: 2
  max_session_length: 50
  min_item_freq: 5
```

## Uso

### 1. Preparar Dataset

```bash
make data
```

### 2. Executar Benchmark

```bash
make benchmark
make benchmark MODELS=GRU4Rec
```

## Comandos Úteis

```bash
make help          # Ver todos comandos
make clean         # Limpar cache e logs
make format        # Formatar código
```