# Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessões para plataforma de classificados imobiliários.

## Requisitos

- Python 3.9+
- Java 11+ (para PySpark)
- 16GB+ RAM
- GPU recomendada (treinamento de modelos neurais)

## Instalação

```bash
git clone <repo-url>
cd fermi
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
# Todos os modelos
make benchmark

# Modelo específico
make benchmark MODELS=GRU4Rec
```

### 3. Hyperparameter Tuning

```bash
# Modelo específico
make tune MODEL=GRU4Rec MAX_EVALS=20

# Todos os modelos
make tune
```

### 4. API Web

```bash
make api MODEL=GRU4Rec
```

Acesse: `http://localhost:8000`

## Modelos Disponíveis

**Neurais**: GRU4Rec, NARM, STAMP, SASRec, Caser, SRGNN, GCSAN, BERT4Rec  
**Fatorização**: FPMC, FOSSIL  
**KNN**: ItemKNN  
**Baselines**: BPR, Pop, Random

## Comandos Úteis

```bash
make help          # Ver todos comandos
make clean         # Limpar cache e logs
make format        # Formatar código
```
MODELS="  GCSAN " make benchmark
MODELS=" SRGNN" make benchmark
MODELS=" " make benchmark

BERT4Rec SASRec  Caser

05-06
MODELS="BPR" make benchmark       - Rodando 16 - Baixado
MODELS="ItemKNN" make benchmark   - Rodando 16 - Baixado
MODELS="Pop" make benchmark       - Rodando 16 - Baixado
MODELS="Random" make benchmark    - Rodando 16 - Baixado
MODELS="FOSSIL" make benchmark    - Rodando 16 - Baixado
MODELS="FPMC" make benchmark      - Rodando 16 - Baixado
MODELS="TransRec" make benchmark  - Rodando 17 - Baixado
MODELS="GRU4Rec" make benchmark   - Rodando 16 - Baixado
MODELS="NARM" make benchmark      - Rodando 16 - Baixado
MODELS="LightGCN" make benchmark  - Rodando 16 - Baixado
MODELS="SASRec" make benchmark    - Rodando 17 - Baixado
MODELS="NextItNet" make benchmark - Rodando 16 - Baixado
MODELS="BERT4Rec" make benchmark  - Rodando 17 - Baixado
MODELS="Caser" make benchmark     - Rodando 16 

06-07
MODELS="BPR" make benchmark       - Rodando 54 - Baixado
MODELS="ItemKNN" make benchmark   - Rodando 54 - Baixado
MODELS="Pop" make benchmark       - Rodando 54 - Baixado
MODELS="Random" make benchmark    - Rodando 54 - Baixado
MODELS="FOSSIL" make benchmark    - Rodando 54 - Baixado
MODELS="FPMC" make benchmark      - Rodando 54 - Baixado
MODELS="TransRec" make benchmark  - Rodando 17 - Baixado
MODELS="GRU4Rec" make benchmark   - Rodando 54 - Baixado
MODELS="NARM" make benchmark      - Rodando 54 - Baixado
MODELS="LightGCN" make benchmark  - Rodando 17 - Baixado
MODELS="SASRec" make benchmark    - Rodando 17
MODELS="NextItNet" make benchmark - 
MODELS="BERT4Rec" make benchmark  - 
MODELS="Caser" make benchmark     -  Rodando 54


POLL_SECONDS=60 MODELS="Pop Random FOSSIL FPMC BPR NARM GRU4Rec ItemKNN" make benchmark-queue
POLL_SECONDS=60 MODELS="TransRec LightGCN SASRec NextItNet BERT4Rec Caser" make benchmark-queue