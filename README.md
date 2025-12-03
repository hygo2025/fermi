# Fermi - Session-Based Recommendation Benchmark

Replicação da metodologia de session-based recommendation do paper Domingues et al. (2024) aplicada ao domínio imobiliário.

**Referência:**  
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2024).  
A large scale benchmark for session-based recommendations in the legal domain.  
Artificial Intelligence and Law, 33, 43-78. DOI: 10.1007/s10506-023-09378-3

## Metodologia - Sliding Window Protocol

Protocolo de sliding window temporal seguindo Domingues et al. (2024):

**Configuração:**
- Período: 30 dias (2024-03-01 a 2024-03-30)
- Slices: 5 splits temporais
- Cada slice: 5 dias treino + 1 dia teste

**Filtragem:**
- Sessões: 2-50 interações
- Itens: mínimo 5 ocorrências
- Eventos: apenas interações reais (ListingRendered, GalleryClicked, etc)

**Estatísticas:**

| Slice | Train Period | Test Period | Train Events | Test Events | Train Sessions | Test Sessions |
|-------|-------------|-------------|--------------|-------------|----------------|---------------|
| 1     | Mar 01-05   | Mar 06      | 763,159      | 153,674     | 43,232         | 8,763         |
| 2     | Mar 07-11   | Mar 12      | 691,302      | 164,295     | 39,251         | 9,379         |
| 3     | Mar 13-17   | Mar 18      | 748,690      | 199,885     | 42,241         | 11,358        |
| 4     | Mar 19-23   | Mar 24      | 806,480      | 136,408     | 45,721         | 7,585         |
| 5     | Mar 25-29   | Mar 30      | 737,564      | 106,016     | 41,747         | 5,948         |

**Totais:** 4.5M eventos, 256K sessões, 45K itens, 180K usuários

## Replicando o Experimento

Execute os comandos na ordem abaixo para replicar o experimento completo:

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Criar sliding window splits (30 dias → 5 slices)
#    Entrada: /home/hygo2025/Documents/data/processed_data/enriched_events
#    Saída: data/sliding_window/slice_{1..5}/
make prepare-data

# 3. Converter para formato RecBole
#    Entrada: data/sliding_window/
#    Saída: recbole_data/realestate_slice{1..5}/*.inter
make convert-recbole

# 4. Executar todos os modelos em todos os slices
#    Saída: results/slice_{1..5}/*.csv
make run-all

# 5. Gerar tabelas de resultados agregados
#    Saída: results/aggregated_results.csv
make aggregate-results
```

**Tempo estimado:** ~6-8 horas (dependendo do hardware)

**Requisitos:**
- Python 3.9+
- 16GB+ RAM (para PySpark)
- GPU recomendada (para modelos neurais)

## Estrutura

```
fermi/
├── src/
│   ├── preprocessing/
│   │   ├── sliding_window_pipeline.py    # PySpark pipeline
│   │   └── recbole_converter.py          # Converter para RecBole
│   ├── configs/                          # Configs de modelos
│   └── run_experiments.py                # Runner
├── data/
│   └── sliding_window/                   # Dados processados (parquet)
├── recbole_data/                         # Formato RecBole
├── results/                              # Resultados
└── artigo/                               # Paper e metodologia
```

## Modelos

**Redes Neurais (RecBole):**
- GRU4Rec (Hidasi et al. 2016)
- NARM (Li et al. 2017)
- STAMP (Liu et al. 2018)
- SRGNN (Wu et al. 2019)

**KNN e Baselines:**
- ItemKNN
- SKNN  
- Pop

## Métricas

**Next Item Prediction:**
- HitRate@K, MRR@K, Coverage@K, Popularity@K

**Rest of Session:**
- Precision@K, Recall@K, NDCG@K, MAP@K

K ∈ {5, 10, 20}

## Status do Pipeline

```
[✓] Sliding Window Data Preparation (PySpark)
    └─> data/sliding_window/slice_{1..5}/
    
[✓] RecBole Format Conversion (Pandas)  
    └─> recbole_data/realestate_slice{1..5}/*.inter
    
[⏳] Experimentos (em desenvolvimento)
    └─> results/
```

## Comandos do Makefile

```bash
# Pipeline de dados
make prepare-data       # Criar sliding window splits (PySpark)
make convert-recbole    # Converter para RecBole (.inter)

# Experimentos - Todos os modelos
make run-all            # Executar todos modelos em todos slices

# Experimentos - Modelos individuais
make run-gru4rec        # Apenas GRU4Rec em todos slices
make run-narm           # Apenas NARM em todos slices
make run-stamp          # Apenas STAMP em todos slices
make run-sasrec         # Apenas SASRec em todos slices

# Resultados
make aggregate-results  # Agregar resultados (média ± std)

# Utilidades
make clean              # Limpar cache
make help               # Ver todos comandos
```
