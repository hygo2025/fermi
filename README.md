# Fermi - Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessões para dados de real estate, replicando a metodologia de Domingues et al. (2024).

## Documentação

- [Metodologia](docs/metodologia.md) - Detalhes da metodologia do artigo base

## Visão Geral

Pipeline completo para benchmarking de modelos de recomendação baseados em sessão:

1. Preparação de dados com sliding window temporal (5 slices de 30 dias)
2. Conversão para formato RecBole
3. Treinamento de 10 modelos (4 neurais + 2 fatoração + 4 baselines) em cada slice
4. Avaliação com métricas padrão (Recall, MRR, NDCG, Hit)
5. Agregação de resultados (média ± desvio padrão)

Total: 50 experimentos (10 modelos × 5 slices)

## Modelos Implementados

**Modelos Neurais (RNN):**
- **GRU4Rec:** RNN para recomendação sessional (Hidasi et al., 2016)
- **NARM:** RNN com mecanismo de atenção (Li et al., 2017)
- **STAMP:** Short-Term Attention/Memory Priority (Liu et al., 2018)
- **SASRec:** Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)

**Modelos de Fatoração de Matrizes:**
- **FPMC:** Factorized Personalized Markov Chains (Rendle et al., 2010)
- **FOSSIL:** Hybrid FISM + FPMC (He & McAuley, 2016)

**Baselines:**
- **Random:** Recomendação aleatória
- **POP:** Popularidade global
- **RPOP:** Popularidade recente (últimos 7 dias)
- **SPOP:** Popularidade por sessão

## Quick Start

```bash
# 1. Preparar dados (sliding window)
make prepare-data

# 2. Converter para RecBole
make convert-recbole

# 3. Executar experimentos (agregação automática ao final)
make run-all
```

**Outputs gerados automaticamente em `outputs/results/YYYYMMDD_HHMMSS/`:**
- `README.md` - Informações da execução e top 3 modelos
- `raw_results.csv` - Resultados brutos (todos modelos × slices)
- `aggregated_results.csv` - Resultados agregados (mean ± std)
- `results_table.md` - Tabela Markdown para README
- `metrics_comparison.png` - Gráfico de barras comparativo
- `performance_heatmap.png` - Heatmap de performance
- `slice_consistency.png` - Consistência temporal
- `loss_curves_[model].png` - Curvas de training loss por modelo
- `loss_curves_average.png` - Curvas de training loss médias (comparação entre modelos)
- `losses/` - Diretório com histórico de loss em JSON

Cada execução cria uma pasta nova com timestamp para organizar tudo.


## Guia de Execução

### Pipeline Completo

```bash
# 1. Preparar dados (sliding window)
make prepare-data

# 2. Converter para RecBole
make convert-recbole

# 3. Executar todos experimentos
make run-all
```

**Tempo estimado:** ~2-3 horas (GPU RTX 4090)

### Execução por Modelo

```bash
make run-gru4rec    # GRU4Rec (5 slices)
make run-narm       # NARM (5 slices)
make run-stamp      # STAMP (5 slices)
make run-sasrec     # SASRec (5 slices)
make run-fpmc       # FPMC (5 slices)
make run-fossil     # FOSSIL (5 slices)
```

### Execução Granular

```bash
# Um modelo em um slice
python src/run_experiments.py --models GRU4Rec --slices 1

# Múltiplos modelos e slices
python src/run_experiments.py --models GRU4Rec NARM --slices 1 2 3

# Todos os slices
python src/run_experiments.py --models GRU4Rec --all-slices
```


## Replicação do Experimento

Execute os comandos na ordem abaixo para replicar o experimento completo:

```bash
# 1. Instalar dependências
pip install -e .

# 2. Criar sliding window splits (30 dias → 5 slices)
#    Entrada: /home/hygo2025/Documents/data/processed_data/enriched_events
#    Saída: outputs/data/sliding_window/slice_{1..5}/
make prepare-data

# 3. Converter para formato RecBole
#    Entrada: outputs/data/sliding_window/
#    Saída: outputs/data/recbole/realestate_slice{1..5}/*.inter
make convert-recbole

# 4. Executar todos os modelos em todos os slices
#    Saída: outputs/results/YYYYMMDD_HHMMSS/
#    (Agregação automática ao final)
make run-all
```

Requisitos:
- Python 3.9+
- 16GB+ RAM (para PySpark)
- GPU recomendada (para modelos neurais)

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
make run-fpmc           # Apenas FPMC em todos slices
make run-fossil         # Apenas FOSSIL em todos slices

# Resultados
# (Agregação automática no final de run-all)

# Utilidades
make clean              # Limpar cache
make help               # Ver todos comandos
```

## Estrutura

```
fermi/
├── src/                          # Código fonte
│   ├── models/                   # Modelos baseline (POP, RPOP, SPOP, Random)
│   ├── preprocessing/            # Pipeline de dados
│   │   ├── sliding_window_pipeline.py  # Criação de splits temporais
│   │   ├── recbole_converter.py        # Conversão para RecBole
│   │   └── prepare_dataset.py          # Preparação de datasets
│   ├── configs/                  # Configurações dos modelos
│   │   ├── baselines/            # Configs de baselines
│   │   └── neural/               # Configs de modelos neurais
│   ├── utils/                    # Utilitários
│   │   ├── spark_session.py      # Configuração do Spark
│   │   └── gpu_cooling.py        # Sistema de cooling GPU
│   ├── run_experiments.py        # Runner principal
│   ├── aggregate_results.py      # Agregação de resultados
│   └── metrics.py                # Métricas customizadas
├── scripts/                      # Scripts shell
│   ├── run_parallel.sh           # Execução paralela de slices
│   ├── run_all_experiments.sh    # Executar todos modelos
│   └── monitor_gpu.sh            # Monitor de GPU
├── docs/                         # Documentação
│   ├── experiments.md
│   ├── execution.md
│   ├── gpu-optimization.md
│   ├── gpu-cooling.md
│   ├── REORGANIZATION.md         # Log de reorganização
│   └── papers/                   # Artigos de referência
├── outputs/                      # Tudo que é gerado
│   ├── data/
│   │   ├── sliding_window/       # Splits temporais (Parquet)
│   │   └── recbole/              # Formato RecBole (.inter)
│   ├── results/                  # Resultados CSV
│   ├── saved/                    # Checkpoints de modelos
│   └── logs/tensorboard/         # Logs TensorBoard
├── pyproject.toml                # Configuração do pacote
├── requirements.txt              # Dependências
├── Makefile                      # Comandos do pipeline
└── README.md
```

## Metodologia

Seguindo o paper de Domingues et al. (2024):

- Sliding window temporal: 5 slices de 30 dias
- Protocolo: next-item prediction
- Split temporal: últimos 7 dias de cada janela para teste (sem validação)
- Métricas: Recall@K, MRR@K, NDCG@K, Hit@K (K=5,10,20)
- Agregação: média ± desvio padrão entre slices

## Configurações GPU

**Otimizações para RTX 4090:**
- Batch size: 4096 (padrão: 512)
- Hidden/Embedding size: 256 (padrão: 100)
- VRAM usada: 6-8 GB por experimento

**Sistema de Cooling Automático:**
- Pausas de 60s a cada 5 epochs
- Limite de temperatura: 80°C
- Desativar com: `--no-gpu-cooling`

## Referência

Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2024).  
A large scale benchmark for session-based recommendations in the legal domain.  
Artificial Intelligence and Law, 33, 43-78.  
DOI: 10.1007/s10506-023-09378-3

## Licença

MIT License

## Referências dos Modelos

**Modelos Neurais:**
- Hidasi, B., et al. (2016). Session-based recommendations with recurrent neural networks. ICLR.
- Li, J., et al. (2017). Neural attentive session-based recommendation. CIKM.
- Liu, Q., et al. (2018). STAMP: Short-term attention/memory priority model. KDD.
- Kang, W., & McAuley, J. (2018). Self-attentive sequential recommendation. ICDM.

**Modelos de Fatoração:**
- Rendle, S., et al. (2010). Factorizing personalized markov chains for next-basket recommendation. WWW.
- He, R., & McAuley, J. (2016). FISM: Factored item similarity models for top-n recommender systems. KDD.
