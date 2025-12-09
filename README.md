# Fermi - Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessões para dados de real estate, replicando a metodologia de Domingues et al. (2024).

## Documentação

- [Configuração de Experimentos](docs/experiments.md) - Modelos, dados e métricas
- [Guia de Execução](docs/execution.md) - Como reproduzir os experimentos
- [Otimização GPU](docs/gpu-optimization.md) - Configurações para RTX 4090
- [Sistema de Cooling GPU](docs/gpu-cooling.md) - Proteção térmica automática

## Visão Geral

Pipeline completo para benchmarking de modelos de recomendação baseados em sessão:

1. Preparação de dados com sliding window temporal (5 slices de 30 dias)
2. Conversão para formato RecBole
3. Treinamento de 4 modelos neurais em cada slice
4. Avaliação com métricas padrão (Recall, MRR, NDCG, Hit)
5. Agregação de resultados (média ± desvio padrão)

Total: 20 experimentos (4 modelos × 5 slices)

## Modelos Implementados

- **GRU4Rec:** RNN para recomendação sessional (Hidasi et al., 2016)
- **NARM:** RNN com mecanismo de atenção (Li et al., 2017)
- **STAMP:** Short-Term Attention/Memory Priority (Liu et al., 2018)
- **SASRec:** Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)

## Quick Start

```bash
# 1. Preparar dados (sliding window)
make prepare-data

# 2. Converter para RecBole
make convert-recbole

# 3. Executar todos os experimentos
make run-all

# 4. Agregar resultados
make aggregate-results
```

Tempo estimado: 1-2 horas (GPU RTX 4090 com batch_size=4096)

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
#    Saída: outputs/results/raw_results.csv
make run-all

# 5. Gerar tabelas de resultados agregados
#    Saída: outputs/results/aggregated_results.csv
make aggregate-results
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

# Resultados
make aggregate-results  # Agregar resultados (média ± std)

# Utilidades
make clean              # Limpar cache
make help               # Ver todos comandos
```

## Estrutura

```
fermi/
├── src/                          # Código fonte
│   ├── models/                   # Modelos baseline
│   ├── preprocessing/            # Pipeline de dados
│   │   ├── sliding_window_pipeline.py
│   │   └── recbole_converter.py
│   ├── configs/                  # Configurações dos modelos
│   │   └── neural/*.yaml
│   ├── utils/                    # Utilidades
│   │   ├── spark_session.py
│   │   └── gpu_cooling.py
│   ├── run_experiments.py        # Runner principal
│   ├── aggregate_results.py      # Agregação de resultados
│   └── metrics.py                # Métricas customizadas
├── scripts/                      # Scripts auxiliares
│   ├── run_parallel.sh
│   └── monitor_gpu.sh
├── docs/                         # Documentação
│   ├── experiments.md
│   ├── execution.md
│   ├── gpu-optimization.md
│   ├── gpu-cooling.md
│   └── papers/                   # Artigos de referência
├── outputs/                      # Dados e resultados gerados
│   ├── data/                     # Dados processados
│   │   ├── sliding_window/       # Dados preparados (Parquet)
│   │   └── recbole/              # Dados convertidos (.inter)
│   ├── results/                  # Resultados dos experimentos
│   │   ├── raw_results.csv
│   │   └── aggregated_results.csv
│   ├── models/                   # Modelos treinados
│   └── logs/                     # Logs de treinamento
│       └── tensorboard/
├── pyproject.toml                # Configuração do pacote
├── requirements.txt              # Dependências
└── README.md
```

## Metodologia

Seguindo Domingues et al. (2024):

- Sliding window temporal: 5 slices de 30 dias
- Protocolo de avaliação: next-item prediction
- Split temporal: últimos 7 dias de cada janela para teste
- Métricas: Recall@K, MRR@K, NDCG@K, Hit@K (K=5,10,20)
- Agregação: média ± desvio padrão entre slices

## Configurações GPU

Os modelos estão configurados para maximizar o uso da GPU RTX 4090:

- Batch size: 4096 (8x maior que padrão)
- Hidden size: 256 (2.5x maior que padrão)
- Embedding size: 256 (2.5x maior que padrão)

Sistema de cooling automático ativo por padrão:
- Pausas a cada 5 epochs
- Duração: 60 segundos
- Temperatura máxima: 80°C

Ver [docs/gpu-optimization.md](docs/gpu-optimization.md) e [docs/gpu-cooling.md](docs/gpu-cooling.md) para detalhes.

## Referência

Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2024).  
A large scale benchmark for session-based recommendations in the legal domain.  
Artificial Intelligence and Law, 33, 43-78.  
DOI: 10.1007/s10506-023-09378-3

## Licença

MIT License
