# Fermi - Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessões para plataforma de classificados, replicando a metodologia de Domingues et al. (2024).

## Documentação

- [Metodologia](docs/metodologia.md) - Detalhes da metodologia do artigo base

## Pipeline de Dados

O projeto processa dados em 4 etapas:

### Etapa 0: Preparação de Dados Brutos

Processa os dados brutos da plataforma de classificados (listings + events) e gera o dataset enriquecido.

Input:
- Listings: CSV com dados dos anúncios
- Events: CSV.GZ com eventos de usuários

Processamento:
- Limpeza e normalização de dados
- Resolução de identidades (usuários anônimos vs logados)
- Criação de sessões (timeout 30 min)
- Enriquecimento com dados geográficos
- Merge final dos dados

Output:
- Dataset enriquecido em `/home/hygo2025/Documents/data/processed_data/enriched_events`

```bash
make prepare-raw-data
```

Requisitos:
- Configurar `config/project_config.yaml` com os caminhos dos dados brutos (veja [CONFIGURATION.md](CONFIGURATION.md))
- PySpark 3.x, Java 11+, 16GB+ RAM

### Etapa 1: Sliding Window

Cria 5 slices temporais de 30 dias para avaliação temporal.

```bash
make prepare-data
```

### Etapa 2: Conversão RecBole

Converte os dados para o formato esperado pela biblioteca RecBole.

```bash
make convert-recbole
```

### Etapa 3: Experimentos

Executa os 10 modelos em cada slice e agrega os resultados.

```bash
make run-all
```

## Fonte dos Dados

Os dados utilizados são de uma plataforma de classificados:
- Listings: Dados dos anúncios (localização, preço, características)
- Events: Eventos de interação dos usuários (visualizações, cliques, etc.)
- Período: Março de 2024 (30 dias)

## Visão Geral

0. Preparação de dados brutos (listings + events)
1. Sliding window temporal (5 slices de 30 dias)
2. Conversão para formato RecBole
3. Treinamento de 10 modelos (4 neurais + 2 fatoração + 4 baselines)
4. Avaliação com métricas padrão (Recall, MRR, NDCG, Hit)
5. Agregação de resultados (média ± desvio padrão)

Total: 50 experimentos (10 modelos × 5 slices)

## Modelos Implementados

Modelos Neurais (RNN):
- GRU4Rec: RNN para recomendação sessional (Hidasi et al., 2016)
  - Config: 2 layers, 256 hidden, dropout 0.3, 50 epochs
- NARM: RNN com mecanismo de atenção (Li et al., 2017)
  - Config: 2 layers, 256 hidden, dropout 0.3, 50 epochs
- STAMP: Short-Term Attention/Memory Priority (Liu et al., 2018)
  - Config: 256 hidden, dropout 0.3, 50 epochs
- SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
  - Config: 3 layers, 4 heads, 256 hidden, 1024 inner, dropout 0.3, 50 epochs

Modelos de Fatoração de Matrizes:
- FPMC: Factorized Personalized Markov Chains (Rendle et al., 2010)
  - Config: 256 embedding, reg_weight 0.0001, 50 epochs
- FOSSIL: Hybrid FISM + FPMC (He & McAuley, 2016)
  - Config: order 3, alpha 0.5, 256 embedding, reg_weight 0.0001, 50 epochs

Baselines:
- Random: Recomendação aleatória
- POP: Popularidade global
- RPOP: Popularidade recente (últimos 7 dias)
- SPOP: Popularidade por sessão

Todas as configurações incluem:
- Early stopping com patience 10
- Gradient clipping (max_norm: 5.0)
- Batch size: 4096
- Learning rate: 0.001

## Quick Start

### Pipeline Completo (Automático)

```bash
# Executar todo pipeline de uma vez (prepare-raw-data + prepare-data + convert-recbole + run-all)
make pipeline-complete
```

Tempo estimado: 40-60 horas (com early stopping)

### Pipeline Manual (Passo a Passo)

```bash
# 0. Preparar dados brutos (primeira vez apenas)
make prepare-raw-data

# 1. Criar sliding window
make prepare-data

# 2. Converter para RecBole
make convert-recbole

# 3. Executar experimentos
make run-all
```

Outputs gerados automaticamente em `outputs/results/YYYYMMDD_HHMMSS/`:
- README.md - Informações da execução e top 3 modelos
- raw_results.csv - Resultados brutos (todos modelos × slices)
- aggregated_results.csv - Resultados agregados (mean ± std)
- results_table.md - Tabela Markdown para README
- metrics_comparison.png - Gráfico de barras comparativo
- performance_heatmap.png - Heatmap de performance
- slice_consistency.png - Consistência temporal
- loss_curves_[model].png - Curvas de training loss por modelo
- loss_curves_average.png - Curvas de training loss médias
- losses/ - Diretório com histórico de loss em JSON

Cada execução cria uma pasta nova com timestamp.


## Guia de Execução

### Pipeline Completo

```bash
# 0. Preparar dados brutos (primeira vez apenas)
make prepare-raw-data

# 1. Criar sliding window
make prepare-data

# 2. Converter para RecBole
make convert-recbole

# 3. Executar todos experimentos
make run-all
```

Tempo estimado: ~3-4 horas (GPU RTX 4090)

### Execução por Modelo

**Modo Serial** (1 slice por vez):
```bash
make run-gru4rec    # GRU4Rec (5 slices)
make run-narm       # NARM (5 slices)
make run-stamp      # STAMP (5 slices)
make run-sasrec     # SASRec (5 slices)
make run-fpmc       # FPMC (5 slices)
make run-fossil     # FOSSIL (5 slices)
```

**Modo Paralelo** (2 slices simultâneos - 40% mais rápido):
```bash
make run-gru4rec-parallel    # GRU4Rec com 2 slices paralelos
make run-parallel MODEL=NARM # Qualquer modelo em paralelo
```

Modo paralelo requer:
- GPU com 24GB VRAM (RTX 4090) para rodar 2 slices simultaneamente, ou
- 2 GPUs (roda 1 slice por GPU automaticamente)

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

```bash
# 1. Instalar dependências
pip install -e .

# 2. Configurar projeto
# Editar config/project_config.yaml com os caminhos dos dados
# Veja CONFIGURATION.md para detalhes

# 3. Processar dados brutos
make prepare-raw-data

# 4. Preparar dataset para RecBole
make data

# 5. Executar benchmark
make benchmark MODELS=all
```

Requisitos:
- Python 3.9+
- 16GB+ RAM (para PySpark)
- GPU recomendada (para modelos neurais)
- Java 11+ (para Spark)
- Dados brutos da plataforma de classificados

## Comandos do Makefile

```bash
# Pipeline de dados
make prepare-raw-data   # Processar dados brutos (listings + events)
make prepare-data       # Criar sliding window splits (PySpark)
make convert-recbole    # Converter para RecBole (.inter)

# Experimentos - Todos os modelos
make run-all            # Executar todos modelos em todos slices

# Experimentos - Modelos individuais (serial)
make run-gru4rec        # Apenas GRU4Rec em todos slices
make run-narm           # Apenas NARM em todos slices
make run-stamp          # Apenas STAMP em todos slices
make run-sasrec         # Apenas SASRec em todos slices
make run-fpmc           # Apenas FPMC em todos slices
make run-fossil         # Apenas FOSSIL em todos slices

# Experimentos - Execução paralela (2 slices simultâneos)
make run-gru4rec-parallel    # GRU4Rec em paralelo (40% mais rápido)
make run-parallel MODEL=NARM # Qualquer modelo em paralelo

# Resultados
# (Agregação automática no final de run-all)

# Utilidades
make clean              # Limpar cache
make help               # Ver todos comandos
```

## Estrutura

```
fermi/
├── src/
│   ├── data_preparation/         # Preparação de dados brutos
│   │   ├── pipelines/            # Pipelines de processamento
│   │   │   ├── listings_pipeline.py
│   │   │   ├── events_pipeline.py
│   │   │   └── merge_events.py
│   │   └── prepare_raw_data.py
│   ├── preprocessing/            # Pipeline de sliding window
│   │   ├── sliding_window_pipeline.py
│   │   ├── recbole_converter.py
│   │   └── prepare_dataset.py
│   ├── models/                   # Modelos baseline
│   ├── configs/                  # Configurações
│   │   ├── baselines/
│   │   └── neural/
│   ├── utils/                    # Utilitários
│   │   ├── spark_session.py
│   │   ├── spark_utils.py
│   │   ├── data_utils.py
│   │   ├── enviroment.py
│   │   ├── geocode.py
│   │   └── gpu_cooling.py
│   ├── run_experiments.py
│   ├── aggregate_results.py
│   └── metrics.py
├── scripts/
│   ├── run_parallel.sh
│   ├── run_all_experiments.sh
│   └── monitor_gpu.sh
├── docs/
│   ├── metodologia.md
│   └── papers/
├── outputs/
│   ├── data/
│   │   ├── sliding_window/
│   │   └── recbole/
│   ├── results/
│   ├── saved/
│   └── logs/tensorboard/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Metodologia

Seguindo o paper de Domingues et al. (2024):

- Sliding window temporal: 5 slices de 30 dias
- Protocolo: next-item prediction
- Split temporal: últimos 7 dias de cada janela para teste
- Métricas: Recall@K, MRR@K, NDCG@K, Hit@K (K=5,10,20)
- Agregação: média ± desvio padrão entre slices

## Gerenciamento de Checkpoints

Por padrão, checkpoints **NÃO são salvos** para economizar espaço (~25 GB para 50 experimentos).

### Habilitar salvamento

```bash
# Para análise exploratória posterior
python src/run_experiments.py --all-slices --save-checkpoints
```

### Gerenciar checkpoints existentes

```bash
# Ver estatísticas
python scripts/manage_checkpoints.py --strategy stats

# Manter apenas o melhor de cada modelo
python scripts/manage_checkpoints.py --strategy keep-best

# Manter apenas os N mais recentes
python scripts/manage_checkpoints.py --strategy keep-recent --keep-n 5

# Limpar tudo
python scripts/manage_checkpoints.py --strategy clean-all
```

**Recomendação**: Use checkpoints apenas se precisar fazer análise exploratória. Para benchmark, as métricas nos CSVs são suficientes.

### Monitoramento

```bash
# Ver uso de GPU em tempo real
watch -n 1 nvidia-smi

# Acompanhar logs
tail -f outputs/logs/GRU4Rec_slice*.log
```

### Troubleshooting

**Erro: CUDA Out of Memory**
- Solução: Reduzir `train_batch_size` no YAML de configuração
- Alternativa: Rodar modo serial (`make run-gru4rec`)

**Logs não aparecem**
- Processos ainda rodando, aguarde ou verifique com `ps aux | grep python`

## Análise de Recomendações

Após treinar modelos, analise qualitativamente as recomendações geradas comparando com sessões reais do dataset de teste.

### Uso

```bash
# Análise básica (5 sessões, Top-10)
make analyze-model MODEL_PATH=outputs/saved/GRU4Rec-Dec-16-2025_18-06-21.pth

# Análise customizada
make analyze-model MODEL_PATH=outputs/saved/GRU4Rec.pth NUM_SESSIONS=10 TOP_K=20
```

### O que é Analisado

- Compatibilidade de preços entre sessão e recomendações
- Compatibilidade geográfica (cidades)
- Compatibilidade de características (quartos, área)
- Adaptação do modelo a diferentes perfis (luxo, econômico, padrão)

### Output

- Console: análise detalhada de cada sessão
- CSV: `outputs/analysis/<modelo>/analysis_results.csv`
- PNG: `outputs/analysis/<modelo>/comparative_analysis.png`

### Interpretação

Thresholds:
- BOM: ≥60% compatibilidade
- MODERADO: 30-60% compatibilidade
- RUIM: <30% compatibilidade

Se abaixo de 60%:
- Revisar hiperparâmetros
- Testar outras arquiteturas
- Verificar qualidade dos dados

## Resultados

Os resultados são salvos em `outputs/results/<timestamp>/`.

Para agregar resultados:
```bash
make aggregate-last
```

## Referência
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2024).  
A large scale benchmark for session-based recommendations in the legal domain.  
Artificial Intelligence and Law, 33, 43-78.  
DOI: 10.1007/s10506-023-09378-3

## Licença

MIT License
