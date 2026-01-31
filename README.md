# Fermi - Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessões para plataforma de classificados imobiliários, replicando a metodologia de Domingues et al. (2024).

## Visão Geral

O projeto implementa um pipeline completo de avaliação de sistemas de recomendação baseados em sessão:

1. Preparação e filtragem de dados brutos
2. Criação de dataset em formato RecBole
3. Treinamento de 13 modelos (4 neurais + 2 fatoração + 4 baselines + 3 KNN)
4. Avaliação com métricas padrão (Recall, MRR, NDCG, Hit)
5. Agregação de resultados
6. Análise visual via API web

## Fonte dos Dados

Os dados utilizados são provenientes de uma plataforma de classificados imobiliários:
- Listings: Anúncios (localização, preço, características)
- Events: Interações de usuários (visualizações, cliques)
- Período: 30 dias (configurável)
- Filtro geográfico: Região Metropolitana da Grande Vitória/ES

## Instalação

```bash
# Clonar repositório
git clone <repo-url>
cd fermi

# Instalar dependências
pip install -e .
```

Requisitos:
- Python 3.9+
- Java 11+ (para PySpark)
- 16GB+ RAM (processamento de dados)
- GPU recomendada (treinamento de modelos neurais)

## Configuração

Editar `config/project_config.yaml` com os caminhos dos dados:

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

Veja [CONFIGURATION.md](CONFIGURATION.md) para detalhes.

## Pipeline de Dados

### Preparar Dataset RecBole

```bash
# Processa eventos, filtra sessões e gera arquivos .inter e .parquet
make data
```

Processamento:
1. Carrega eventos brutos e filtra por período
2. Filtra eventos por localização (cidades-alvo)
3. Mantém apenas eventos de interação relevantes (ListingRendered)
4. Cria sessões com ID único
5. Filtra sessões por comprimento (2-50 interações)
6. Remove itens raros (mínimo 5 ocorrências)
7. Re-filtra sessões após remoção de itens
8. Gera dois outputs:
   - `outputs/data/recbole/realestate/realestate.inter` - Formato RecBole
   - `outputs/data/sessions_for_api.parquet` - Formato para API

Estatísticas típicas:
- ~600K interações
- ~118K sessões
- ~29K itens únicos

## Modelos Implementados

### Modelos Neurais (RNN)

**GRU4Rec** (Hidasi et al., 2016)
- RNN com GRU para recomendação sessional
- Config: 2 layers, 256 hidden, dropout 0.3, 50 epochs

**NARM** (Li et al., 2017)
- RNN com mecanismo de atenção
- Config: 2 layers, 256 hidden, dropout 0.3, 50 epochs

**STAMP** (Liu et al., 2018)
- Short-Term Attention/Memory Priority
- Config: 256 hidden, dropout 0.3, 50 epochs

**SASRec** (Kang & McAuley, 2018)
- Self-Attentive Sequential Recommendation
- Config: 3 layers, 4 heads, 256 hidden, 1024 inner, dropout 0.3, 50 epochs

### Modelos de Fatoração

**FPMC** (Rendle et al., 2010)
- Factorized Personalized Markov Chains
- Config: 256 embedding, reg_weight 0.0001, 50 epochs

**FOSSIL** (He & McAuley, 2016)
- Hybrid FISM + FPMC
- Config: order 3, alpha 0.5, 256 embedding, reg_weight 0.0001, 50 epochs

### Baselines (RecBole Native)

- **Pop**: Popularidade global
- **Random**: Recomendação aleatória

### Modelos KNN (Neighbor-based)

**ItemKNN** (Sarwar et al., 2001)
- Item-based collaborative filtering
- Config: k=500, similarity=cosine

Configurações comuns (neurais/fatoração):
- Early stopping (patience 10)
- Gradient clipping (max_norm 5.0)
- Batch size: 4096
- Learning rate: 0.001

## Execução do Benchmark

### Benchmark Completo

```bash
# Executar todos os modelos
make benchmark MODELS=all
```

### Modelos Específicos

```bash
# Apenas modelos neurais
make benchmark-neurais

# Apenas baselines
make benchmark-baselines

# Apenas modelos de fatoração
make benchmark-factor

# Apenas modelos KNN
python src/run_benchmark.py --model VSKNN
python src/run_benchmark.py --model STAN
python src/run_benchmark.py --model VSTAN

# Teste rápido (apenas GRU4Rec)
make benchmark-quick

# Modelo específico
make benchmark MODELS=GRU4Rec
```

### Opções de Dataset

```bash
# Dataset customizado
make benchmark MODELS=GRU4Rec DATASET=custom_dataset
```

Tempo estimado:
- Teste rápido (1 modelo): 15-30 min
- Benchmark completo (10 modelos): 3-6 horas (com GPU)

## Hyperparameter Tuning

Baseado no guia oficial do RecBole, o script `src/hyperparameter_tuning.py`
executa o HyperTuning (HyperOpt) com os espaços definidos em
`src/configs/tuning/<modelo>_space.yaml`.

```bash
# Forma direta
python src/hyperparameter_tuning.py --model GRU4Rec --max-evals 20 --algo random

# Via Makefile (wrap conveniente)
make tune MODEL=GRU4Rec MAX_EVALS=20 ALGO=bayes COOLDOWN=60

# Lote completo (todos os modelos com search space)
make tune-all MAX_EVALS=10 COOLDOWN=60

# Smoke test (1 trial por modelo, ideal para validar pipeline)
make tune-smoke
```

Argumentos úteis:
- `DATASET=<nome>`: usa outro dataset (default: `config/project_config.yaml`)
- `OUTPUT=<dir>`: salva resultados em um diretório customizado
- `EARLY_STOP=<n>`: número de trials sem melhora antes de encerrar
- `COOLDOWN=<segundos>`: pausa entre trials para controlar temperatura da GPU (default 60s)

Para ajustar o espaço de busca, edite os arquivos em `src/configs/tuning/`.
Cada parâmetro aceita `choice`, `uniform`, `loguniform`, `randint` ou `quniform`,
seguindo a sintaxe explicada no README daquele diretório.

## Resultados

Os resultados são salvos em `outputs/results/<timestamp>/`:

```
outputs/results/20260107_193000/
├── README.md                    # Sumário da execução
├── raw_results.csv             # Resultados por modelo
├── aggregated_results.csv      # Estatísticas agregadas
└── model_checkpoints/          # Modelos salvos (opcional)
```

### Agregação de Resultados

```bash
# Agregar resultados da última execução
make aggregate
```

## Web API - Análise de Sessões e Recomendações

A API permite visualizar sessões e gerar recomendações interativamente.

### Iniciar API

```bash
# Com nome do modelo (busca o mais recente)
make api MODEL=GRU4Rec

# Com path específico
make api MODEL=outputs/saved/GRU4Rec-Jan-07-2026_19-25-00.pth

# Modo desenvolvimento (auto-reload)
make api-dev MODEL=GRU4Rec
```

Acesse: `http://localhost:8000`

### Funcionalidades

- Lista de sessões com paginação e ordenação
- Visualização de sessões com mapa interativo (Folium)
- Geração de recomendações para qualquer sessão
- Análise espacial (distâncias geográficas)
- Comparação de características (preço, quartos, área)
- API REST para integração

### Endpoints Principais

- `GET /sessions` - Lista paginada de sessões
- `GET /session/{session_id}` - Detalhes de sessão
- `POST /recommend-from-session/{session_id}` - Gera recomendações
- `GET /api/health` - Status da API
- `GET /api/recommend?session_ids=1,2,3&top_k=10` - API JSON

### Requisitos da API

O arquivo de sessões para API é gerado automaticamente pelo pipeline de dados em:
```
outputs/data/sessions_for_api.parquet
```

Se não existir, execute:
```bash
make data
```

## Metodologia

Seguindo Domingues et al. (2024):

- Protocolo: next-item prediction
- Split: Temporal Leave-One-Out (penúltimo para validação, último para teste)
- Avaliação: Full ranking contra todo catálogo
- Métricas: Recall@K, MRR@K, NDCG@K, Hit@K (K=5,10,20)
- Métrica principal: MRR@10 (early stopping)

## Estrutura do Projeto

```
fermi/
├── src/
│   ├── pipeline/
│   │   └── recbole_data_pipeline.py    # Pipeline de dados
│   ├── models/                          # Baselines personalizados
│   ├── configs/                         # Configurações de modelos
│   │   ├── neural/
│   │   ├── baselines/
│   │   └── factorization/
│   ├── api/
│   │   ├── app.py                      # FastAPI server
│   │   ├── recommendation_analyzer.py   # Análise de recomendações
│   │   └── templates/                  # Templates HTML
│   ├── utils/
│   │   ├── enviroment.py
│   │   └── spark_session.py
│   ├── run_benchmark.py
│   └── aggregate_results.py
├── config/
│   └── project_config.yaml
├── outputs/
│   ├── data/                           # Datasets processados
│   │   ├── recbole/
│   │   └── sessions_for_api.parquet
│   ├── results/                        # Resultados de experimentos
│   ├── saved/                          # Modelos treinados
│   └── logs/
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Comandos do Makefile

```bash
# Instalação
make install              # Instalar dependências

# Pipeline de dados
make data                 # Preparar dataset RecBole (completo)
make data-custom START_DATE=... END_DATE=...  # Período customizado

# Benchmark
make benchmark            # Executar modelos (MODELS=... DATASET=...)
make benchmark-neurais    # Apenas modelos neurais
make benchmark-baselines  # Apenas baselines
make benchmark-factor     # Apenas modelos de fatoração
make benchmark-quick      # Teste rápido (GRU4Rec)

# Resultados
make aggregate            # Agregar última execução

# API
make api MODEL=...        # Iniciar API web
make api-dev MODEL=...    # API com auto-reload

# Manutenção
make clean                # Limpar cache Python
make clean-all            # Limpar tudo (cache, logs, resultados)
make help                 # Ver todos comandos
```

## Troubleshooting

### CUDA Out of Memory

Reduzir batch size nas configs:
```yaml
train_batch_size: 2048  # ao invés de 4096
eval_batch_size: 2048
```

### API não encontra sessions

Execute o pipeline de dados:
```bash
make data
```

Verifique se foi criado:
```bash
ls -lh outputs/data/sessions_for_api.parquet
```

### Erro ao carregar modelo

Verifique se o modelo existe:
```bash
ls -lh outputs/saved/
```

Use o path completo:
```bash
make api MODEL=outputs/saved/GRU4Rec-Jan-07-2026_19-25-00.pth
```

## Monitoramento

```bash
# GPU em tempo real
watch -n 1 nvidia-smi

# Logs de treinamento
tail -f outputs/logs/GRU4Rec.log

# Tensorboard (se habilitado)
tensorboard --logdir log_tensorboard
```

## Referência

Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2024).  
A large scale benchmark for session-based recommendations in the legal domain.  
Artificial Intelligence and Law, 33, 43-78.  
DOI: 10.1007/s10506-023-09378-3

## Licença

MIT License


## Item Canonicalization (Correção de Cold Start)

**Problema:** Imóveis idênticos recebem novos IDs ao reentrar no catálogo, zerando o histórico de interações.

**Solução:** `canonical_listing_id` agrupa imóveis fisicamente similares via fingerprint:
- Localização: lat/lon arredondado (4 casas decimais ~11m)
- Área: bucketizada em 5m²
- Tipologia: bedrooms + suites + unit_type
- Hash: MD5 do fingerprint

**Resultado:** 58% de redução de esparsidade (506k IDs → 210k clusters)

**Uso:** O `listing_id_numeric` é gerado **por canonical_id**, não por anonymized_id. Todos os membros do mesmo cluster compartilham o mesmo ID numérico nos modelos.

```python
# Mapeamento gerado automaticamente em listings_pipeline.py
# anonymized_listing_id -> canonical_listing_id -> listing_id_numeric
# Exemplo: LISTING_A, LISTING_B, LISTING_C (mesmo cluster) → ID numérico 1
```

**Validação:**
```bash
python scripts/validate_canonical_id.py
```
