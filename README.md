# Fermi - Session-Based Recommendation Benchmark

Benchmark de recomendação baseada em sessão para o domínio imobiliário, seguindo a metodologia de Domingues et al. (2025).

**Artigo de Referência:**  
"A large scale benchmark for session-based recommendations on the legal domain"  
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2025)  
Artificial Intelligence and Law, 33, 43-78.  
DOI: 10.1007/s10506-023-09378-3

## 📋 Visão Geral

Este projeto implementa e avalia múltiplos modelos de recomendação baseados em sessão usando dados reais de interações de usuários com listagens de imóveis. 

**Objetivo:** Predizer o próximo imóvel que um usuário vai interagir baseado na sequência de interações da sessão atual.

### Diferenças do Artigo Original

| Aspecto    | Artigo Original (Jusbrasil)   | Nossa Implementação           |
|------------|-------------------------------|-------------------------------|
| **Domínio**    | Legal (documentos jurídicos)  | Imobiliário (listings)        |
| **Itens**      | Documentos                    | Imóveis                       |
| **Framework**  | session-rec (deprecated)      | **RecBole** (moderno)         |
| **GPU**        | CPU only                      | **CUDA nativo (PyTorch)**     |
| **Métricas**   | Recall@K, MRR@K               | Recall, MRR, NDCG, Hit@K      |

### Por que RecBole?

Migramos do framework session-rec original para **RecBole** devido a problemas críticos:

✅ **Moderno e Mantido** - Desenvolvimento ativo, Python 3.9+  
✅ **GPU Nativo** - Aceleração CUDA completa via PyTorch  
✅ **Sem Incompatibilidades** - Sem dependências Theano/Aesara legadas  
✅ **Performance** - 10-20x mais rápido com GPU  
✅ **Métricas Ricas** - NDCG, Hit@K, Precision@K além de MRR/Recall  
✅ **Código Simples** - API limpa, sem wrappers necessários  

**RecBole:** https://github.com/RUCAIBox/RecBole

## 🚀 Quick Start

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/hygo2025/fermi.git
cd fermi

# Instalar dependências (Python 3.9+)
pip install -r requirements.txt

# Verificar GPU (opcional mas recomendado)
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

**Requisitos:**
- Python 3.9+
- CUDA 11.8+ (para aceleração GPU)
- 16GB+ RAM (para dataset completo)

### 2. Preparar Dados

O dataset já está preparado em formato Parquet. Converta para formato RecBole:

```bash
# Preparar dados (se necessário)
make prepare-data

# Ou manualmente
python src/data/prepare_dataset.py
```

**Nota:** Se você já tem dados no formato RecBole em `recbole_data/realestate/`, pode pular esta etapa.

**Output esperado:**
```
recbole_data/realestate/
  ✓ realestate.inter        (2.7M interações)
  ✓ realestate.train.inter  (2.1M treino)
  ✓ realestate.test.inter   (541K teste)
```

### 3. Executar Benchmarks

#### Testar Modelos Individuais

```bash
# Redes Neurais
make test-gru4rec    # GRU4Rec (com GPU!)
make test-narm       # NARM
make test-stamp      # STAMP
make test-srgnn      # SR-GNN

# Modelos KNN
make test-itemknn    # Item-based KNN
make test-sknn       # Session-based KNN

# Baseline
make test-pop        # Popularidade

# Ou usar o script
./scripts/run_all.sh
```

#### Executar Todos os Modelos

```bash
make run-all
```

Executa todos os 7 modelos sequencialmente e salva logs em `logs/`.

**Tempo Estimado (RTX 4090):**
- GRU4Rec: ~10-20 min (10 epochs)
- NARM: ~20-40 min (20 epochs)
- STAMP: ~10-20 min (10 epochs)
- SRGNN: ~10-20 min (10 epochs)
- ItemKNN: ~2-5 min
- SKNN: ~2-5 min
- Pop: <1 min

## 📁 Estrutura do Projeto

```
fermi/
├── src/                          # Código fonte
│   ├── configs/                  # Configurações YAML RecBole
│   │   ├── neural/               # 4 modelos neurais
│   │   │   ├── gru4rec.yaml
│   │   │   ├── narm.yaml
│   │   │   ├── stamp.yaml
│   │   │   └── srgnn.yaml
│   │   ├── knn/                  # 2 modelos KNN
│   │   │   ├── itemknn.yaml
│   │   │   └── sknn.yaml
│   │   └── baselines/            # 1 baseline
│   │       └── pop.yaml
│   │
│   ├── data/                     # Preparação de dados
│   │   └── prepare_dataset.py   # Script Spark
│   │
│   ├── utils/                    # Utilitários
│   │   └── spark_session.py     # Config Spark
│   │
│   ├── data_converter.py         # Parquet → RecBole
│   └── run_recbole.py            # Runner principal
│
├── recbole_data/                 # Dados RecBole
│   └── realestate/
│       ├── realestate.inter
│       ├── realestate.train.inter
│       └── realestate.test.inter
│
├── session_rec_format/           # Dados Parquet originais
│   └── realestate/
│       ├── realestate_train_full.parquet
│       └── realestate_test.parquet
│
├── scripts/
│   └── run_all.sh                # Executar todos benchmarks
│
├── logs/                         # Logs dos benchmarks
│   ├── neural/
│   ├── knn/
│   └── baselines/
│
├── artigo/                       # Documentos do artigo
├── .env                          # Variáveis de ambiente
├── requirements.txt              # Dependências Python
├── Makefile                      # Comandos de build
└── README.md
```

## 📊 Dados

### Estatísticas do Dataset

- **Total de interações:** 2,684,502
- **Sessões únicas:** 50,465
- **Itens únicos:** 4,799
- **Período:** 2024-04-01 a 2024-04-30

### Origem dos Dados

Os dados brutos estão em `/home/hygo2025/Documents/data/processed_data/`:
- **events/** - Eventos de usuários (~25M eventos, 182 dias)
- **listings/** - Catálogo de propriedades (187k imóveis)

### Pipeline de Preparação

1. **Filtrar eventos** por período
2. **Criar sessões** (timeout de 30min de inatividade)
3. **Remover sessões curtas** (<2 eventos) e itens raros (<5 ocorrências)
4. **Split temporal** (80% treino, 20% teste por SessionId)
5. **Converter** para formato RecBole (.inter)

### Formato RecBole

Arquivos `.inter` separados por tab:

```
session_id:tokenitem_id:tokentimestamp:float
S_10000301854441712214577.0
S_10000301854441712214578.0
...
```

**Tipos de campos:**
- `:token` - Campo categórico (string/int convertido para IDs)
- `:float` - Campo numérico (timestamps, ratings)

## 🤖 Modelos Implementados

### Redes Neurais (4/4 da metodologia) ✅

| Modelo   | Descrição                                      | Config                              | Paper                     |
|----------|------------------------------------------------|-------------------------------------|---------------------------|
| GRU4Rec  | Gated Recurrent Units for Recommendations     | src/configs/neural/gru4rec.yaml     | Hidasi et al. (2016)      |
| NARM     | Neural Attentive Recommendation Machine        | src/configs/neural/narm.yaml        | Li et al. (2017)          |
| STAMP    | Short-Term Attention Memory Priority           | src/configs/neural/stamp.yaml       | Liu et al. (2018)         |
| SRGNN    | Session-based Graph Neural Network             | src/configs/neural/srgnn.yaml       | Wu et al. (2019)          |

**Parâmetros comuns:**
- Embedding size: 100
- Hidden size: 100
- Learning rate: 0.001
- Batch size: 512
- Device: CUDA (GPU)

### Modelos KNN (2) ✅

| Modelo   | Descrição                      | Config                         | Tipo                      |
|----------|--------------------------------|--------------------------------|---------------------------|
| ItemKNN  | Item-based K-Nearest Neighbors | src/configs/knn/itemknn.yaml   | Item similarity           |
| SKNN     | Session-based KNN              | src/configs/knn/sknn.yaml      | Session similarity        |

**Parâmetros:**
- ItemKNN: k=100, similaridade coseno
- SKNN: k=500, sample_size=1000

### Baselines ✅

| Modelo | Descrição              | Config                             |
|--------|------------------------|------------------------------------|
| Pop    | Popularidade global    | src/configs/baselines/pop.yaml     |

## 📈 Métricas de Avaliação

Seguindo as métricas padronizadas do RecBole:

- **Recall@K** - Proporção de itens relevantes no top-K
- **MRR@K** - Mean Reciprocal Rank (posição do primeiro item relevante)
- **NDCG@K** - Normalized Discounted Cumulative Gain (qualidade do ranking)
- **Hit@K** - Taxa de acerto (binário: item relevante no top-K)

**Configuração:**
- K ∈ {5, 10, 20}
- Avaliação: predição do próximo item
- Protocolo: Leave-one-out (último item como teste)

### Interpretação das Métricas

- **Recall@20 = 0.15**: 15% das sessões têm o item correto no top-20
- **MRR@20 = 0.05**: Rank médio do item correto é ~20 (1/0.05)
- **NDCG@20 = 0.10**: Score de qualidade do ranking normalizado
- **Hit@20 = 0.15**: Mesmo que Recall@20 para predição de item único

## ⚙️ Configuração

### Exemplo de Config (YAML)

Cada modelo tem uma configuração YAML dedicada em `src/configs/`:

```yaml
# src/configs/neural/gru4rec.yaml
model: GRU4Rec
dataset: realestate
data_path: recbole_data/

# Treinamento
epochs: 10
train_batch_size: 512
eval_batch_size: 512
learning_rate: 0.001
train_neg_sample_args: ~  # None para CE loss

# Parâmetros do Modelo
embedding_size: 100
hidden_size: 100
num_layers: 1
dropout_prob: 0.1
loss_type: 'CE'

# Avaliação
metrics: ['Recall', 'MRR', 'NDCG', 'Hit']
topk: [5, 10, 20]
valid_metric: Recall@10

# Campos de Sessão
SESSION_ID_FIELD: session_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
USER_ID_FIELD: session_id  # Usa sessão como usuário
load_col:
  inter: [session_id, item_id, timestamp]

# Device
device: cuda
gpu_id: 0
```

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
BASE_PATH=/home/hygo2025/Documents/data
JAVA_HOME=/opt/jdk/amazon-corretto-21
PYTHONUNBUFFERED=1
```

## �� Comandos Úteis

### Setup

```bash
make prepare-data      # Preparar dados (se necessário)
```

**Nota:** Dados já preparados em `recbole_data/realestate/` podem ser usados diretamente.

### Modelos Neurais

```bash
make test-gru4rec      # GRU4Rec
make test-narm         # NARM
make test-stamp        # STAMP
make test-srgnn        # SRGNN
```

### Modelos KNN

```bash
make test-itemknn      # ItemKNN
make test-sknn         # SKNN
```

### Baseline

```bash
make test-pop          # Popularidade
```

### Executar Todos

```bash
make run-all           # Todos os 7 modelos sequencialmente
./scripts/run_all.sh   # Mesmo via script
```

### Ver Logs

```bash
tail -f logs/neural/gru4rec.log
tail -f logs/knn/sknn.log
```

## 📊 Resultados

Os resultados são salvos em dois locais:

1. **Console/Logs** - `logs/{neural,knn,baselines}/*.log`
2. **Modelos Salvos RecBole** - `saved/` (checkpoints, configs)

**Formato do Log:**
```
03 Dec 14:25    INFO  test result: {'recall@5': 0.1234, 'recall@10': 0.2345, ...}
03 Dec 14:25    INFO  best valid result: {'recall@10': 0.2456}
```

## 🐛 Troubleshooting

### Problemas Comuns

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solução:** Reduzir batch size no config:
```yaml
train_batch_size: 256  # Em vez de 512
eval_batch_size: 256
```

#### GPU Não Detectada

```
Device: cpu
```

**Verificar:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solução:** Instalar PyTorch com CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Dados Não Encontrados

```
FileNotFoundError: 'recbole_data/realestate/realestate.inter'
```

**Solução:**
```bash
# Preparar dados novamente
make prepare-data

# Ou verificar se os dados já existem
ls -lh recbole_data/realestate/
```

### Migração do session-rec

Se você tem código antigo do session-rec:

1. ✅ **Nova biblioteca** - Migrado para RecBole (mais moderna)
2. ✅ **Dados compatíveis** - Formato `.inter` do RecBole
3. ✅ **Novos configs** - Use formato YAML do RecBole
4. ❌ **Não misture** - Remova pasta session-rec-lib antiga se existir

## 📚 Performance Comparada

### session-rec (Antigo)

- ❌ Incompatibilidades Theano/Aesara
- ❌ Apenas CPU (sem GPU)
- ❌ Problemas de compatibilidade Python 3.9
- ❌ Wrappers complexos necessários
- ⏱️ GRU4Rec: **NÃO FUNCIONA** (erros de dimensão)

### RecBole (Novo)

- ✅ PyTorch moderno (GPU nativo)
- ✅ Python 3.9+ totalmente suportado
- ✅ API limpa (sem wrappers)
- ✅ Desenvolvimento ativo
- ⏱️ GRU4Rec: **10-20 min** na RTX 4090

**Speedup:** 10-20x mais rápido com aceleração GPU! 🚀

## 📖 Referências

### Artigo Principal

```
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2025).
A large scale benchmark for session-based recommendations on the legal domain.
Artificial Intelligence and Law, 33, 43-78.
DOI: 10.1007/s10506-023-09378-3
```

### Framework RecBole

```
Zhao, W. X., Mu, S., Hou, Y., Lin, Z., Chen, Y., Pan, X., ... & Wen, J. R. (2021).
RecBole: Towards a unified, comprehensive and efficient framework for recommendation algorithms.
In CIKM 2021.
URL: https://github.com/RUCAIBox/RecBole
```

### Modelos

- **GRU4Rec:** Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). Session-based recommendations with recurrent neural networks. ICLR.
- **NARM:** Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T., & Ma, J. (2017). Neural attentive session-based recommendation. CIKM.
- **STAMP:** Liu, Q., Zeng, Y., Mokhosi, R., & Zhang, H. (2018). STAMP: short-term attention/memory priority model for session-based recommendation. KDD.
- **SR-GNN:** Wu, S., Tang, Y., Zhu, Y., Wang, L., Xie, X., & Tan, T. (2019). Session-based recommendation with graph neural networks. AAAI.

## 📄 Licença

Este projeto faz parte de pesquisa acadêmica.

---

**Criado:** Dezembro 2024  
**Framework:** RecBole 1.2.1  
**GPU:** NVIDIA RTX 4090 (24GB)  
**Última Atualização:** 3 de Dezembro de 2024
