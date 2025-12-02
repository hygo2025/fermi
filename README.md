# Fermi - Session-Based Recommendation Benchmark

Benchmark de sistemas de recomendação baseados em sessão para o domínio imobiliário, inspirado no artigo:

**"A large scale benchmark for session-based recommendations on the legal domain"**  
Domingues et al. (2025) - Artificial Intelligence and Law

## 📋 Visão Geral

Este projeto implementa e avalia múltiplos modelos de recomendação baseados em sessão usando dados reais de interações de usuários com listagens de imóveis. O objetivo é predizer o próximo imóvel que um usuário vai interagir baseado na sequência de interações da sessão atual.

### Diferenças do Artigo Original

| Aspecto | Artigo (Jusbrasil) | Nosso Benchmark |
|---------|-------------------|-----------------|
| Domínio | Legal (documentos jurídicos) | Imobiliário (listings) |
| Itens | Documentos | Imóveis |
| Framework | session-rec | session-rec (mesmo) |
| Métricas | Recall@K, MRR@K, Coverage | Recall@K, MRR@K, Coverage (mesmas) |

## 🗂️ Estrutura do Projeto

```
fermi/
├── src/                        # 🔬 Benchmark implementation
│   ├── run_benchmark.py        # Main execution script
│   └── configs/                # Experiment configurations
│
├── data/                       # 📊 Data processing scripts
│   ├── prepare_dataset.py      # Spark-based data preparation
│   └── convert_to_session_rec.py  # Format conversion to session-rec
│
├── session-rec-lib/            # 🔧 Session-rec framework (git submodule)
│   ├── algorithms/             # All models implementations
│   └── evaluation/             # Metrics and evaluation
│
├── scripts/                    # 🛠️ Installation & utilities
│   └── install.sh              # Automated installation
│
├── utils/                      # 💡 Helper utilities
│   └── spark_session.py        # Spark configuration
│
├── .env                        # Environment variables (BASE_PATH, JAVA_HOME)
├── requirements.txt            # Python dependencies
├── Makefile                    # Common commands
└── README.md                   # This file
```

## ⚙️ Configuração

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
BASE_PATH=/home/hygo2025/Documents/data
JAVA_HOME=/opt/jdk/amazon-corretto-21
PYTHONUNBUFFERED=1
```

**Nota:** `BASE_PATH` aponta para onde seus dados brutos estão armazenados.

## 🚀 Início Rápido

### 1. Instalação

```bash
# Clone com submódulos
git clone --recursive <repository-url>
cd fermi

# Se já clonou sem --recursive
git submodule update --init --recursive

# Instale dependências Python
pip install -r requirements.txt
```

O projeto usa:
- ✅ **session-rec-lib** como submódulo Git (fork com correções Python 3.9+)
- ✅ Todas as dependências via `requirements.txt`

### 2. Preparar Dados

```bash
# Preparar dataset (14 dias de dados)
python data/prepare_dataset.py \
    --start-date 2024-03-01 \
    --end-date 2024-03-15
```

### 3. Executar Benchmark

```bash
# Testar com modelo baseline (POP)
python src/run_session_rec.py --config src/configs/pop_only.yml

# Executar benchmark completo
python src/run_session_rec.py --config src/configs/session_rec_config.yml
```

## 📊 Dados

Os dados brutos estão em `/home/hygo2025/Documents/data/processed_data/`:
- **events/** - Eventos de usuários (~25M eventos, 182 dias)
- **listings/** - Catálogo de imóveis (187k imóveis)

O pipeline de preparação:
1. Filtra eventos por período
2. Cria sessões (30min de inatividade)
3. Remove sessões curtas (<2 eventos) e itens raros (<5 ocorrências)
4. Split temporal (80% train, 10% val, 10% test)
5. Converte para formato session-rec (tab-separated)

## 🔧 Framework: Session-Rec

Utilizamos o **session-rec**, mesmo framework usado no artigo original:

- **Fork Python 3.9+:** https://github.com/hygo2025/session-rec-3-9
- **Branch:** `python39-compatibility`
- **Original:** https://github.com/rn5l/session-rec

### Correções Aplicadas no Fork

1. ✅ `time.clock()` → `time.perf_counter()` (removido no Python 3.8)
2. ✅ `yaml.load()` → `yaml.load(Loader=FullLoader)` (segurança)
3. ✅ `Pop.fit()` signature fix
4. ✅ Telegram notifications desabilitadas

### Por Que Session-Rec?

- ✅ Mesmo framework do artigo (comparabilidade)
- ✅ 20+ modelos session-based implementados
- ✅ Métricas padronizadas
- ✅ Benchmark estabelecido na literatura

## 📊 Modelos Implementados

### Baselines
- **pop** - Popularity-based recommender
- **ar** - Association Rules
- **sr** - Sequential Rules
- **markov** - Markov Chains

### KNN-based
- **iknn** - Item k-Nearest Neighbors
- **sknn** - Session-based KNN
- **vsknn** - Vector Multiplication Session-based KNN
- **stan** - Sequence and Time-aware Neighborhood

### Deep Learning
- **gru4rec** - Gated Recurrent Units for Recommendations
- **narm** - Neural Attentive Recommendation Machine
- **STAMP** - Short-Term Attention Memory Priority

## 📈 Métricas de Avaliação

- **Recall@K** - Taxa de acerto nas top-K recomendações
- **MRR@K** - Mean Reciprocal Rank
- **Coverage** - Cobertura do catálogo

Com K ∈ {5, 10, 20}

## 🔬 Pipeline Completo

1. **Preparação:** Filtra eventos → cria sessões → split temporal
2. **Conversão:** CSV → formato session-rec (tab-separated)
3. **Treinamento:** Treina modelos com dados de treino
4. **Avaliação:** Next-item prediction nas sessões de teste
5. **Análise:** Comparação de métricas entre modelos

## 🛠️ Comandos Úteis

```bash
# Ver comandos disponíveis
make help

# Instalar dependências do projeto
make install

# Limpar ambiente
make clean

# Rodar teste rápido
make test-pop
```

## 🔍 Troubleshooting

### Erro: `time.clock()` not found

**Solução:** Use o fork Python 3.9+ compatível (já incluído no install.sh)

### Erro: `fit() takes 2 positional arguments but 3 were given`

**Solução:** Fork já contém correção. Reexecute `./scripts/install.sh`

### Dados carregam muito lento

**Solução:** 
- Use `pyarrow` para leitura de Parquet
- Aplique filtros de data ao carregar events
- Considere usar apenas um subset dos dados para testes

### Session-rec não encontrado

**Solução:** 
```bash
export PYTHONPATH=/home/hygo2025/Development/projects/fermi:$PYTHONPATH
```

## 📚 Referências

**Artigo Principal:**
```
Domingues, M. A., de Moura, E. S., Marinho, L. B., & da Silva, A. (2025).
A large scale benchmark for session-based recommendations on the legal domain.
Artificial Intelligence and Law, 33, 43-78.
DOI: 10.1007/s10506-023-09378-3
```

**Session-Rec Framework:**
```
Ludewig, M., & Jannach, D. (2018).
Evaluation of session-based recommendation algorithms.
User Modeling and User-Adapted Interaction, 28(4-5), 331-390.
```

## 📄 Licença

Este projeto é parte de pesquisa acadêmica.

---

**Criado em:** 02 de dezembro de 2024  
**Baseado em:** Domingues et al. (2025)  
**Framework:** [session-rec](https://github.com/hygo2025/session-rec-3-9) (fork Python 3.9+)  
**Última atualização:** 02 de dezembro de 2024
