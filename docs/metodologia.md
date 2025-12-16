# Metodologia de Benchmark para Recomendação Baseada em Sessão

Este documento resume a metodologia do paper de Domingues et al. (2024) sobre benchmarking de sistemas de recomendação baseados em sessão no domínio jurídico. O estudo apresenta o dataset JusBrasilRec e compara diferentes tipos de algoritmos de recomendação.

## Metodologia Experimental

A ideia é simular um cenário real onde o sistema precisa prever o próximo interesse do usuário baseado apenas nas interações da sessão atual.

### Coleta e Tratamento de Dados (JusBrasilRec)

* **Fonte:** Logs da plataforma Jusbrasil com usuários anônimos e logados
* **Período:** 30 dias (fev-mar 2021)
* **Definição de Sessão:** Timeout de 30 minutos de inatividade. Isso permite tratar todas as sessões como anônimas, independente do login.
* **Filtragem:** Removidas sessões com 1 interação apenas ou mais de 50 (possíveis bots). Itens com menos de 5 ocorrências também foram descartados.

### Protocolo de Avaliação: Janela Deslizante (Sliding Window)

Para respeitar a ordem temporal dos dados (sem usar informações do futuro no treino), foi usado o protocolo de janela deslizante ao invés de k-fold tradicional:

1. Dataset de 30 dias dividido em 5 fatias temporais
2. Cada fatia tem 6 dias:
   - Treino: primeiros 5 dias
   - Teste: 6º dia
3. Modelos são treinados e testados em cada fatia, reportando a média das métricas

### Cenários de Avaliação

Dois tipos de tarefas foram avaliadas:

1. **Predição do Próximo Item (Next Item):** Prever qual será o próximo documento que o usuário vai acessar.
   - Métricas: Hit Rate (HR@10) e Mean Reciprocal Rank (MRR@10)

2. **Predição do Restante da Sessão (Rest of Session):** Dado o início da sessão, prever todos os itens subsequentes.
   - Métricas: Precision, Recall, MAP e NDCG

Além disso, foram medidas a Cobertura (porção do catálogo recomendada) e o Viés de Popularidade (tendência de recomendar só os populares).

---
## Algoritmos e Modelos Avaliados

O benchmark comparou 21 modelos de recomendação baseada em sessão, mais um modelo de conteúdo. Os modelos foram divididos em cinco categorias:

### A. Modelos Não-Personalizados (Baselines)

Servem como baseline mínimo, sem aprender padrões sequenciais complexos.
* **Random:** Seleciona itens aleatoriamente do catálogo.
* **POP (Popularity):** Recomenda os itens com maior frequência global no dataset de treino.
* **RPOP (Recent Popularity):** Similar ao POP, mas considera apenas a popularidade dos itens no dia atual (filtro de recência).
* **SPOP (Session Popularity):** Recomenda os itens mais populares dentro da própria sessão atual. Em caso de empate ou sessões curtas, utiliza a popularidade global como desempate.

### B. Modelos Baseados em Mineração de Padrões (Pattern Mining)

Extraem regras de co-ocorrência simples entre itens.

* **AR (Association Rules):** Calcula a frequência com que um item j aparece após um item i nas sessões.
* **Markov:** Usa Cadeias de Markov de primeira ordem para calcular probabilidades de transição entre itens consecutivos.
* **SR (Sequential Rules):** Evolução do AR que considera distância entre itens. Aplica decaimento: quanto mais distantes, menor o peso.

### C. Modelos de Vizinhos Mais Próximos (Nearest Neighbors - KNN)

Assumem que sessões (ou itens) similares levam a recomendações similares.

* **IKNN (Item-KNN):** Foca no último item da sessão. Encontra itens similares usando co-ocorrência (similaridade de cosseno).
* **SKNN (Session-KNN):** Compara a sessão inteira com sessões passadas. Recomenda itens das k sessões mais parecidas.
* **VSKNN (Vector Multiplication Session-KNN):** Variante do SKNN que dá pesos diferentes aos itens da sessão usando decaimento linear. Itens recentes têm mais peso.
* **STAN (Sequence and Time-aware Neighborhood):** Modelo mais avançado. Considera: (1) posição do item na sessão, (2) recência temporal da sessão vizinha, (3) posição do item recomendado.
* **VSTAN:** Combina VSKNN e STAN. Adiciona pontuação sequencial e ponderação IDF para penalizar itens muito populares.

### D. Modelos de Fatoração (Factorization Models)

Adaptam fatoração de matriz para sessões, tratando a sessão como um "usuário".

* **BPRMF:** Usa Bayesian Personalized Ranking. O vetor da sessão é a média dos vetores dos itens.
* **FPMC (Factorized Personalized Markov Chains):** Combina fatoração de matriz com Markov. Usa tensor 3D para modelar transições e preferências.
* **FISM (Factored Item Similarity Models):** Aprende matriz de similaridade item-item como produto de duas matrizes de baixa dimensão. Não modela sequência explicitamente.
* **FOSSIL:** Híbrido que combina FISM (preferências gerais) com FPMC (sequencialidade).
* **SMF (Session-based Matrix Factorization):** Similar ao FPMC, mas usa vetor de preferência de sessão ao invés de usuário. Bom para cold-start.

### E. Modelos de Redes Neurais (Deep Neural Networks)

Usam aprendizado profundo para capturar dependências não-lineares e sequenciais complexas.

* **GRU4Rec:** Primeiro modelo a usar RNN com GRU para recomendação de sessão. Processa a sessão item a item atualizando o estado oculto.
* **NARM (Neural Attentive Recommendation Machine):** Melhora o GRU4Rec com mecanismo de atenção. Tem um codificador híbrido que considera comportamento sequencial e propósito principal da sessão.
* **STAMP:** Usa só atenção (tipo Transformers simplificado) sem RNN. Mantém memória de longo prazo (interesses gerais) e curto prazo (último clique).
* **SGNN (Session-based Graph Neural Network):** Modela a sessão como grafo (itens = nós, transições = arestas). Usa GNN para aprender embeddings que capturam relações estruturais.

### F. Modelo Comparativo: Content-Based

Para verificar se comportamento supera conteúdo, testaram um modelo clássico de Bag-of-Words (TF-IDF).

* **Metodologia:** Texto dos documentos vetorizado via TF-IDF. O último item da sessão serve como query. Recomenda os 10 documentos com maior similaridade de cosseno.

---

## Principais Resultados

* **KNN dominou:** Os modelos KNN (STAN, VSTAN, VSKNN) tiveram o melhor desempenho geral, superando até os modelos de Deep Learning em várias métricas.
* **Redes Neurais competitivas:** Entre as neurais, o NARM teve o melhor resultado, chegando bem perto (e às vezes superando) os KNN.
* **Comportamento > Conteúdo:** Modelos baseados em interação superaram de longe o modelo baseado em conteúdo (TF-IDF). A sequência de navegação carrega mais informação que similaridade textual pura.
* **Fatoração não funcionou bem:** Modelos como FISM e FOSSIL tiveram desempenho fraco, provavelmente pela alta esparsidade dos dados e sessões curtas.
* **Navegação focada:** O domínio jurídico tem navegação mais "focada" que e-commerce ou música. A partir de um documento, o universo de próximos documentos prováveis é menor, facilitando a acurácia.