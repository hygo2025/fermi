Com base no artigo analisado, segue um resumo técnico detalhado, com foco aprofundado na metodologia e nos algoritmos, sem a utilização de citações diretas.

---

# Benchmark em Larga Escala para Recomendações Baseadas em Sessão no Domínio Jurídico

Este estudo estabelece o primeiro benchmark dedicado a sistemas de recomendação baseados em sessão (*session-based recommendation*) aplicados ao contexto jurídico. O trabalho introduz o dataset **JusBrasilRec**, extraído da plataforma Jusbrasil, e avalia como diferentes famílias de algoritmos lidam com as especificidades da navegação de usuários em busca de informação legal.

## 1. Metodologia Experimental

A metodologia foi desenhada para simular um cenário realista de recomendação, onde o sistema deve prever o interesse futuro do usuário baseando-se apenas nas interações anônimas da sessão atual.

### 1.1 Coleta e Tratamento de Dados (JusBrasilRec)
* **Fonte:** Logs de interação de usuários anônimos e logados da plataforma Jusbrasil.
* **Período:** 30 dias de dados coletados entre fevereiro e março de 2021.
* **Definição de Sessão:** Foi utilizada a regra padrão de inatividade. Uma sessão é encerrada (timeout) após **30 minutos** sem interação do usuário. Isso permite tratar todas as sessões como anônimas, independente de o usuário estar logado ou não.
* **Filtragem:** Foram removidas sessões com apenas 1 interação (sem alvo para predição) ou com mais de 50 interações (possíveis robôs ou anomalias). Itens que apareceram menos de 5 vezes no dataset também foram descartados.

### 1.2 Protocolo de Avaliação: Janela Deslizante (Sliding Window)
Para respeitar a ordem cronológica dos dados e evitar o vazamento de informações do futuro para o passado, os autores não usaram validação cruzada aleatória (*random k-fold*). Em vez disso, aplicaram o protocolo de janela deslizante:
1.  O dataset de 30 dias foi dividido em **5 fatias (slices)** temporais.
2.  Cada fatia compreende **6 dias**:
    * **Treino:** Os primeiros 5 dias da fatia.
    * **Teste:** O dia seguinte (6º dia).
3.  Os modelos são treinados e testados sequencialmente nessas fatias e a média das métricas é reportada.

### 1.3 Cenários de Avaliação
O estudo avaliou os algoritmos em duas tarefas distintas:
1.  **Predição do Próximo Item (Next Item):** O objetivo é prever exatamente qual será o documento imediatamente seguinte que o usuário acessará.
    * *Métricas:* Hit Rate (HR@10) e Mean Reciprocal Rank (MRR@10).
2.  **Predição do Restante da Sessão (Rest of Session):** Dado o início de uma sessão, o objetivo é prever todos os itens subsequentes até o fim dela.
    * *Métricas:* Precision, Recall, MAP (Mean Average Precision) e NDCG (Normalized Discounted Cumulative Gain).

Além das métricas de acurácia, foram medidas a **Cobertura (Coverage)** (quanto do catálogo o algoritmo consegue recomendar) e o **Viés de Popularidade** (tendência do algoritmo em sugerir apenas os itens mais acessados).

---
## 2. Algoritmos e Modelos Avaliados

O benchmark comparou 21 modelos de recomendação baseada em sessão, além de um modelo de conteúdo para base de comparação. Eles foram categorizados em cinco famílias baseadas em seus princípios de funcionamento.

### A. Modelos Não-Personalizados (Baselines)
Servem como linha de base mínima. Não aprendem padrões sequenciais complexos.
* **Random:** Seleciona itens aleatoriamente do catálogo.
* **POP (Popularity):** Recomenda os itens com maior frequência global no dataset de treino.
* **RPOP (Recent Popularity):** Similar ao POP, mas considera apenas a popularidade dos itens no dia atual (filtro de recência).
* **SPOP (Session Popularity):** Recomenda os itens mais populares dentro da própria sessão atual. Em caso de empate ou sessões curtas, utiliza a popularidade global como desempate.

### B. Modelos Baseados em Mineração de Padrões (Pattern Mining)
Focam na extração de regras de co-ocorrência simples entre itens.
* **AR (Association Rules):** Simplificação das regras de associação. Calcula a frequência com que um item $j$ aparece nas sessões após um item $i$.
* **Markov:** Baseado em Cadeias de Markov de primeira ordem. Calcula a probabilidade de transição direta de um item para o próximo, baseando-se na sequência imediata.
* **SR (Sequential Rules):** Uma evolução do AR que considera a distância entre os itens dentro da sessão. Aplica uma função de decaimento: quanto mais distantes dois itens estiverem na sessão, menor o peso da regra de associação entre eles.

### C. Modelos de Vizinhos Mais Próximos (Nearest Neighbors - KNN)
Estes modelos assumem que sessões similares (ou itens similares) levam a recomendações similares.
* **IKNN (Item-KNN):** Foca no *último item* da sessão atual. Encontra itens similares a este último item baseando-se na co-ocorrência deles em outras sessões (usando similaridade de cosseno).
* **SKNN (Session-KNN):** Compara a *sessão inteira* atual com todas as sessões passadas de treino. Encontra as $k$ sessões mais parecidas (vizinhos) e recomenda os itens que aparecem nelas.
* **VSKNN (Vector Multiplication Session-KNN):** Variante do SKNN. Ao comparar sessões, dá pesos diferentes aos itens da sessão atual usando uma função de decaimento linear. Itens clicados mais recentemente na sessão têm maior influência na busca pelos vizinhos.
* **STAN (Sequence and Time-aware Neighborhood):** Um dos modelos mais avançados desta categoria. Considera três fatores na ponderação: (1) a posição do item na sessão atual, (2) a recência da sessão vizinha (sessões de teste mais próximas temporalmente das de treino têm mais peso) e (3) a posição do item recomendado na sessão vizinha.
* **VSTAN:** Uma fusão do VSKNN e STAN. Incorpora todas as ponderações do STAN, mas adiciona um esquema de pontuação sequencial e uma ponderação IDF (Inverse Document Frequency) para penalizar itens que aparecem em quase todas as sessões (muito populares).

### D. Modelos de Fatoração (Factorization Models)
Adaptam técnicas clássicas de fatoração de matriz para o contexto de sessões, tratando a sessão como se fosse um "usuário".
* **BPRMF:** Utiliza o critério de *Bayesian Personalized Ranking*. O vetor latente da "sessão" é calculado como a média dos vetores latentes dos itens que a compõem.
* **FPMC (Factorized Personalized Markov Chains):** Combina fatoração de matriz com Cadeias de Markov. Usa um tensor de três dimensões para modelar transições entre itens e preferências do usuário (aqui, sessão).
* **FISM (Factored Item Similarity Models):** Aprende uma matriz de similaridade item-item como o produto de duas matrizes de baixa dimensão. Não modela transições sequenciais explicitamente, mas sim a similaridade agregada.
* **FOSSIL:** Um híbrido que tenta combinar o melhor do FISM (preferências de longo prazo/gerais) com o FPMC (sequencialidade de curto prazo).
* **SMF (Session-based Matrix Factorization):** Similar ao FPMC, mas substitui explicitamente o vetor de usuário por um "vetor de preferência de sessão", otimizado para cenários onde não há histórico longo do usuário (cold-start).

### E. Modelos de Redes Neurais (Deep Neural Networks)
Utilizam arquiteturas de aprendizado profundo para capturar dependências não-lineares e sequenciais complexas.
* **GRU4Rec:** O primeiro modelo a usar Redes Neurais Recorrentes (RNN) com unidades GRU para recomendação baseada em sessão. Processa a sessão item a item para atualizar o estado oculto e prever o próximo.
* **NARM (Neural Attentive Recommendation Machine):** Melhora o GRU4Rec adicionando um mecanismo de **atenção**. Ele possui um codificador híbrido que considera tanto o comportamento sequencial quanto o propósito principal da sessão, focando mais nos itens relevantes.
* **STAMP:** Abandona as RNNs em favor de uma arquitetura baseada inteiramente em **atenção** (similar a Transformers simplificados). Mantém uma memória de longo prazo (interesses gerais da sessão) e curto prazo (último clique), ponderando-os para fazer a predição.
* **SGNN (Session-based Graph Neural Network):** Modela a sessão como um **grafo**, onde os itens são nós e as transições são arestas. Usa GNNs para aprender embeddings complexos que capturam como os itens se relacionam estruturalmente na sessão.

### F. Modelo Comparativo: Content-Based
Para verificar se o comportamento supera o conteúdo, foi testado um modelo clássico de **Bag-of-Words (TF-IDF)**.
* **Metodologia:** O texto dos documentos jurídicos é vetorizado via TF-IDF. O último item da sessão serve como "query". O sistema recomenda os 10 documentos com maior similaridade de cosseno em relação a esse último item.

---

## 3. Resumo dos Principais Resultados

* **Dominância dos Vizinhos Próximos:** Surpreendentemente, os modelos baseados em KNN (**STAN, VSTAN, VSKNN**) apresentaram o melhor desempenho geral em acurácia, superando modelos complexos de Deep Learning em várias métricas.
* **Redes Neurais Competitivas:** Entre as redes neurais, o **NARM** obteve o melhor desempenho, ficando muito próximo (e às vezes superando) os modelos KNN.
* **Comportamento > Conteúdo:** Os modelos baseados apenas em interação (session-based) superaram significativamente o modelo baseado em conteúdo (TF-IDF), provando que a sequência de navegação carrega sinais de preferência mais fortes que a similaridade textual pura no domínio jurídico.
* **Fatoração Ineficiente:** Modelos de fatoração (como FISM e FOSSIL) tiveram desempenho ruim, atribuído à alta esparsidade dos dados e ao tamanho curto das sessões.
* **Navegação Focada:** A análise comparativa mostrou que o domínio jurídico possui uma navegação mais "focada" do que e-commerce ou música. Isso significa que, a partir de um documento jurídico, o universo de prováveis próximos documentos é menor, o que tende a facilitar a acurácia dos algoritmos neste domínio específico.