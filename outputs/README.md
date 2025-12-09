# Outputs Directory

Este diretório contém todos os arquivos gerados pelo pipeline de experimentos.

## Estrutura

```
outputs/
├── data/               # Dados processados
│   ├── sliding_window/ # Splits temporais (Parquet)
│   └── recbole/        # Dados formatados para RecBole (.inter)
│
├── results/            # Resultados dos experimentos
│   ├── raw_results.csv
│   └── aggregated_results.csv
│
├── models/             # Modelos treinados salvos
│   └── [checkpoints por modelo/slice]
│
└── logs/               # Logs de treinamento
    └── tensorboard/    # Logs para TensorBoard
```

## Gitignore

O conteúdo deste diretório é ignorado pelo git (exceto os READMEs).
Para compartilhar resultados, use os arquivos CSV em `results/`.

## Limpeza

Para limpar todos os outputs gerados:

```bash
# Manter estrutura, remover conteúdo
rm -rf outputs/data/sliding_window/*
rm -rf outputs/data/recbole/*
rm -rf outputs/results/*.csv
rm -rf outputs/models/*
rm -rf outputs/logs/tensorboard/*
```
