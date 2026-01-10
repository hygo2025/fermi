# Hyperparameter Search Spaces

Este diretório reúne os **espaços de busca** utilizados pelo `hyperparameter_tuning.py`.
Cada arquivo YAML descreve como o HyperTuning do RecBole deve amostrar um
hiperparâmetro específico do modelo.

## Formato

```yaml
learning_rate:
  type: loguniform
  min: 0.0001
  max: 0.01

hidden_size:
  type: choice
  values: [128, 256, 512]
```

Tipos suportados:

| Tipo       | Descrição                                   |
|------------|---------------------------------------------|
| `choice`   | Seleciona um valor discreto da lista         |
| `uniform`  | Amostra contínua uniforme (`min`, `max`)     |
| `loguniform` | Amostra contínua log-uniforme (valores > 0) |
| `randint`  | Inteiro no intervalo `[min, max)`            |
| `quniform` | Amostra contínua com passo fixo `q`          |

Adicione um novo arquivo seguindo o padrão `<modelo>_space.yaml` (ex.: `gru4rec_space.yaml`)
para disponibilizar o espaço de busca a um modelo.
