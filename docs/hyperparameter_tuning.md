# Guia de Tuning de Hiperparâmetros - GRU4Rec

## Análise da Configuração Atual

### Problemas Identificados

1. **Epochs muito baixo (5)** - Modelo não converge completamente
2. **Apenas 1 layer GRU** - Capacidade limitada para padrões complexos
3. **Sem early stopping** - Risco de overfit ou treino desnecessário
4. **Comparado ao paper base**: Estamos sub-treinando (paper usa 20-50 epochs)

## Configurações Disponíveis

### 1. Original (Baseline)
```yaml
arquivo: src/configs/neural/gru4rec.yaml

epochs: 5
num_layers: 1
stopping_step: não habilitado
learning_rate: 0.001
dropout_prob: 0.3

Tempo: ~3-4 horas (todos modelos)
Uso: Testes rápidos, validação inicial
```

### 2. Melhor Esforço (Recomendado para começar)
```yaml
arquivo: src/configs/neural/gru4rec_improved.yaml

epochs: 15              # 3x mais
num_layers: 2           # Dobra capacidade
stopping_step: 10       # Adiciona proteção
learning_rate: 0.001    # Mantém
dropout_prob: 0.3       # Mantém

Tempo: ~6-8 horas
Melhoria esperada: +15-30%
Uso: Melhor custo-benefício
```

### 3. Alta Qualidade (Produção)
```yaml
arquivo: src/configs/neural/gru4rec_high_quality.yaml

epochs: 50              # Muitas epochs
num_layers: 2           # Boa capacidade
stopping_step: 10       # Para se não melhorar
learning_rate: 0.001    # Padrão
dropout_prob: 0.3       # Regularização
clip_grad_norm:         # Estabilidade
  max_norm: 5.0

Tempo: ~15-20 horas
Melhoria esperada: +30-50%
Uso: Modelo final, produção
```

### 4. Experimental (Pesquisa)
```yaml
arquivo: src/configs/neural/gru4rec_experimental.yaml

epochs: 100             # Máximo
num_layers: 3           # Alta capacidade
stopping_step: 15       # Mais paciência
learning_rate: 0.001    # Padrão
dropout_prob: 0.4       # Mais regularização
clip_grad_norm:         # Estabilidade
  max_norm: 5.0

Tempo: ~30-40 horas
Melhoria esperada: +50-80% (se tiver dados)
Uso: Exploração de limite superior
```

## Como Usar

### Testar config melhorada

```bash
# Editar run_experiments.py para usar nova config
# Ou copiar sobre a original:
cp src/configs/neural/gru4rec_improved.yaml src/configs/neural/gru4rec.yaml

# Rodar experimento
make run-all
```

### Testar apenas GRU4Rec com config específica

```bash
# 1. Backup da config original
cp src/configs/neural/gru4rec.yaml src/configs/neural/gru4rec_original.yaml

# 2. Usar config melhorada
cp src/configs/neural/gru4rec_improved.yaml src/configs/neural/gru4rec.yaml

# 3. Rodar só GRU4Rec
python src/run_experiments.py --models GRU4Rec --all-slices

# 4. Restaurar original (se quiser)
cp src/configs/neural/gru4rec_original.yaml src/configs/neural/gru4rec.yaml
```

## Comparação de Hiperparâmetros

| Parâmetro | Original | Melhor Esforço | Alta Qualidade | Experimental |
|-----------|----------|----------------|----------------|--------------|
| epochs | 5 | 15 | 50 | 100 |
| num_layers | 1 | 2 | 2 | 3 |
| stopping_step | - | 10 | 10 | 15 |
| dropout_prob | 0.3 | 0.3 | 0.3 | 0.4 |
| clip_grad_norm | - | - | 5.0 | 5.0 |
| Tempo estimado | 3-4h | 6-8h | 15-20h | 30-40h |
| Melhoria esperada | baseline | +15-30% | +30-50% | +50-80% |

## Impacto dos Hiperparâmetros

### Epochs
- **O que faz**: Número de passagens completas pelos dados
- **Impacto**: +++ (CRÍTICO)
- **Mais epochs**: Modelo aprende mais padrões
- **Muito poucos**: Modelo sub-treinado (nosso caso atual)
- **Muito muitos**: Overfit (memoriza treino)
- **Solução**: Early stopping previne overfit

### Num Layers
- **O que faz**: Profundidade da rede GRU
- **Impacto**: ++ (ALTO)
- **Mais layers**: Captura padrões mais complexos
- **1 layer**: Limitado a padrões simples
- **2-3 layers**: Bom para sessões
- **4+ layers**: Pode overfittar, difícil treinar

### Dropout
- **O que faz**: Desliga neurônios aleatoriamente (regularização)
- **Impacto**: + (MÉDIO)
- **Mais dropout (0.4-0.5)**: Previne overfit
- **Menos dropout (0.1-0.2)**: Mais capacidade
- **0.3**: Bom equilíbrio padrão

### Learning Rate
- **O que faz**: Tamanho do passo do gradiente
- **Impacto**: + (MÉDIO)
- **Alto (0.01)**: Treina rápido, pode ser instável
- **Médio (0.001)**: Bom equilíbrio (padrão)
- **Baixo (0.0001)**: Estável, treina devagar

### Stopping Step
- **O que faz**: Para treino se não melhorar por N epochs
- **Impacto**: ++ (ALTO)
- **Benefício**: Economiza tempo, previne overfit
- **Valor típico**: 10 (para epochs 50+)

### Batch Size
- **O que faz**: Quantidade de exemplos por atualização
- **Impacto**: + (MÉDIO)
- **4096**: Já otimizado para RTX 4090
- **Não mexer**: A menos que tenha problema de VRAM

## Estratégia Recomendada

### Fase 1: Validação (AGORA)
```
Config: Melhor Esforço
Tempo: 6-8 horas
Objetivo: Validar se melhoria é significativa
```

Se melhoria > 20% → Ir para Fase 2
Se melhoria < 10% → Problema não são hiperparâmetros

### Fase 2: Otimização
```
Config: Alta Qualidade
Tempo: 15-20 horas
Objetivo: Modelo para produção
```

### Fase 3: Exploração (Opcional)
```
Config: Experimental
Tempo: 30-40 horas
Objetivo: Limite superior de performance
```

## Outras Otimizações Possíveis

### Learning Rate Scheduler
```yaml
# Reduz LR ao longo do treino
# Não disponível no RecBole por padrão
# Implementação customizada necessária
```

### Optimizer
```yaml
# Trocar Adam por AdamW
# Melhor regularização
# Implementação customizada necessária
```

### Loss Function
```yaml
loss_type: 'BPR'  # Em vez de 'CE'
# Pode ser melhor para ranking
# Teste empírico necessário
```

## Monitoramento

Ao treinar com mais epochs, observe:

1. **Training loss**: Deve diminuir consistentemente
2. **Validation metric**: Deve melhorar
3. **Gap train/valid**: Se muito grande = overfit
4. **Early stopping**: Se ativar cedo demais, aumentar stopping_step

## Checklist de Execução

- [ ] Backup da config original
- [ ] Escolher configuração (Melhor Esforço recomendado)
- [ ] Atualizar gru4rec.yaml
- [ ] Rodar experimento completo ou só GRU4Rec
- [ ] Comparar resultados com baseline
- [ ] Decidir próximos passos
- [ ] Atualizar documentação com resultados

## Troubleshooting

### VRAM insuficiente
- Reduzir batch_size: 4096 → 2048
- Reduzir embedding_size: 256 → 128
- Reduzir num_layers: 3 → 2

### Treino muito lento
- Verificar se está usando GPU
- Verificar batch_size (maior = mais rápido)
- Considerar early stopping mais agressivo

### Não melhora com mais epochs
- Verificar learning rate (pode ser alto)
- Verificar dropout (pode estar muito alto)
- Verificar se dataset é grande o suficiente

### Overfit (valid piora)
- Aumentar dropout: 0.3 → 0.4
- Adicionar/reduzir stopping_step
- Reduzir capacidade (num_layers, embedding_size)
