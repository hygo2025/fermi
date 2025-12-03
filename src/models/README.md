# Session-Rec Wrappers

Este projeto usa wrappers para adaptar algoritmos da lib `session-rec` sem modificar o código original.

## Estrutura

```
src/models/knn/
├── iknn.py    # Wrapper para ItemKNN
└── sknn.py    # Wrapper para ContextKNN

session-rec-lib/algorithms/
└── models -> ../../src/models  # Symlink
```

## Por que usar wrappers?

1. **Correção de bugs**: SKNN tinha bug onde `sessions_for_item()` retornava `None`
2. **Adaptação de interfaces**: IKNN tinha assinatura incompatível `fit(data)` vs `fit(data, test)`
3. **Manutenção da lib original**: Não modificamos o código da lib externa

## Como funciona?

1. Config usa: `models.knn.sknn.ContextKNN`
2. `run_config.py` adiciona prefixo: `algorithms.models.knn.sknn.ContextKNN`
3. Symlink resolve: `session-rec-lib/algorithms/models` → `src/models`
4. Wrapper é carregado e corrige problemas

## Setup

O symlink é criado automaticamente ao rodar:
```bash
make install-benchmark
```

Ou manualmente:
```bash
./scripts/setup_wrappers.sh
```

## Wrappers Implementados

### IKNN (ItemKNN)
**Problema**: Assinatura `fit(data)` incompatível  
**Solução**: Wrapper aceita `fit(data, test=None)` e ignora `test`

### SKNN (ContextKNN)
**Problema**: `sessions_for_item()` retorna `None` para itens não vistos  
**Solução**: Wrapper retorna `set()` vazio em vez de `None`

## Referências

- Documentação completa: `artigo/knn_wrapper_documentation.md`
- Pattern Mining: `artigo/pattern_mining_config.md`
