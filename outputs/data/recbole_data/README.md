# Conversão para Formato RecBole - IMPORTANTE

## Naming Convention

O RecBole espera que os arquivos de dados sigam esta convenção:
```
{data_path}/{dataset_name}/{dataset_name}.inter
```

## Nossa Estrutura

```
recbole_data/
├── realestate_slice1/
│   ├── realestate_slice1.inter         # Dados completos (train+test)
│   ├── realestate_slice1.train.inter   # Apenas treino
│   └── realestate_slice1.test.inter    # Apenas teste
├── realestate_slice2/
│   ├── realestate_slice2.inter
│   ├── realestate_slice2.train.inter
│   └── realestate_slice2.test.inter
...
```

## Converter Dados

```bash
# Converter todos os slices
make convert-recbole

# Ou manualmente:
python src/preprocessing/recbole_converter.py \
    --input data/sliding_window \
    --output recbole_data
```

**Importante:** O conversor cria automaticamente os arquivos com os nomes corretos (`realestate_slice{N}.*.inter`). **Não é necessário criar links simbólicos** manualmente.

## Reproduzibilidade

Para reproduzir a conversão do zero:

```bash
# 1. Deletar dados RecBole antigos (se existirem)
rm -rf recbole_data/

# 2. Reconverter
make convert-recbole

# 3. Verificar estrutura
ls -R recbole_data/
```

O processo é **totalmente reproduzível** - não requer etapas manuais ou links simbólicos.

## Troubleshooting

### Erro: "File realestate_slice1.inter not exist"

Se você vir este erro:
1. Verifique se os arquivos existem: `ls recbole_data/realestate_slice1/`
2. Verifique os nomes: Devem ser `realestate_slice1.*.inter`
3. Se não, reconverta: `make convert-recbole`

### Arquivos com nome errado (ex: realestate.inter)

Se os arquivos se chamam `realestate.inter` em vez de `realestate_slice1.inter`:
- **Solução:** Reconverta usando o código atualizado
- **NÃO** crie links simbólicos manualmente (não é reproduzível)

```bash
rm -rf recbole_data/
make convert-recbole
```
